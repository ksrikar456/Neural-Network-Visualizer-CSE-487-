import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import time


from google.colab import drive
drive.mount('/content/drive')

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Resizing,Adding color, and reformatting to the Nueral network needs/expects based on what is was preprocessed  (Vg19)


# Image loading and preprocessing
def image_loader(image_path, imsize=512):
    loader = transforms.Compose([
        transforms.Resize(imsize),  # Scale imported image
        transforms.CenterCrop(imsize),  # Ensure square size
        transforms.ToTensor(),  # Transform into torch tensor
        transforms.Lambda(lambda x: x.repeat(1, 1, 1) if x.size(0) == 1 else x)  # Convert grayscale to RGB if needed
    ])

    image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
    # Add batch dimension (1, 3, h, w)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    # Convert tensor to PIL image
    image = tensor.cpu().clone()
    image = image.squeeze(0)  # Remove batch dimension
    image = transforms.ToPILImage()(image)

    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.axis('on')
    plt.show()

    return image


#Measures how much your generated image matches the content of the original photo.
#Looks at deep features from a specific layer in the VGG19 model (not just pixels).
#Uses MSE (mean squared error) between those features and the content image's features.
#Perserving the content of image
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # Detach the target content from the tree used to dynamically compute gradients
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

#Gram matrix and Style loss
#Measures how much your generated image matches the style of the style image.
#Uses the Gram matrix to measure texture/patterns/color relationships in feature maps.
#Compares those matrices using MSE too.
def gram_matrix(input):
    batch_size, n_channels, height, width = input.size()
    features = input.view(batch_size * n_channels, height * width)
    G = torch.mm(features, features.t())
    # Normalize by total number of elements
    return G.div(batch_size * n_channels * height * width)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach() #Computes and stores the style's Gram matrix, detached from the graph

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target) #For every forward pass, compares the style of the current image's features to the style target (again using MSE).
        return input

#Makes sure the input image is scaled the same way the VGG19 model expects (because it was trained on ImageNet).
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # View the mean and std as 1x3x1x1 tensors
        self.mean = mean.clone().detach().view(-1, 1, 1).to(device)
        self.std = std.clone().detach().view(-1, 1, 1).to(device)

    def forward(self, img):
        # Normalize img
        return (img - self.mean) / self.std



def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    normalization = Normalization(normalization_mean, normalization_std)

    # Build a sequential model with our content and style loss modules
    content_losses = []
    style_losses = []

    # Create a "sequential" module with added content/style loss layers
    model = nn.Sequential(normalization)

    i = 0  # Increment for each conv layer
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            # Replace in-place version with out-of-place
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        # Add content loss
        if name in content_layers:
            # Add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        # Add style loss
        if name in style_layers:
            # Add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

# Optimization loop
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    num_steps = min(num_steps, 400)

    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)


    # We want to optimize the input image only
    input_img.requires_grad_(True)
    model.eval()  # We don't need gradients for the model parameters
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_img]) #few optimizers

    print('Optimizing..')
    run = [0]
    best_img = None
    best_loss = float('inf')

    start_time = time.time()

    # Function to be used with optimizer
    def closure():
        # Correct the values of updated input image
        with torch.no_grad():
            input_img.clamp_(0, 1)

        optimizer.zero_grad()
        model(input_img)
        style_score = 0
        content_score = 0

        #print(style_losses)
        #print(content_losses)


        for sl in style_losses:
            style_score += sl.loss # mess w this
        for cl in content_losses:
            content_score += cl.loss # mess w this

        style_score *= style_weight
        content_score *= content_weight

        loss = style_score + content_score
        loss.backward()

        run[0] += 1
        if run[0] % 50 == 0:
            elapsed = time.time() - start_time
            print(f"run {run[0]}:")
            print(f'Style Loss : {style_score.item():.4f} Content Loss: {content_score.item():.4f}')
            print(f'Time elapsed: {elapsed:.1f}s')
            plt.figure(figsize=(8, 8))
            imshow(input_img.clone(), title=f'Iteration {run[0]}, Style Loss : {style_score.item():.4f} Content Loss: {content_score.item():.4f}')

        # Save best result so far
        nonlocal best_loss, best_img
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_img = input_img.clone()

        return style_score + content_score

    # Run optimization
    for _ in range(num_steps):
        optimizer.step(closure)

    # A final correction
    with torch.no_grad():
        input_img.clamp_(0, 1)

    print(f"Total time: {time.time() - start_time:.1f}s")

    # Return both the final and best image (often the same)
    return input_img, best_img


def main(content_path, style_path, num_steps=300):
    num_steps = min(num_steps, 400)

    # Load images
    content_img = image_loader(content_path)
    style_img = image_loader(style_path)

    # Create a white noise image or start with content image
    input_img = content_img.clone()  # Start with content image for faster convergence

    # Display the original images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    imshow(content_img, title='Content Image')

    plt.subplot(1, 2, 2)
    imshow(style_img, title='Style Image')

    # VGG19 for feature extraction
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # Mean and std for normalization
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # style_weight = 1e3
    # content_weight = 1

    style_weight = 1e4
    content_weight = 3

#How much  loss/how much they are pulling away during optimization
#Optimization pulls equally from both content and style
# style_weight = 1e5
# content_weight = 1

# style_weight = 1e6
# content_weight = 1

# style_weight = 1e7
# content_weight = 1

# style_weight = 1e6
# content_weight = 0


#extraa trying
# style_weight = 5e6
# content_weight = 1

# style_weight = 7e5
# content_weight = 1

# style_weight = 5e5
# content_weight = 1

# style_weight = 5e4
# content_weight = 2

# style_weight = 2e5
# content_weight = 1

# style_weight = 3e4
# content_weight = 2

# style_weight = 1e4
# content_weight = 3

# style_weight = 5e2
# content_weight = 1





    # Run style transfer
    output, best_output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                         content_img, style_img, input_img,
                                         num_steps=num_steps,
                                         style_weight=style_weight,
                                         content_weight=content_weight)

    # Display result
    plt.figure(figsize=(10, 10))
    imshow(output, title='Output Image')

    # Save result
    output_img = transforms.ToPILImage()(best_output.cpu().squeeze(0))
    output_img.save('style_transfer_result.jpg')

    return output_img

from google.colab import files
uploaded = files.upload()
content_path = next(iter(uploaded.keys()))  # Get the first uploaded file
uploaded = files.upload()
style_path = next(iter(uploaded.keys()))  # Get the second uploaded file
result = main(content_path, style_path, num_steps=300)