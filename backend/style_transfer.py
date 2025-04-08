import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import time
import os
import io

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def load_image_from_bytes(image_bytes, imsize=512):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(1, 1, 1) if x.size(0) == 1 else x)
    ])
    
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Content Loss: Measures content similarity
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # Detach the target content from the tree used to dynamically compute gradients
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# Gram matrix calculation for style representation
def gram_matrix(input):
    batch_size, n_channels, height, width = input.size()
    features = input.view(batch_size * n_channels, height * width)
    G = torch.mm(features, features.t())
    # Normalize by total number of elements
    return G.div(batch_size * n_channels * height * width)

# Style Loss: Measures style similarity using Gram matrices
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.weight = 1.0  # Default weight for this layer

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# Normalization layer for VGG compatibility
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # View the mean and std as 1x3x1x1 tensors
        self.mean = mean.clone().detach().view(-1, 1, 1).to(device)
        self.std = std.clone().detach().view(-1, 1, 1).to(device)

    def forward(self, img):
        # Normalize img
        return (img - self.mean) / self.std

# Build model with content and style losses
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'],
                               layer_weights=None):
    normalization = Normalization(normalization_mean, normalization_std)
    
    # Set default layer weights if not provided
    if layer_weights is None:
        layer_weights = {layer: 1.0 for layer in style_layers}

    # Lists to keep track of losses
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
            
            # Apply customized layer weight
            style_loss.weight = layer_weights.get(name, 1.0)
            
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

# Optimization loop for style transfer
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1,
                       layer_weights=None, progress_callback=None):
    """Run the style transfer."""
    num_steps = min(num_steps, 400)
    
    print('Building the style transfer model...')
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, 
        style_img, content_img, 
        layer_weights=layer_weights
    )

    # We want to optimize the input image only
    input_img.requires_grad_(True)
    model.eval()  # We don't need gradients for the model parameters
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_img])
    best_img = None
    best_loss = float('inf')
    prev_loss = float('inf')
    current_step = 0

    start_time = time.time()

    # Function to be used with optimizer
    def closure():
        nonlocal current_step
        # Correct the values of updated input image
        with torch.no_grad():
            input_img.clamp_(0, 1)

        optimizer.zero_grad()
        model(input_img)
        style_score = 0
        content_score = 0

        for sl in style_losses:
            # Apply per-layer weight
            style_score += sl.loss * sl.weight
            
        for cl in content_losses:
            content_score += cl.loss

        style_score *= style_weight
        content_score *= content_weight

        loss = style_score + content_score
        loss.backward()

        current_step += 1
        if current_step % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Iteration: {current_step}, Style Loss: {style_score.item():.2f}, Content Loss: {content_score.item():.2f}, Total Loss: {loss.item():.2f}, Time: {elapsed:.1f}s")
            
            if progress_callback:
                progress = {
                    'iteration': current_step,
                    'style_loss': style_score.item(),
                    'content_loss': content_score.item(),
                    'elapsed_time': elapsed
                }
                progress_callback(progress)

        # Save best result so far
        nonlocal best_loss, best_img, prev_loss
        current_loss = loss.item()
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_img = input_img.clone()
        
        # Update previous loss for next iteration
        prev_loss = current_loss
        return loss

    # Run optimization with early stopping
    while current_step < num_steps:
        optimizer.step(closure)
        
        # Check stopping conditions after minimum iterations
        if current_step >= 50 and prev_loss > 1000:
            print(f"Stopping early at iteration {current_step} due to high loss: {prev_loss:.2f}")
            break

    # A final correction
    with torch.no_grad():
        input_img.clamp_(0, 1)

    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Best loss achieved: {best_loss:.2f}")

    # Return both the final and best image (often the same)
    return input_img, best_img, best_loss

# Save tensor as image
def save_image(tensor, path):
    image = tensor.cpu().clone()
    image = image.squeeze(0)  # Remove batch dimension
    image = transforms.ToPILImage()(image)
    image.save(path)
    return image

# Main style transfer function
def transfer_style(content_path, style_path, output_path, style_weight=1000000, 
                  content_weight=1, num_steps=300, layer_weights=None,
                  progress_callback=None):
    """
    Perform style transfer and save the result
    
    Args:
        content_path: Path to content image
        style_path: Path to style image
        output_path: Where to save the output image
        style_weight: Weight for style loss
        content_weight: Weight for content loss
        num_steps: Number of optimization steps
        layer_weights: Dictionary of weights for each style layer
        progress_callback: Function to call for progress updates
    
    Returns:
        Tuple of (output_path, best_loss)
    """
    # Load images
    content_img = image_loader(content_path)
    style_img = image_loader(style_path)
    
    # Start with content image for faster convergence
    input_img = content_img.clone()

    # Load VGG19 for feature extraction
    cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
    
    # Mean and std for normalization (from ImageNet)
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    # Run style transfer
    output, best_output, best_loss = run_style_transfer(
        cnn, 
        cnn_normalization_mean, 
        cnn_normalization_std,
        content_img, 
        style_img, 
        input_img,
        num_steps=num_steps,
        style_weight=style_weight,
        content_weight=content_weight,
        layer_weights=layer_weights,
        progress_callback=progress_callback
    )
    
    # Save result and return path
    save_image(best_output, output_path)
    
    return output_path, best_loss 