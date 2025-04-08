# Neural Style Transfer Web Application

This is a full-stack web application for neural style transfer. It allows users to upload content and style images, then applies style transfer to create artistic compositions.

## Features

- **Content and Style Image Upload**: Drag-and-drop or file selection
- **Style Transfer Controls**: Adjust style weight, content weight, and layer weights
- **Real-time Progress Tracking**: Monitor the optimization process
- **Download Results**: Save your generated artwork

## Project Structure

- **Frontend**: Angular 17 application with TypeScript, TailwindCSS
- **Backend API**: FastAPI for REST endpoints and async job handling
- **Style Transfer Model**: PyTorch implementation with VGG19 feature extraction

## Getting Started

### Running the Backend

1. Navigate to the backend directory:

```bash
cd backend
```

2. Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

3. Start the backend server:

```bash
python run.py
```

The backend API will be available at http://localhost:8000

### Running the Frontend

1. From the project root:

```bash
npm install
```

2. Start the development server:

```bash
npm start
```

The web application will be available at http://localhost:4200

## Using the Application

1. Navigate to the web interface
2. Upload a content image (the photo you want to transform)
3. Upload a style image (artwork that provides the style)
4. Adjust parameters as desired:
   - Higher style weight emphasizes artistic style
   - Higher content weight preserves more original content
   - Advanced settings allow fine-tuning layer weights
5. Click "Apply Style Transfer" to process
6. View the progress and download the final image when complete

## Technologies Used

- **Frontend**: Angular, TypeScript, TailwindCSS, RxJS
- **Backend**: FastAPI, PyTorch, NumPy
- **Model**: VGG19 pretrained network for style transfer

## License

MIT
