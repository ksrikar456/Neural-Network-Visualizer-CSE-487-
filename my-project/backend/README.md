# Neural Style Transfer Backend

This backend implements a FastAPI service that provides style transfer capabilities using PyTorch.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the server:

```bash
python run.py
```

The server will be available at http://localhost:8000

> Note: When running for the first time, PyTorch will automatically download the VGG19 model weights, which may take a few minutes depending on your internet connection.

## API Endpoints

- `POST /api/transfer` - Start a new style transfer job
- `GET /api/transfer/{job_id}` - Check status of a job
- `GET /api/health` - Health check endpoint

## Example Usage

1. Upload a content image and style image
2. Adjust style weight, content weight, and other parameters
3. Submit for processing
4. Poll for updates and view the result

## Architecture

- FastAPI for the REST API
- PyTorch for style transfer model
- VGG19 for feature extraction 