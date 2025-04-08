import uvicorn
import os

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

if __name__ == "__main__":
    print("Starting Neural Style Transfer API")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 