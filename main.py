"""
FastAPI Server for LLM Fine-tuned Models
Exposes all 5 models as REST API endpoints
Optimized for H100 GPU inference
"""
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.router import router as api_routes


# Initialize FastAPI app
app = FastAPI(
    title="LLM Fine-tuned Models API",
    description="API for HR, Finance, Sales, Healthcare, and Marketing fine-tuned models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(api_routes, prefix="/v1/api")

# Preload models on startup (optional - comment out if not needed)
@app.on_event("startup")
def preload_models():
    """Preload all models on startup for faster first requests"""
    print("\n" + "="*60)
    print("Starting API Server - Preloading Models...")
    print("="*60)
    
    # Uncomment the models you want to preload
    # Warning: Preloading all models requires significant GPU memory
    
    # try:
    #     load_model('hr')
    #     load_model('sales')
    #     load_model('healthcare')
    #     load_model('marketing')
    #     # load_model('finance')  # Uncomment if available
    # except Exception as e:
    #     print(f"Warning: Could not preload some models: {e}")
    
    print("\n✓ API Server Ready!")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        app,
        port=8000,
        log_level="info"
    )
