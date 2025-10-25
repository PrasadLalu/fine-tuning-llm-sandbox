import os
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import APIRouter, HTTPException
from .models import InferenceRequest, InferenceResponse
from .config import MODEL_CONFIGS, MODELS_CACHE, BASE_MODEL_NAME


router = APIRouter()

def load_model(model_name: str):
    """Load model into cache if not already loaded"""
    if model_name in MODELS_CACHE:
        return MODELS_CACHE[model_name]
    
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    model_path = config['path']
    model_type = config['type']
    
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Model files not found at {model_path}. Please train the model first."
        )
    
    print(f"Loading {model_name} model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model based on type
    if model_type in ['lora', 'peft', 'qlora', 'dpo']:
        # PEFT-based models
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map='auto',
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Full fine-tuned models
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            torch_dtype=torch.float16
        )
    
    model.eval()
    
    # Cache the model
    MODELS_CACHE[model_name] = {
        'model': model,
        'tokenizer': tokenizer,
        'config': config
    }
    
    print(f"✓ {model_name} model loaded successfully")
    return MODELS_CACHE[model_name]

def generate_response(model_name: str, query: str, max_tokens: int, temperature: float, top_p: float):
    """Generate response from specified model"""
    
    # Load model if not cached
    model_data = load_model(model_name)
    model = model_data['model']
    tokenizer = model_data['tokenizer']
    
    # Prepare prompt
    prompt = f"### Instruction: {query}\n### Response:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to GPU
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response.split("### Response:")[-1].strip()
    
    tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
    
    return response, tokens_generated


@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models_loaded": list(MODELS_CACHE.keys())
    }


@router.get('/')
def root():
    """Root endpoint with API information"""
    return {
        "message": "LLM Fine-tuned Models API",
        "version": "1.0.0",
        "models": list(MODEL_CONFIGS.keys()),
        "endpoints": {
            "hr": "/api/hr",
            "finance": "/api/finance",
            "sales": "/api/sales",
            "healthcare": "/api/healthcare",
            "marketing": "/api/marketing",
            "all_models": "/api/models"
        },
        "docs": "/docs",
        "gpu": "CUDA available" if torch.cuda.is_available() else "CPU only"
    }


@router.get("/models")
def list_models():
    """List all available models"""
    return {
        "models": [
            {
                "name": name,
                "path": config['path'],
                "type": config['type'],
                "description": config['description'],
                "loaded": name in MODELS_CACHE
            }
            for name, config in MODEL_CONFIGS.items()
        ]
    }
    
@router.post("/hr", response_model=InferenceResponse)
def hr_inference(request: InferenceRequest):
    """HR model inference endpoint"""
    try:
        response, tokens = generate_response(
            'hr', 
            request.query, 
            request.max_tokens, 
            request.temperature, 
            request.top_p
        )
        return InferenceResponse(
            model='hr',
            query=request.query,
            response=response,
            tokens_generated=tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/finance", response_model=InferenceResponse)
def finance_inference(request: InferenceRequest):
    """Finance model inference endpoint"""
    try:
        response, tokens = generate_response(
            'finance', 
            request.query, 
            request.max_tokens, 
            request.temperature, 
            request.top_p
        )
        return InferenceResponse(
            model='finance',
            query=request.query,
            response=response,
            tokens_generated=tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sales", response_model=InferenceResponse)
def sales_inference(request: InferenceRequest):
    """Sales model inference endpoint"""
    try:
        response, tokens = generate_response(
            'sales', 
            request.query, 
            request.max_tokens, 
            request.temperature, 
            request.top_p
        )
        return InferenceResponse(
            model='sales',
            query=request.query,
            response=response,
            tokens_generated=tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/healthcare", response_model=InferenceResponse)
def healthcare_inference(request: InferenceRequest):
    """Healthcare model inference endpoint"""
    try:
        response, tokens = generate_response(
            'healthcare', 
            request.query, 
            request.max_tokens, 
            request.temperature, 
            request.top_p
        )
        return InferenceResponse(
            model='healthcare',
            query=request.query,
            response=response,
            tokens_generated=tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/marketing", response_model=InferenceResponse)
def marketing_inference(request: InferenceRequest):
    """Marketing model inference endpoint"""
    try:
        response, tokens = generate_response(
            'marketing', 
            request.query, 
            request.max_tokens, 
            request.temperature, 
            request.top_p
        )
        return InferenceResponse(
            model='marketing',
            query=request.query,
            response=response,
            tokens_generated=tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/infer/{model_name}", response_model=InferenceResponse)
def generic_inference(model_name: str, request: InferenceRequest):
    """Generic inference endpoint - works with any model"""
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=404, 
            detail=f"Model {model_name} not found. Available: {list(MODEL_CONFIGS.keys())}"
        )
    
    try:
        response, tokens = generate_response(
            model_name, 
            request.query, 
            request.max_tokens, 
            request.temperature, 
            request.top_p
        )
        return InferenceResponse(
            model=model_name,
            query=request.query,
            response=response,
            tokens_generated=tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
