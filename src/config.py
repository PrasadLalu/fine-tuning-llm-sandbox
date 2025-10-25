# Global model cache
MODELS_CACHE = {}
BASE_MODEL_NAME = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

# Model configurations
MODEL_CONFIGS = {
    'hr': {
        'path': 'models/hr_full_finetuned',
        'type': 'full',
        'description': 'HR policies, leave, PF, salary queries'
    },
    'finance': {
        'path': 'models/finance_dpo_finetuned',
        'type': 'dpo',
        'description': 'Finance, GST, tax, investments queries'
    },
    'sales': {
        'path': 'models/sales_peft_finetuned',
        'type': 'peft',
        'description': 'Sales, customer service, e-commerce queries'
    },
    'healthcare': {
        'path': 'models/healthcare_lora_finetuned',
        'type': 'lora',
        'description': 'Healthcare, medical, Ayurveda queries'
    },
    'marketing': {
        'path': 'models/marketing_qlora_finetuned',
        'type': 'qlora',
        'description': 'Marketing campaigns, strategies queries'
    }
}
