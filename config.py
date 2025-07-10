import torch
import os

class Config:
    # Model Architecture
    MODEL_DIM = 256
    NUM_HEADS = 4
    NUM_LAYERS = 4
    FF_DIM = 1024
    MAX_SEQ_LENGTH = 256
    VOCAB_SIZE = 25000
    
    # Training Parameters
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    WARMUP_STEPS = 500
    GRADIENT_CLIP = 1.0
    GRADIENT_ACCUMULATION = 4
    
    USE_MIXED_PRECISION = True
    CHECKPOINT_GRADIENT = True
    
    NUM_SYNTHETIC_POSTS = 10000
    THEMES = [
        'career_advice',
        'industry_insights', 
        'leadership',
        'entrepreneurship',
        'professional_development',
        'technology_trends',
        'personal_branding',
        'networking',
        'innovation',
        'workplace_culture'
    ]
    
    DATA_DIR = 'data'
    MODEL_DIR = 'models'
    LOGS_DIR = 'logs'
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    VERSION = "1.0.0-Optimized"
    COPYRIGHT = "Copyright (c) 2025 Kritarth Ranjan"
    LICENSE = "Proprietary - All Rights Reserved"
    
    def print_memory_config(self):
        print(f"   GPU Optimization Enabled:")
        print(f"   Model Size: ~{self.estimate_model_size():.1f}M parameters")
        print(f"   Batch Size: {self.BATCH_SIZE} (with {self.GRADIENT_ACCUMULATION}x accumulation)")
        print(f"   Mixed Precision: {self.USE_MIXED_PRECISION}")
        print(f"   Gradient Checkpointing: {self.CHECKPOINT_GRADIENT}")
    
    def estimate_model_size(self):
        token_embed = self.VOCAB_SIZE * self.MODEL_DIM
        pos_embed = self.MAX_SEQ_LENGTH * self.MODEL_DIM
        
        attention_params = self.NUM_LAYERS * (4 * self.MODEL_DIM * self.MODEL_DIM)
        ff_params = self.NUM_LAYERS * (2 * self.MODEL_DIM * self.FF_DIM)
        
        output_params = self.MODEL_DIM * self.VOCAB_SIZE
        
        total_params = token_embed + pos_embed + attention_params + ff_params + output_params
        return total_params / 1_000_000