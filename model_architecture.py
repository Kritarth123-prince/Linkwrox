import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from config import Config

class OriginalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        

        social_encoding = torch.sin(position / 100.0) * 0.1
        pe += social_encoding
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]
    
class OriginalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.content_gate = nn.Linear(d_model, num_heads)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.mask_value = -65504.0
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        batch_size, seq_len = query.size(0), query.size(1)
        
        Q = self.w_q(query)
        K = self.w_k(key) 
        V = self.w_v(value)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        content_weights = torch.sigmoid(self.content_gate(query))
        content_weights = content_weights.unsqueeze(-1).transpose(1, 2)
        scores = scores * content_weights
        
        if mask is not None:
            mask_value = self.mask_value if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(mask == 0, mask_value)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        
        return self.layer_norm(output + query)
class OriginalFeedForward(nn.Module):   
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.professional_gate = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.professional_gate(x))
        
        hidden = self.linear1(x)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)
        output = self.linear2(hidden)
        
        gate = torch.clamp(gate, 0.0, 1.0)
        output = output * gate
        
        return self.layer_norm(output + x)
    
class OriginalTransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = OriginalMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = OriginalFeedForward(d_model, d_ff, dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.self_attention(x, x, x, mask)
        x = self.feed_forward(x)
        return x
    
class LinkedInLLMOriginal(nn.Module):    
    def __init__(self, config: Config):
        super().__init__()
        
        self.config = config
        
        self.token_embedding = nn.Embedding(config.VOCAB_SIZE, config.MODEL_DIM)
        self.theme_embedding = nn.Embedding(len(config.THEMES), config.MODEL_DIM)
        self.position_encoding = OriginalPositionalEncoding(config.MODEL_DIM, config.MAX_SEQ_LENGTH)
        
        self.layers = nn.ModuleList([
            OriginalTransformerLayer(
                config.MODEL_DIM, 
                config.NUM_HEADS, 
                config.FF_DIM,
                dropout=0.1
            ) for _ in range(config.NUM_LAYERS)
        ])
        
        self.output_projection = nn.Linear(config.MODEL_DIM, config.VOCAB_SIZE)
        
        self.professional_classifier = nn.Linear(config.MODEL_DIM, 1)
        self.theme_predictor = nn.Linear(config.MODEL_DIM, len(config.THEMES))
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, input_ids: torch.Tensor, theme_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        
        x = self.token_embedding(input_ids)
        
        if theme_ids is not None:
            theme_emb = self.theme_embedding(theme_ids).unsqueeze(1)
            x = x + theme_emb
        
        x = self.position_encoding(x)
        
        mask = self.create_causal_mask(seq_len).to(input_ids.device)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        logits = self.output_projection(x)
        
        pooled = x.mean(dim=1)
        professionalism_score = torch.sigmoid(self.professional_classifier(pooled))
        theme_prediction = self.theme_predictor(pooled)
        
        if x.dtype == torch.float16:
            logits = torch.clamp(logits, -65504, 65504)
            professionalism_score = torch.clamp(professionalism_score, 0.0, 1.0)
            theme_prediction = torch.clamp(theme_prediction, -65504, 65504)
        
        return logits, professionalism_score, theme_prediction
    
    def generate(self, prompt_ids: torch.Tensor, theme_id: Optional[int] = None, 
                max_length: int = 200, temperature: float = 0.8) -> torch.Tensor:
        self.eval()
        
        with torch.no_grad():
            generated = prompt_ids.clone()
            batch_size = prompt_ids.size(0)
            
            theme_ids = None
            if theme_id is not None:
                theme_ids = torch.full((batch_size,), theme_id, device=prompt_ids.device)
            
            for _ in range(max_length):
                logits, _, _ = self.forward(generated, theme_ids)
                
                next_token_logits = logits[:, -1, :] / temperature
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                if next_token.item() == 3:
                    break
            
            return generated
        
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)