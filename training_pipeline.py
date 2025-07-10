import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from typing import List, Dict, Tuple
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
from config import Config
from model_architecture import LinkedInLLMOriginal
from tokenizer import OriginalLinkedInTokenizer

class LinkedInDataset(Dataset):
    def __init__(self, posts: List[Dict], tokenizer: OriginalLinkedInTokenizer, config: Config):
        self.posts = posts
        self.tokenizer = tokenizer
        self.config = config
        
        self.theme_to_id = {theme: i for i, theme in enumerate(config.THEMES)}
        
        print("Pre-tokenizing dataset for memory efficiency...")
        self.tokenized_posts = []
        
        for i, post in enumerate(tqdm(posts, desc="Tokenizing")):
            token_ids = self.tokenizer.encode(post['post'])
            
            if len(token_ids) > self.config.MAX_SEQ_LENGTH:
                token_ids = token_ids[:self.config.MAX_SEQ_LENGTH]
            else:
                token_ids.extend([0] * (self.config.MAX_SEQ_LENGTH - len(token_ids)))
            
            theme_id = self.theme_to_id.get(post['theme'], 0)
            
            self.tokenized_posts.append({
                'input_ids': token_ids[:-1],
                'target_ids': token_ids[1:],
                'theme_id': theme_id,
                'attention_mask': [1 if tid != 0 else 0 for tid in token_ids[:-1]]
            })
        
        print(f"Pre-tokenization complete: {len(self.tokenized_posts)} posts")
    
    def __len__(self) -> int:
        return len(self.tokenized_posts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.tokenized_posts[idx]
        
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'target_ids': torch.tensor(item['target_ids'], dtype=torch.long),
            'theme_id': torch.tensor(item['theme_id'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long)
        }
    
class MemoryOptimizedLossFunction(nn.Module):    
    def __init__(self, vocab_size: int, num_themes: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_themes = num_themes
        
        self.language_weight = 1.0
        self.professionalism_weight = 0.1
        self.theme_weight = 0.1
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                professionalism_scores: torch.Tensor, theme_predictions: torch.Tensor,
                theme_targets: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        
        language_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            targets.view(-1),
            ignore_index=0,
            reduction='mean'
        )
        
        professionalism_target = torch.clamp(torch.ones_like(professionalism_scores) * 0.7, 0.0, 1.0)
        professionalism_loss = F.mse_loss(professionalism_scores, professionalism_target)
        
        theme_loss = F.cross_entropy(theme_predictions, theme_targets)
        
        total_loss = (
            self.language_weight * language_loss +
            self.professionalism_weight * professionalism_loss +
            self.theme_weight * theme_loss
        )
        
        if total_loss.dtype == torch.float16:
            total_loss = torch.clamp(total_loss, 0.0, 65504.0)
        
        return total_loss, {
            'language_loss': language_loss.item(),
            'professionalism_loss': professionalism_loss.item(),
            'theme_loss': theme_loss.item(),
            'total_loss': total_loss.item()
        }
    
class MemoryOptimizedLinkedInTrainer:   
    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        
        config.print_memory_config()
        
        os.makedirs(config.DATA_DIR, exist_ok=True)
        os.makedirs(config.MODEL_DIR, exist_ok=True) 
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        self.scaler = GradScaler('cuda') if config.USE_MIXED_PRECISION else None
        
        self.training_history = {
            'losses': [],
            'memory_usage': []
        }
    
    def clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def get_memory_usage(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0
    
    def prepare_tokenizer(self, posts: List[Dict]) -> OriginalLinkedInTokenizer:
        print("Preparing tokenizer...")
        self.tokenizer = OriginalLinkedInTokenizer(self.config.VOCAB_SIZE)
        texts = [post['post'] for post in posts]
        self.tokenizer.train(texts)
        tokenizer_path = os.path.join(self.config.MODEL_DIR, 'tokenizer.pkl')
        self.tokenizer.save(tokenizer_path)
        
        print(f"Tokenizer trained and saved to {tokenizer_path}")
        return self.tokenizer
    
    def prepare_model(self) -> LinkedInLLMOriginal:
        print("Initializing Linkwrox...")
        
        self.model = LinkedInLLMOriginal(self.config).to(self.device)
        
        param_count = self.model.count_parameters()
        print(f"Model initialized with {param_count:,} parameters")
        print(f"Estimated memory usage: {param_count * 4 / 1024**3:.2f} GB")
        
        return self.model
    
    def prepare_optimizer_and_loss(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=0.01,
            eps=1e-8
        )
        
        steps_per_epoch = len(self.train_dataloader) if hasattr(self, 'train_dataloader') else 563
        total_steps = self.config.NUM_EPOCHS * steps_per_epoch // self.config.GRADIENT_ACCUMULATION
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.LEARNING_RATE,
            total_steps=total_steps,
            pct_start=0.1
        )
        
        print(f"Scheduler configured for {total_steps} total steps")
        
        self.criterion = MemoryOptimizedLossFunction(
            self.config.VOCAB_SIZE,
            len(self.config.THEMES)
        )
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        self.model.train()
        
        total_loss = 0
        total_samples = 0
        epoch_metrics = {
            'language_loss': 0,
            'professionalism_loss': 0,
            'theme_loss': 0
        }
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        accumulation_steps = self.config.GRADIENT_ACCUMULATION
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            target_ids = batch['target_ids'].to(self.device, non_blocking=True)
            theme_ids = batch['theme_id'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            
            if self.config.USE_MIXED_PRECISION:
                with autocast('cuda'):
                    logits, professionalism_scores, theme_predictions = self.model(input_ids, theme_ids)
                    loss, metrics = self.criterion(
                        logits, target_ids, professionalism_scores,
                        theme_predictions, theme_ids, attention_mask
                    )
                    loss = loss / accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                logits, professionalism_scores, theme_predictions = self.model(input_ids, theme_ids)
                loss, metrics = self.criterion(
                    logits, target_ids, professionalism_scores,
                    theme_predictions, theme_ids, attention_mask
                )
                loss = loss / accumulation_steps
                loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.config.USE_MIXED_PRECISION:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size * accumulation_steps
            total_samples += batch_size
            
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key] * batch_size
            
            if batch_idx % 10 == 0:
                memory_usage = self.get_memory_usage()
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * accumulation_steps:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.6f}",
                    'mem': f"{memory_usage:.1f}GB"
                })
                
                if batch_idx % 50 == 0:
                    self.clear_memory()
        
        if len(dataloader) % accumulation_steps != 0:
            if self.config.USE_MIXED_PRECISION:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / total_samples
        for key in epoch_metrics:
            epoch_metrics[key] /= total_samples
        
        self.training_history['memory_usage'].append(self.get_memory_usage())
        
        return {'avg_loss': avg_loss, **epoch_metrics}
    
    def evaluate_model(self, dataloader: DataLoader) -> Dict:
        self.model.eval()
        
        total_loss = 0
        total_samples = 0
        correct_themes = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                target_ids = batch['target_ids'].to(self.device, non_blocking=True)
                theme_ids = batch['theme_id'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                
                if self.config.USE_MIXED_PRECISION:
                    with autocast('cuda'):
                        logits, professionalism_scores, theme_predictions = self.model(input_ids, theme_ids)
                        loss, _ = self.criterion(
                            logits, target_ids, professionalism_scores,
                            theme_predictions, theme_ids, attention_mask
                        )
                else:
                    logits, professionalism_scores, theme_predictions = self.model(input_ids, theme_ids)
                    loss, _ = self.criterion(
                        logits, target_ids, professionalism_scores,
                        theme_predictions, theme_ids, attention_mask
                    )
                
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                theme_pred = torch.argmax(theme_predictions, dim=1)
                correct_themes += (theme_pred == theme_ids).sum().item()
        
        return {
            'eval_loss': total_loss / total_samples,
            'theme_accuracy': correct_themes / total_samples
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'training_history': self.training_history
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = os.path.join(self.config.MODEL_DIR, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        best_path = os.path.join(self.config.MODEL_DIR, 'best_model.pt')
        torch.save(checkpoint, best_path)
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def train(self, train_posts: List[Dict], val_posts: List[Dict] = None):
        print("Starting Linkwrox training...")
        
        self.clear_memory()
        self.prepare_tokenizer(train_posts)
        self.prepare_model()
        
        print("Creating datasets...")
        train_dataset = LinkedInDataset(train_posts, self.tokenizer, self.config)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_dataloader = None
        if val_posts:
            val_dataset = LinkedInDataset(val_posts, self.tokenizer, self.config)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
        
        self.prepare_optimizer_and_loss()
        
        best_loss = float('inf')
        
        print(f"Starting training on {self.device}")
        print(f"Initial memory usage: {self.get_memory_usage():.2f} GB")
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            
            train_metrics = self.train_epoch(self.train_dataloader, epoch)
            
            eval_metrics = {}
            if val_dataloader:
                eval_metrics = self.evaluate_model(val_dataloader)
                print(f"Validation Loss: {eval_metrics['eval_loss']:.4f}, "
                      f"Theme Accuracy: {eval_metrics['theme_accuracy']:.4f}")
            
            self.training_history['losses'].append(train_metrics['avg_loss'])
            
            if train_metrics['avg_loss'] < best_loss:
                best_loss = train_metrics['avg_loss']
                self.save_checkpoint(epoch, {**train_metrics, **eval_metrics})
            
            print(f"Training Loss: {train_metrics['avg_loss']:.4f}")
            print(f"Memory Usage: {self.get_memory_usage():.2f} GB")
            
            self.clear_memory()
        
        print("Training completed!")
        return self.model
    
    def plot_training_history(self):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.training_history['losses'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 3, 2)
        plt.plot(self.training_history['memory_usage'])
        plt.title('GPU Memory Usage')
        plt.xlabel('Epoch')
        plt.ylabel('Memory (GB)')
        
        plt.subplot(1, 3, 3)
        if len(self.training_history['losses']) > 1:
            plt.plot(self.training_history['losses'][1:])
            plt.title('Training Loss (excluding first epoch)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.LOGS_DIR, 'training_history.png'))
        plt.show()