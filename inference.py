import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
import random

from config import Config
from model_architecture import LinkedInLLMOriginal
from tokenizer import OriginalLinkedInTokenizer

class LinkedInLLMInference:
    def __init__(self, model_path: str, tokenizer_path: str, config: Config = None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        
        self.tokenizer = OriginalLinkedInTokenizer()
        self.tokenizer.load(tokenizer_path)
        
        self.model = LinkedInLLMOriginal(self.config)
        
        try:
            torch.serialization.add_safe_globals([Config])
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except Exception:
            print("Using weights_only=False for your trusted checkpoint file")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        
        self.theme_to_id = {theme: i for i, theme in enumerate(self.config.THEMES)}
        self.id_to_theme = {i: theme for theme, i in self.theme_to_id.items()}
        
        print("Linkwrox loaded successfully!")
        print(f"Model parameters: {self.model.count_parameters():,}")
    
    def generate_post(self, 
                     prompt: str = "",
                     theme: str = None,
                     max_length: int = 200,
                     temperature: float = 0.8,
                     top_p: float = 0.9,
                     repetition_penalty: float = 1.1) -> Dict:
        if not prompt:
            prompt = "<START>"
        
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(self.device)
        
        theme_id = None
        if theme and theme in self.theme_to_id:
            theme_id = self.theme_to_id[theme]
        elif theme is None:
            theme_id = random.choice(list(self.theme_to_id.values()))
        
        generated_ids = self._generate_with_sampling(
            prompt_tensor, theme_id, max_length, temperature, top_p, repetition_penalty
        )
        
        generated_text = self.tokenizer.decode(generated_ids[0].cpu().tolist())
        
        generated_text = self._post_process_text(generated_text)

        analysis = self._analyze_post(generated_text, generated_ids[0])
        
        return {
            'post': generated_text,
            'theme': self.id_to_theme.get(theme_id, 'unknown'),
            'analysis': analysis,
            'metadata': {
                'prompt': prompt,
                'temperature': temperature,
                'top_p': top_p,
                'length': len(generated_text.split())
            }
        }
    
    def _generate_with_sampling(self, prompt_tensor: torch.Tensor, theme_id: Optional[int],
                               max_length: int, temperature: float, top_p: float,
                               repetition_penalty: float) -> torch.Tensor:
        
        with torch.no_grad():
            generated = prompt_tensor.clone()
            batch_size = prompt_tensor.size(0)
            
            theme_tensor = None
            if theme_id is not None:
                theme_tensor = torch.full((batch_size,), theme_id, device=self.device)
            
            token_history = {}
            
            for step in range(max_length):
                logits, professionalism_score, theme_prediction = self.model(generated, theme_tensor)
                
                next_token_logits = logits[:, -1, :] / temperature
                
                for token_id, count in token_history.items():
                    next_token_logits[:, token_id] /= (repetition_penalty ** count)
                
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                token_id = next_token.item()
                token_history[token_id] = token_history.get(token_id, 0) + 1
                
                generated = torch.cat([generated, next_token], dim=1)
                
                if token_id == self.tokenizer.special_tokens.get('<END>', 3):
                    break
                
                if generated.size(1) > self.config.MAX_SEQ_LENGTH:
                    break
            
            return generated
    
    def _post_process_text(self, text: str) -> str:
        special_tokens = ['<START>', '<END>', '<PAD>', '<UNK>']
        for token in special_tokens:
            text = text.replace(token, '')
        
        text = ' '.join(text.split())
        
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        text = text.replace(' !', '!')
        text = text.replace(' ?', '?')
        
        sentences = text.split('. ')
        sentences = [s.strip().capitalize() for s in sentences if s.strip()]
        text = '. '.join(sentences)
        
        return text.strip()
    
    def _analyze_post(self, text: str, token_ids: torch.Tensor) -> Dict:
        with torch.no_grad():
            input_tensor = token_ids.unsqueeze(0)
            _, professionalism_score, theme_prediction = self.model(input_tensor)
            
            words = text.split()
            sentences = text.split('.')
            
            professional_keywords = [
                'leadership', 'growth', 'success', 'innovation', 'strategy',
                'team', 'career', 'development', 'opportunity', 'achievement'
            ]
            
            keyword_count = sum(1 for word in words if word.lower() in professional_keywords)
            
            hashtags = [word for word in words if word.startswith('#')]
            
            return {
                'word_count': len(words),
                'sentence_count': len([s for s in sentences if s.strip()]),
                'professionalism_score': professionalism_score.item(),
                'professional_keywords': keyword_count,
                'hashtag_count': len(hashtags),
                'hashtags': hashtags,
                'readability': 'high' if len(words) < 100 else 'medium' if len(words) < 200 else 'low'
            }
    
    def generate_multiple_posts(self, num_posts: int, theme: str = None, **kwargs) -> List[Dict]:

        posts = []
        base_temperature = kwargs.get('temperature', 0.8)
        
        for i in range(num_posts):
            temp = base_temperature + random.uniform(-0.1, 0.1)
            temp = max(0.1, min(1.0, temp))
            
            generation_kwargs = kwargs.copy()
            generation_kwargs['temperature'] = temp
            
            post = self.generate_post(theme=theme, **generation_kwargs)
            post['id'] = i + 1
            posts.append(post)
        
        return posts
    
    def generate_themed_campaign(self, themes: List[str], posts_per_theme: int = 3) -> Dict:
        campaign = {}
        for theme in themes:
            if theme in self.theme_to_id:
                posts = self.generate_multiple_posts(posts_per_theme, theme=theme)
                campaign[theme] = posts
            else:
                print(f"Warning: Theme '{theme}' not recognized")
        
        return campaign
    
    def interactive_generation(self):      
        print("\n=== Linkwrox Interactive Mode ===")
        print("Available themes:", list(self.theme_to_id.keys()))
        print("Commands: 'quit' to exit, 'themes' to see themes")
        
        while True:
            print("\n" + "="*50)
            prompt = input("Enter prompt (or press Enter for automatic): ").strip()
            theme = input("Enter theme (or press Enter for random): ").strip()
            
            if prompt.lower() == 'quit':
                break
            elif prompt.lower() == 'themes':
                print("Available themes:", list(self.theme_to_id.keys()))
                continue
            
            try:
                result = self.generate_post(
                    prompt=prompt if prompt else "",
                    theme=theme if theme else None
                )
                
                print(f"\n--- Generated Post (Theme: {result['theme']}) ---")
                print(result['post'])
                print(f"\n--- Analysis ---")
                analysis = result['analysis']
                print(f"Word Count: {analysis['word_count']}")
                print(f"Professionalism Score: {analysis['professionalism_score']:.3f}")
                print(f"Professional Keywords: {analysis['professional_keywords']}")
                print(f"Hashtags: {analysis['hashtags']}")
                
            except Exception as e:
                print(f"Error generating post: {e}")