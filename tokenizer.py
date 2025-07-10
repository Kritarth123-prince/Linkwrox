import re
import json
import pickle
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

class OriginalLinkedInTokenizer:
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1, 
            '<START>': 2,
            '<END>': 3,
            '<THEME>': 4,
            '<HASHTAG>': 5,
            '<MENTION>': 6,
            '<LINK>': 7
        }
        
        self.professional_terms = {
            'leadership', 'innovation', 'strategy', 'growth', 'success',
            'career', 'professional', 'development', 'skills', 'experience',
            'team', 'collaboration', 'networking', 'opportunity', 'achievement'
        }
        
        self._initialize_vocab()
    
    def _initialize_vocab(self):
        self.vocab.update(self.special_tokens)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def _preprocess_text(self, text: str) -> str:
        text = re.sub(r'#(\w+)', r'<HASHTAG> \1', text)
        text = re.sub(r'@(\w+)', r'<MENTION> \1', text)
        text = re.sub(r'http[s]?://\S+', '<LINK>', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_subwords(self, word: str, max_len: int = 6) -> List[str]:
        if len(word) <= max_len:
            return [word]
        
        subwords = []
        for i in range(len(word)):
            for j in range(i + 2, min(i + max_len + 1, len(word) + 1)):
                subword = word[i:j]
                if len(subword) >= 2:
                    subwords.append(subword)
        
        return subwords
    
    def train(self, texts: List[str]):
        print("Training tokenizer...")
        
        token_counts = Counter()
        
        for text in texts:
            processed = self._preprocess_text(text)
            words = processed.lower().split()
            
            for word in words:
                token_counts[word] += 1
                
                subwords = self._extract_subwords(word)
                for subword in subwords:
                    token_counts[subword] += 1
                
                for char in word:
                    token_counts[char] += 1
        
        for term in self.professional_terms:
            if term in token_counts:
                token_counts[term] *= 5
        
        most_common = token_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        current_id = len(self.special_tokens)
        for token, count in most_common:
            if token not in self.vocab:
                self.vocab[token] = current_id
                self.inverse_vocab[current_id] = token
                current_id += 1
        
        print(f"Vocabulary built with {len(self.vocab)} tokens")
    
    def encode(self, text: str) -> List[int]:
        processed = self._preprocess_text(text)
        words = processed.lower().split()
        
        token_ids = [self.special_tokens['<START>']]
        
        for word in words:
            if word in self.vocab:
                token_ids.append(self.vocab[word])
            else:
                subwords = self._extract_subwords(word)
                found = False
                for subword in subwords:
                    if subword in self.vocab:
                        token_ids.append(self.vocab[subword])
                        found = True
                        break
                
                if not found:
                    for char in word:
                        if char in self.vocab:
                            token_ids.append(self.vocab[char])
                        else:
                            token_ids.append(self.special_tokens['<UNK>'])
        
        token_ids.append(self.special_tokens['<END>'])
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                if token not in ['<START>', '<END>', '<PAD>']:
                    tokens.append(token)
        
        return ' '.join(tokens)
    
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'inverse_vocab': self.inverse_vocab,
                'vocab_size': self.vocab_size,
                'special_tokens': self.special_tokens,
                'professional_terms': self.professional_terms
            }, f)
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vocab = data['vocab']
            self.inverse_vocab = data['inverse_vocab'] 
            self.vocab_size = data['vocab_size']
            self.special_tokens = data['special_tokens']
            self.professional_terms = data['professional_terms']