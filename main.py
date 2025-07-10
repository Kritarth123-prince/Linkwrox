import os
import argparse
import json
from typing import List, Dict
from config import Config
from data_generator import OriginalLinkedInDataGenerator
from training_pipeline import MemoryOptimizedLinkedInTrainer
from inference import LinkedInLLMInference

def print_banner():    
    banner = """
    ╔════════════════════════════════════════════════════════════╗
    ║                       Linkwrox                             ║
    ║                                                            ║
    ║    Proprietary Large Language Model for LinkedIn Posts     ║
    ║            Copyright (c) 2025 Kritarth Ranjan              ║
    ║                    All Rights Reserved                     ║
    ╚════════════════════════════════════════════════════════════╝
    """
    print(banner)
def generate_data(config: Config, num_posts: int = None) -> List[Dict]:
    print("\n=== Data Generation Phase ===")
    
    generator = OriginalLinkedInDataGenerator()
    num_posts = num_posts or config.NUM_SYNTHETIC_POSTS
    
    dataset = generator.generate_dataset(num_posts)
    
    data_path = os.path.join(config.DATA_DIR, 'linkedin_posts.json')
    generator.save_dataset(dataset, data_path)
    
    return dataset

def train_model(config: Config, dataset: List[Dict]) -> str:
    print("\n=== Memory-Optimized Model Training Phase ===")
    
    split_idx = int(len(dataset) * 0.9)
    train_posts = dataset[:split_idx]
    val_posts = dataset[split_idx:]
    
    print(f"Training set: {len(train_posts)} posts")
    print(f"Validation set: {len(val_posts)} posts")
    
    trainer = MemoryOptimizedLinkedInTrainer(config)
    model = trainer.train(train_posts, val_posts)    
    trainer.plot_training_history()
    
    best_model_path = os.path.join(config.MODEL_DIR, 'best_model.pt')
    return best_model_path

def run_inference(config: Config, model_path: str, tokenizer_path: str):
    print("\n=== Inference Phase ===")
    inference_engine = LinkedInLLMInference(model_path, tokenizer_path, config)
    
    print("\n--- Generating Sample Posts ---")
    themes = ['career_advice', 'leadership', 'innovation', 'networking']
    
    for theme in themes:
        print(f"\n Theme: {theme.replace('_', ' ').title()}")
        result = inference_engine.generate_post(theme=theme)
        print(f" Post:\n{result['post']}")
        print(f" Analysis: {result['analysis']['word_count']} words, "
              f"Professionalism: {result['analysis']['professionalism_score']:.3f}")
        print("-" * 60)
    
    print("\n--- Starting Interactive Mode ---")
    inference_engine.interactive_generation()

def save_generated_posts(posts: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    posts_path = os.path.join(output_dir, 'generated_posts.json')
    with open(posts_path, 'w', encoding='utf-8') as f:
        json.dump(posts, f, indent=2, ensure_ascii=False)
    
    for i, post_data in enumerate(posts):
        filename = f"post_{i+1}_{post_data['theme']}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Theme: {post_data['theme']}\n")
            f.write(f"Generated: {post_data['metadata']}\n")
            f.write("-" * 50 + "\n\n")
            f.write(post_data['post'])
            f.write("\n\n" + "-" * 50 + "\n")
            f.write(f"Analysis: {post_data['analysis']}\n")
    
    print(f"Posts saved to {output_dir}")

def batch_generation_mode(config: Config, model_path: str, tokenizer_path: str):
    print("\n=== Batch Generation Mode ===")
    
    inference_engine = LinkedInLLMInference(model_path, tokenizer_path, config)
    
    themes = config.THEMES
    posts_per_theme = 5
    
    print(f"Generating {posts_per_theme} posts for each of {len(themes)} themes...")
    all_posts = []
    theme_stats = {}
    
    for theme in themes:
        print(f"\n Generating posts for theme: {theme}")
        
        theme_posts = inference_engine.generate_multiple_posts(
            num_posts=posts_per_theme, 
            theme=theme,
            max_length=150
        )
        
        for post in theme_posts:
            post['theme'] = theme
            all_posts.append(post)
        
        avg_length = sum(p['analysis']['word_count'] for p in theme_posts) / len(theme_posts)
        avg_professionalism = sum(p['analysis']['professionalism_score'] for p in theme_posts) / len(theme_posts)
        
        theme_stats[theme] = {
            'posts_generated': len(theme_posts),
            'avg_word_count': avg_length,
            'avg_professionalism': avg_professionalism
        }
        
        print(f" Generated {len(theme_posts)} posts (Avg: {avg_length:.1f} words, Prof: {avg_professionalism:.3f})")
    
    output_dir = os.path.join(config.DATA_DIR, 'generated_output')
    save_generated_posts(all_posts, output_dir)
    
    print("\n=== Generation Summary ===")
    total_posts = len(all_posts)
    total_words = sum(p['analysis']['word_count'] for p in all_posts)
    avg_professionalism = sum(p['analysis']['professionalism_score'] for p in all_posts) / total_posts
    
    print(f"Total posts generated: {total_posts}")
    print(f"Total words generated: {total_words}")
    print(f"Average professionalism score: {avg_professionalism:.3f}")
    
    print("\nPer-theme statistics:")
    for theme, stats in theme_stats.items():
        print(f"  {theme}: {stats['posts_generated']} posts, "
              f"{stats['avg_word_count']:.1f} avg words, "
              f"{stats['avg_professionalism']:.3f} professionalism")
    
    return all_posts

def demo_mode(config: Config, model_path: str, tokenizer_path: str):
    print("\n=== Demo Mode ===")
    inference_engine = LinkedInLLMInference(model_path, tokenizer_path, config)
    
    demo_scenarios = [
        {
            'prompt': 'After 10 years in technology',
            'theme': 'career_advice',
            'description': 'Career reflection post'
        },
        {
            'prompt': 'Innovation is not just about',
            'theme': 'innovation',
            'description': 'Innovation insights'
        },
        {
            'prompt': 'Leadership lesson learned',
            'theme': 'leadership',
            'description': 'Leadership story'
        },
        {
            'prompt': 'Networking changed my career',
            'theme': 'networking',
            'description': 'Networking success story'
        },
        {
            'prompt': '',
            'theme': 'professional_development',
            'description': 'Professional growth (no prompt)'
        }
    ]
    
    print(" Running demo scenarios...\n")
    
    for i, scenario in enumerate(demo_scenarios, 1):
        print(f"Demo {i}: {scenario['description']}")
        print(f"Prompt: '{scenario['prompt']}'")
        print(f"Theme: {scenario['theme']}")
        print("-" * 60)
        
        result = inference_engine.generate_post(
            prompt=scenario['prompt'],
            theme=scenario['theme'],
            temperature=0.7
        )
        
        print(f"Generated Post:\n{result['post']}\n")
        
        analysis = result['analysis']
        print(f"   Analysis:")
        print(f"   Word count: {analysis['word_count']}")
        print(f"   Professionalism: {analysis['professionalism_score']:.3f}")
        print(f"   Professional keywords: {analysis['professional_keywords']}")
        print(f"   Hashtags: {analysis['hashtags']}")
        print(f"   Readability: {analysis['readability']}")
        
        print("\n" + "="*80 + "\n")
        
        input("Press Enter to continue to next demo...")

def export_model_info(config: Config, model_path: str):
    print("\n=== Exporting Model Information ===")
    model_info = {
        'model_name': 'Linkwrox',
        'version': config.VERSION,
        'copyright': config.COPYRIGHT,
        'license': config.LICENSE,
        'owner': 'Kritarth Ranjan',
        'creation_date': '2025-07-10',
        'model_architecture': {
            'type': 'Transformer (Proprietary)',
            'model_dim': config.MODEL_DIM,
            'num_heads': config.NUM_HEADS,
            'num_layers': config.NUM_LAYERS,
            'vocab_size': config.VOCAB_SIZE,
            'max_sequence_length': config.MAX_SEQ_LENGTH
        },
        'training_config': {
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'num_epochs': config.NUM_EPOCHS
        },
        'capabilities': [
            'LinkedIn post generation',
            'Theme-aware content creation',
            'Professional tone optimization',
            'Multi-theme support',
            'Interactive generation'
        ],
        'supported_themes': config.THEMES,
        'model_path': model_path,
        'proprietary_features': [
            'Transformer architecture',
            'Custom attention mechanisms',
            'Professional content gating',
            'Theme-aware generation',
            'Proprietary loss functions',
            'LinkedIn-specific tokenization'
        ]
    }
    
    info_path = os.path.join(config.MODEL_DIR, 'model_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    copyright_notice = f"""
        Linkwrox - Copyright Notice
        ===========================
        Model Name: Linkwrox
        Version: {config.VERSION}
        Owner: Kritarth Ranjan
        Creation Date: 2025-07-10
        Last Modified: 2025-07-10
        COPYRIGHT NOTICE:
        {config.COPYRIGHT}
        LICENSE:
        {config.LICENSE}
        This is a completely implementation of a Large Language Model
        specifically designed for LinkedIn post generation. All code, algorithms,
        and training methodologies are proprietary.
        Key Proprietary Features:
        - Transformer architecture with LinkedIn-specific enhancements
        - Custom tokenization system optimized for professional content
        - Proprietary loss functions for professional tone optimization
        - Theme-aware generation capabilities
        - Data generation pipeline
        - Custom attention mechanisms
        No external datasets or pre-trained models were used in the creation
        of this system. All training data was generated synthetically using
        algorithms.
        For licensing inquiries, please contact: Kritarth Ranjan
        PATENT PENDING - All novel algorithms and implementations described
        herein are subject to patent protection.
    """
    
    copyright_path = os.path.join(config.MODEL_DIR, 'COPYRIGHT.txt')
    with open(copyright_path, 'w', encoding='utf-8') as f:
        f.write(copyright_notice)
    
    print(f"Model information exported to {info_path}")
    print(f"Copyright notice saved to {copyright_path}")

def main():    
    parser = argparse.ArgumentParser(description="Linkwrox System")
    parser.add_argument('--mode', choices=['full', 'data', 'train', 'inference', 'batch', 'demo'], 
                       default='full', help='Execution mode')
    parser.add_argument('--num-posts', type=int, default=5000,
                       help='Number of synthetic posts to generate')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--model-path', type=str, 
                       help='Path to trained model for inference')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and use existing model')
    parser.add_argument('--export-info', action='store_true',
                       help='Export model information and copyright details')
    
    args = parser.parse_args()
    print_banner()
    config = Config()
    
    if args.epochs != 20:
        config.NUM_EPOCHS = args.epochs
    
    print(f"Configuration loaded - Version {config.VERSION}")
    print(f"Device: {config.DEVICE}")
    print(f"Model dimensions: {config.MODEL_DIM}")
    print(f"Vocabulary size: {config.VOCAB_SIZE}")
    
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    if args.mode in ['full', 'data']:
        dataset = generate_data(config, args.num_posts)
    else:
        data_path = os.path.join(config.DATA_DIR, 'linkedin_posts.json')
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                dataset = json.load(f)
            print(f"Loaded existing dataset: {len(dataset)} posts")
        else:
            print("No existing dataset found. Please run data generation first.")
            return
    
    if args.mode in ['full', 'train'] and not args.skip_training:
        model_path = train_model(config, dataset)
    else:
        model_path = args.model_path or os.path.join(config.MODEL_DIR, 'best_model.pt')
    
    if args.mode in ['full', 'inference', 'batch', 'demo']:
        tokenizer_path = os.path.join(config.MODEL_DIR, 'tokenizer.pkl')
        
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            if args.mode == 'batch':
                batch_generation_mode(config, model_path, tokenizer_path)
            elif args.mode == 'demo':
                demo_mode(config, model_path, tokenizer_path)
            else:
                run_inference(config, model_path, tokenizer_path)
        else:
            print("Model or tokenizer not found. Please train the model first.")
            print(f"Looking for model at: {model_path}")
            print(f"Looking for tokenizer at: {tokenizer_path}")
            
            print("\n To train the model, run:")
            print("   python main.py --mode train")
            print("\n To generate data and train, run:")
            print("   python main.py --mode full")
            print("\n To generate just data, run:")
            print("   python main.py --mode data")
            
            return
    
    if args.export_info and os.path.exists(model_path):
        export_model_info(config, model_path)
    
    print("\n Linkwrox execution completed successfully!")
    print(f" System ready for LinkedIn post generation")
    print(f" All rights reserved - Kritarth Ranjan")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Execution interrupted by user")
    except Exception as e:
        print(f"\n Error during execution: {e}")
        print("Please check the error message above and try again.")
        import traceback
        traceback.print_exc()
    finally:
        print("\n Thank you for using Linkwrox!")