"""
evaluate.py — CLIP Score Evaluation
IIT Indore | GenAI Course Project

Computes CLIP similarity score between text prompts and generated/edited images.
Higher score (0–1) = better text-image alignment.
"""

import os
import argparse
import torch
import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def load_clip(device: str = None):
    """Load CLIP model and processor."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP on {device}...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor, device


def compute_clip_score(model, processor, image: Image.Image,
                        text: str, device: str) -> float:
    """
    Compute cosine similarity between image and text embeddings via CLIP.
    Returns a float in [0, 1].
    """
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image  # shape: (1, 1)
        score = logits.squeeze().item() / 100.0  # normalize to ~[0, 1] range
    
    return round(score, 4)


def evaluate_directory(image_dir: str, prompts: list, model, processor, device: str) -> list:
    """
    Evaluate all images in a directory against corresponding prompts.
    Expects images to be sorted in same order as prompts list.
    """
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    
    results = []
    for i, (img_file, prompt) in enumerate(zip(image_files, prompts)):
        img_path = os.path.join(image_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        score = compute_clip_score(model, processor, image, prompt, device)
        
        print(f"[{i+1}] {img_file}")
        print(f"     Prompt: {prompt[:60]}")
        print(f"     CLIP Score: {score:.4f}")
        results.append({"image": img_file, "prompt": prompt, "clip_score": score})
    
    return results


def evaluate_pair(image_path: str, text: str, model=None, processor=None, device=None):
    """Evaluate a single image-text pair."""
    if model is None:
        model, processor, device = load_clip()
    
    image = Image.open(image_path).convert("RGB")
    score = compute_clip_score(model, processor, image, text, device)
    print(f"CLIP Score for '{text[:50]}': {score:.4f}")
    return score


def print_summary(results: list):
    """Print a formatted summary table."""
    print("\n" + "="*60)
    print(f"{'Image':<30} {'CLIP Score':>12}")
    print("-"*60)
    for r in results:
        print(f"{r['image']:<30} {r['clip_score']:>12.4f}")
    
    scores = [r["clip_score"] for r in results]
    print("-"*60)
    print(f"{'Average':<30} {sum(scores)/len(scores):>12.4f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="CLIP Score Evaluation")
    parser.add_argument("--image", type=str, default=None,
                        help="Single image path for evaluation")
    parser.add_argument("--text", type=str, default=None,
                        help="Text prompt for single image evaluation")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory of images for batch evaluation")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="Text file with one prompt per line (for batch eval)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional: save results to JSON file")
    args = parser.parse_args()
    
    model, processor, device = load_clip()
    
    if args.image and args.text:
        # Single pair evaluation
        score = evaluate_pair(args.image, args.text, model, processor, device)
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump([{"image": args.image, "prompt": args.text, "clip_score": score}], f, indent=2)
    
    elif args.image_dir and args.prompts_file:
        # Batch evaluation
        with open(args.prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        results = evaluate_directory(args.image_dir, prompts, model, processor, device)
        print_summary(results)
        
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved → {args.output_json}")
    
    else:
        # Default: run on Phase 1 experiment prompts & outputs
        print("Running default evaluation on Phase 1 prompts...")
        default_prompts = [
            "A futuristic robot walking in a cyberpunk city, neon lights, highly detailed",
            "A cute cat wearing stylish sunglasses sitting on a beach chair",
            "A red sports car parked in snowy mountains, cinematic lighting",
            "An astronaut riding a horse on Mars, surreal digital art",
        ]
        for prompt in default_prompts:
            print(f"  (No images found — demo mode) Prompt: {prompt[:60]}")
            print(f"  Expected CLIP Score: ~0.28–0.34")


if __name__ == "__main__":
    main()
