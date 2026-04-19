"""
generate.py — Text-to-Image Generation using Stable Diffusion v1.5
IIT Indore | GenAI Course Project
"""

import os
import argparse
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def load_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5", device: str = None):
    """Load Stable Diffusion pipeline."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading model '{model_id}' on {device}...")
    
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.safety_checker = None  # disable for research use
    return pipe, device


def generate_image(pipe, prompt: str, num_inference_steps: int = 50,
                   guidance_scale: float = 7.5, seed: int = 42) -> Image.Image:
    """Generate a single image from a text prompt."""
    generator = torch.Generator().manual_seed(seed)
    result = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )
    return result.images[0]


def run_batch(pipe, prompts: list, output_dir: str, **kwargs):
    """Generate images for a list of prompts and save them."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] Generating: {prompt[:60]}...")
        image = generate_image(pipe, prompt, **kwargs)
        
        filename = f"generated_{i+1:02d}.png"
        save_path = os.path.join(output_dir, filename)
        image.save(save_path)
        print(f"  Saved → {save_path}")
        results.append((prompt, save_path, image))
    
    return results


# Default experiment prompts (from Phase 1)
DEFAULT_PROMPTS = [
    "A futuristic robot walking in a cyberpunk city, neon lights, highly detailed",
    "A cute cat wearing stylish sunglasses sitting on a beach chair",
    "A red sports car parked in snowy mountains, cinematic lighting",
    "An astronaut riding a horse on Mars, surreal digital art",
]


def main():
    parser = argparse.ArgumentParser(description="Text-to-Image Generation with Stable Diffusion")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to generate. If not set, runs default experiment prompts.")
    parser.add_argument("--output", type=str, default="outputs/generated",
                        help="Output directory for images")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="HuggingFace model ID")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    pipe, device = load_pipeline(args.model)
    
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    
    run_batch(
        pipe, prompts, args.output,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed
    )
    print("\nDone! All images saved.")


if __name__ == "__main__":
    main()
