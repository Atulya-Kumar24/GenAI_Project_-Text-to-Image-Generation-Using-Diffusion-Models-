"""
edit_image.py — Semantic Image Editing using InstructPix2Pix
IIT Indore | GenAI Course Project

InstructPix2Pix allows editing a real image using a natural language instruction,
e.g., "make it look like it's snowing" or "turn the car red".
"""

import os
import argparse
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


def load_pipeline(device: str = None):
    """Load InstructPix2Pix pipeline."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_id = "timbrooks/instruct-pix2pix"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading InstructPix2Pix on {device}...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe, device


def edit_image(pipe, image: Image.Image, edit_prompt: str,
               num_inference_steps: int = 50,
               image_guidance_scale: float = 1.5,
               text_guidance_scale: float = 7.5,
               seed: int = 42) -> Image.Image:
    """
    Edit an input image using a text instruction.
    
    Args:
        pipe: InstructPix2Pix pipeline
        image: Input PIL Image
        edit_prompt: Text instruction, e.g. "make it snowy"
        image_guidance_scale: How strongly to preserve original image (1.0–2.5)
        text_guidance_scale: How strongly to follow text instruction (5.0–10.0)
    
    Returns:
        Edited PIL Image
    """
    # Resize to multiples of 8 (required by the model)
    w, h = image.size
    w = (w // 8) * 8
    h = (h // 8) * 8
    image = image.resize((w, h))

    generator = torch.Generator().manual_seed(seed)
    result = pipe(
        edit_prompt,
        image=image,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=text_guidance_scale,
        generator=generator
    )
    return result.images[0]


def main():
    parser = argparse.ArgumentParser(description="Semantic Image Editing with InstructPix2Pix")
    parser.add_argument("--input_image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--edit_prompt", type=str, required=True,
                        help="Edit instruction, e.g. 'make it snowy'")
    parser.add_argument("--output", type=str, default="outputs/edited.png",
                        help="Output image path")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--img_guidance", type=float, default=1.5,
                        help="Image guidance scale (higher = preserve more)")
    parser.add_argument("--text_guidance", type=float, default=7.5,
                        help="Text guidance scale (higher = follow prompt more)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image = Image.open(args.input_image).convert("RGB")
    pipe, device = load_pipeline()
    
    print(f"Editing image: '{args.edit_prompt}'")
    edited = edit_image(
        pipe, image, args.edit_prompt,
        num_inference_steps=args.steps,
        image_guidance_scale=args.img_guidance,
        text_guidance_scale=args.text_guidance,
        seed=args.seed
    )
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    edited.save(args.output)
    print(f"Saved edited image → {args.output}")


if __name__ == "__main__":
    main()
