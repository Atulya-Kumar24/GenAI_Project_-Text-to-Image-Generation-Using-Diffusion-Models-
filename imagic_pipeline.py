"""
imagic_pipeline.py — Imagic-style Semantic Editing
IIT Indore | GenAI Course Project

Implements the 3-step Imagic approach:
  Step 1: Optimize text embedding to align with the input image
  Step 2: Fine-tune the diffusion model on the input image
  Step 3: Interpolate between optimized and target embeddings → generate edit

Reference: Kawar et al., "Imagic: Text-Based Real Image Editing with Diffusion Models", 2022
"""

import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from tqdm import tqdm


# ─── Helpers ────────────────────────────────────────────────────────────────

def load_image(path: str, size: int = 512) -> torch.Tensor:
    """Load and preprocess image to latent-ready tensor."""
    img = Image.open(path).convert("RGB").resize((size, size))
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return tfm(img).unsqueeze(0)  # (1, 3, H, W)


def encode_image_to_latent(vae, image_tensor: torch.Tensor, device: str) -> torch.Tensor:
    """Encode image to latent space using VAE."""
    image_tensor = image_tensor.to(device, dtype=torch.float16 if device == "cuda" else torch.float32)
    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.mean
    return latents * 0.18215  # scaling factor


def decode_latent_to_image(vae, latents: torch.Tensor) -> Image.Image:
    """Decode latents back to PIL image."""
    latents = latents / 0.18215
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    return Image.fromarray((image * 255).astype("uint8"))


# ─── Step 1: Text Embedding Optimization ────────────────────────────────────

def optimize_text_embedding(pipe, image_latents: torch.Tensor, target_text: str,
                             num_steps: int = 100, lr: float = 1e-3,
                             device: str = "cuda") -> torch.Tensor:
    """
    Optimize a text embedding so that the diffusion model would reconstruct
    the input image from it. This produces an 'image-aligned' embedding.
    """
    print("Step 1: Optimizing text embedding to align with input image...")
    
    # Get initial embedding for target text
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    
    inputs = tokenizer(target_text, return_tensors="pt", padding="max_length",
                       max_length=tokenizer.model_max_length, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        init_embedding = text_encoder(**inputs).last_hidden_state  # (1, seq, dim)
    
    # Make embedding trainable
    optimized_embedding = init_embedding.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([optimized_embedding], lr=lr)
    
    scheduler = pipe.scheduler
    scheduler.set_timesteps(50)
    
    for step in tqdm(range(num_steps), desc="Embedding optimization"):
        optimizer.zero_grad()
        
        # Sample random timestep
        t_idx = torch.randint(0, len(scheduler.timesteps), (1,)).item()
        t = scheduler.timesteps[t_idx].unsqueeze(0).to(device)
        
        # Add noise to latents
        noise = torch.randn_like(image_latents)
        noisy_latents = scheduler.add_noise(image_latents, noise, t)
        
        # Predict noise with current optimized embedding
        noise_pred = pipe.unet(
            noisy_latents,
            t,
            encoder_hidden_states=optimized_embedding
        ).sample
        
        # Reconstruction loss
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}/{num_steps} | Loss: {loss.item():.4f}")
    
    print("  Embedding optimization complete.")
    return optimized_embedding.detach()


# ─── Step 2: Fine-tune Diffusion Model ──────────────────────────────────────

def finetune_unet(pipe, image_latents: torch.Tensor, optimized_embedding: torch.Tensor,
                  num_steps: int = 150, lr: float = 5e-6, device: str = "cuda"):
    """
    Fine-tune the U-Net to reconstruct the specific input image.
    This captures image-specific appearance details.
    """
    print("Step 2: Fine-tuning U-Net on input image...")
    
    unet = pipe.unet
    unet.train()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    scheduler = pipe.scheduler
    scheduler.set_timesteps(50)
    
    for step in tqdm(range(num_steps), desc="U-Net fine-tuning"):
        optimizer.zero_grad()
        
        t_idx = torch.randint(0, len(scheduler.timesteps), (1,)).item()
        t = scheduler.timesteps[t_idx].unsqueeze(0).to(device)
        
        noise = torch.randn_like(image_latents)
        noisy_latents = scheduler.add_noise(image_latents, noise, t)
        
        noise_pred = unet(
            noisy_latents, t,
            encoder_hidden_states=optimized_embedding
        ).sample
        
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 30 == 0:
            print(f"  Step {step+1}/{num_steps} | Loss: {loss.item():.4f}")
    
    unet.eval()
    print("  Fine-tuning complete.")


# ─── Step 3: Semantic Interpolation & Generation ────────────────────────────

def generate_edited_image(pipe, optimized_embedding: torch.Tensor,
                           target_text: str, alpha: float = 0.7,
                           num_inference_steps: int = 50,
                           guidance_scale: float = 7.5,
                           device: str = "cuda",
                           seed: int = 42) -> Image.Image:
    """
    Interpolate between optimized embedding (image-aligned) and target text
    embedding, then generate the edited image.
    
    alpha=0 → pure target text, alpha=1 → original image reconstruction
    Typical best range: 0.6–0.8
    """
    print(f"Step 3: Generating edited image (alpha={alpha})...")
    
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    
    # Get target text embedding
    inputs = tokenizer(target_text, return_tensors="pt", padding="max_length",
                       max_length=tokenizer.model_max_length, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        target_embedding = text_encoder(**inputs).last_hidden_state
    
    # Interpolate
    interpolated = alpha * optimized_embedding + (1 - alpha) * target_embedding
    
    # Generate via denoising loop
    pipe.unet.eval()
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_inference_steps)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, pipe.unet.in_channels, 64, 64),
        generator=generator, device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    for t in tqdm(scheduler.timesteps, desc="Denoising"):
        with torch.no_grad():
            # Classifier-free guidance: uncond + cond
            uncond_input = tokenizer(
                [""], return_tensors="pt", padding="max_length",
                max_length=tokenizer.model_max_length
            )
            uncond_emb = text_encoder(**{k: v.to(device) for k, v in uncond_input.items()}).last_hidden_state
            
            model_input = torch.cat([latents] * 2)
            cond_input = torch.cat([uncond_emb, interpolated])
            
            noise_pred = pipe.unet(model_input, t, encoder_hidden_states=cond_input).sample
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    return decode_latent_to_image(pipe.vae, latents)


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def run_imagic(input_image_path: str, target_text: str, output_path: str,
               alpha: float = 0.7, device: str = None,
               embed_steps: int = 100, finetune_steps: int = 150):
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dtype = torch.float16 if device == "cuda" else torch.float32
    model_id = "runwayml/stable-diffusion-v1-5"
    
    print(f"Loading Stable Diffusion on {device}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype,
        scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    ).to(device)
    pipe.safety_checker = None
    
    # Load & encode input image
    image_tensor = load_image(input_image_path)
    image_latents = encode_image_to_latent(pipe.vae, image_tensor, device)
    
    # Run Imagic 3-step pipeline
    optimized_emb = optimize_text_embedding(
        pipe, image_latents, target_text,
        num_steps=embed_steps, device=device
    )
    finetune_unet(pipe, image_latents, optimized_emb, num_steps=finetune_steps, device=device)
    edited_image = generate_edited_image(
        pipe, optimized_emb, target_text, alpha=alpha, device=device
    )
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    edited_image.save(output_path)
    print(f"\nEdited image saved → {output_path}")
    return edited_image


def main():
    parser = argparse.ArgumentParser(description="Imagic-style Semantic Image Editing")
    parser.add_argument("--input_image", required=True, help="Path to real input image")
    parser.add_argument("--target_text", required=True, help="Target edit description")
    parser.add_argument("--output", default="outputs/imagic_edited.png")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Interpolation factor (0=pure text, 1=pure image). Try 0.6-0.8.")
    parser.add_argument("--embed_steps", type=int, default=100)
    parser.add_argument("--finetune_steps", type=int, default=150)
    args = parser.parse_args()
    
    run_imagic(
        args.input_image, args.target_text, args.output,
        alpha=args.alpha, embed_steps=args.embed_steps,
        finetune_steps=args.finetune_steps
    )


if __name__ == "__main__":
    main()
