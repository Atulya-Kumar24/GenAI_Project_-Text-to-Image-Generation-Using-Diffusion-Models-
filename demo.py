"""
demo.py — Interactive Gradio Demo
IIT Indore | GenAI Course Project

Launches a web interface with two tabs:
  Tab 1: Text-to-Image Generation (Stable Diffusion)
  Tab 2: Image Editing (InstructPix2Pix)

Usage:
  python src/demo.py
  Then open http://localhost:7860
"""

import torch
import gradio as gr
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler
)

# ─── Model Loading ───────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"Running on: {DEVICE}")

print("Loading Stable Diffusion v1.5 for generation...")
gen_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=DTYPE,
    safety_checker=None
).to(DEVICE)

print("Loading InstructPix2Pix for editing...")
edit_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=DTYPE,
    safety_checker=None
).to(DEVICE)
edit_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(edit_pipe.scheduler.config)


# ─── Generation Function ─────────────────────────────────────────────────────

def generate_image(prompt: str, steps: int, guidance: float, seed: int):
    if not prompt.strip():
        return None, "⚠️ Please enter a prompt."
    
    generator = torch.Generator().manual_seed(int(seed))
    result = gen_pipe(
        prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator
    )
    image = result.images[0]
    return image, f"✅ Generated with seed {seed}"


# ─── Editing Function ────────────────────────────────────────────────────────

def edit_image_fn(input_image: Image.Image, edit_prompt: str,
                  steps: int, img_guidance: float, text_guidance: float, seed: int):
    if input_image is None:
        return None, "⚠️ Please upload an image."
    if not edit_prompt.strip():
        return None, "⚠️ Please enter an edit instruction."
    
    # Resize to model-friendly dimensions
    w, h = input_image.size
    w = (w // 8) * 8
    h = (h // 8) * 8
    input_image = input_image.resize((w, h))
    
    generator = torch.Generator().manual_seed(int(seed))
    result = edit_pipe(
        edit_prompt,
        image=input_image,
        num_inference_steps=int(steps),
        image_guidance_scale=float(img_guidance),
        guidance_scale=float(text_guidance),
        generator=generator
    )
    edited = result.images[0]
    return edited, f"✅ Edit applied: '{edit_prompt}'"


# ─── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="GenAI Project Demo | IIT Indore", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🎨 Text-to-Image Generation & Semantic Editing
    **IIT Indore | GenAI Course Project**  
    Atulya Kumar · Aadil Sheikh | Supervisor: Prof. Chandresh Kumar Maurya
    """)
    
    with gr.Tabs():
        
        # ── Tab 1: Text-to-Image ──────────────────────────────────────────────
        with gr.TabItem("🖼️ Text-to-Image Generation"):
            gr.Markdown("Generate images from text using **Stable Diffusion v1.5**")
            
            with gr.Row():
                with gr.Column():
                    gen_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="A futuristic robot in a cyberpunk city, neon lights...",
                        lines=3
                    )
                    with gr.Row():
                        gen_steps = gr.Slider(10, 100, value=50, step=5, label="Inference Steps")
                        gen_guidance = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")
                    gen_seed = gr.Number(value=42, label="Random Seed")
                    gen_btn = gr.Button("🚀 Generate", variant="primary")
                
                with gr.Column():
                    gen_output = gr.Image(label="Generated Image", type="pil")
                    gen_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Examples(
                examples=[
                    ["A futuristic robot walking in a cyberpunk city, neon lights, highly detailed", 50, 7.5, 42],
                    ["A cute cat wearing stylish sunglasses sitting on a beach chair", 50, 7.5, 123],
                    ["A red sports car parked in snowy mountains, cinematic lighting", 50, 7.5, 7],
                    ["An astronaut riding a horse on Mars, surreal digital art", 50, 7.5, 99],
                ],
                inputs=[gen_prompt, gen_steps, gen_guidance, gen_seed],
                label="📌 Experiment Prompts from Phase 1"
            )
            
            gen_btn.click(
                fn=generate_image,
                inputs=[gen_prompt, gen_steps, gen_guidance, gen_seed],
                outputs=[gen_output, gen_status]
            )
        
        # ── Tab 2: Image Editing ──────────────────────────────────────────────
        with gr.TabItem("✏️ Semantic Image Editing"):
            gr.Markdown("Edit a real image using natural language instructions via **InstructPix2Pix**")
            
            with gr.Row():
                with gr.Column():
                    edit_input = gr.Image(label="Upload Input Image", type="pil")
                    edit_prompt = gr.Textbox(
                        label="Edit Instruction",
                        placeholder="make it look like it's snowing",
                        lines=2
                    )
                    with gr.Row():
                        edit_steps = gr.Slider(10, 100, value=50, step=5, label="Steps")
                        edit_img_guidance = gr.Slider(1.0, 3.0, value=1.5, step=0.1,
                                                       label="Image Guidance (preserve original)")
                    edit_text_guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5,
                                                    label="Text Guidance (follow instruction)")
                    edit_seed = gr.Number(value=42, label="Random Seed")
                    edit_btn = gr.Button("✏️ Edit Image", variant="primary")
                
                with gr.Column():
                    edit_output = gr.Image(label="Edited Image", type="pil")
                    edit_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("""
            **Tips:**
            - *Image Guidance* = 1.2–2.0 → preserves original content more
            - *Text Guidance* = 6.0–10.0 → stronger edits
            - Try: "make it snowy", "turn it to night", "add a sunset", "make it look like a painting"
            """)
            
            edit_btn.click(
                fn=edit_image_fn,
                inputs=[edit_input, edit_prompt, edit_steps, edit_img_guidance,
                        edit_text_guidance, edit_seed],
                outputs=[edit_output, edit_status]
            )
    
    gr.Markdown("""
    ---
    *Built with 🤗 Diffusers, Stable Diffusion v1.5, InstructPix2Pix, and Gradio*
    """)


if __name__ == "__main__":
    demo.launch(share=True)  # share=True gives a public link (useful for Colab)
