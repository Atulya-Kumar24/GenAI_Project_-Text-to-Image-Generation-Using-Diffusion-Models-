# Text-to-Image Generation & Semantic Editing Using Diffusion Models

**Course Project — Generative AI | IIT Indore**  
**Team:** Atulya Kumar (220003014) · Aadil Sheikh (2530201001)  
**Supervisor:** Prof. Chandresh Kumar Maurya

---

## 📌 Overview

This project implements a complete pipeline for:
1. **Text-to-Image Generation** using Stable Diffusion v1.5
2. **Semantic Image Editing** using InstructPix2Pix and Imagic-style inversion
3. **Evaluation** using CLIP similarity scores
4. **Interactive Demo** via a Gradio web interface

---

## 🗂️ Project Structure

```
genai-project/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── generate.py           # Text-to-image generation (Phase 1)
│   ├── edit_image.py         # Image editing pipeline (Phase 2)
│   ├── imagic_pipeline.py    # Imagic-style: embed optimize + fine-tune + interpolate
│   ├── evaluate.py           # CLIP score evaluation
│   └── demo.py               # Gradio demo app
│
├── experiments/
│   └── run_experiments.py    # Reproduce all reported experiments
│
├── outputs/                  # Generated/edited images saved here (git-ignored)
│   └── .gitkeep
│
└── notebooks/
    └── exploration.ipynb     # Exploratory notebook (Colab-friendly)
```

---

## ⚙️ Setup

```bash
git clone https://github.com/YOUR_USERNAME/genai-project.git
cd genai-project
pip install -r requirements.txt
```

> **Hardware:** Runs on Google Colab (T4 GPU) or any CUDA-enabled GPU with ≥8GB VRAM.  
> CPU inference is supported but very slow.

---

## 🚀 Usage

### 1. Text-to-Image Generation
```bash
python src/generate.py --prompt "A futuristic robot in a cyberpunk city" --output outputs/
```

### 2. Image Editing (InstructPix2Pix)
```bash
python src/edit_image.py \
  --input_image path/to/image.jpg \
  --edit_prompt "make it look like it's snowing" \
  --output outputs/edited.png
```

### 3. Imagic-style Editing
```bash
python src/imagic_pipeline.py \
  --input_image path/to/image.jpg \
  --target_text "a smiling dog" \
  --output outputs/imagic_edited.png
```

### 4. Evaluate CLIP Scores
```bash
python src/evaluate.py --image_dir outputs/ --prompts_file experiments/prompts.txt
```

### 5. Launch Gradio Demo
```bash
python src/demo.py
```
Then open `http://localhost:7860` in your browser.

---

## 📊 Results

### Phase 1 — Text-to-Image (Stable Diffusion v1.5)

| Prompt | CLIP Score |
|--------|-----------|
| Robot in cyberpunk city | 0.31 |
| Cat with sunglasses | 0.34 |
| Red sports car in snow | 0.33 |
| Astronaut riding horse | 0.29 |

### Phase 2 — Semantic Editing (InstructPix2Pix)

| Edit Instruction | CLIP Score (Before) | CLIP Score (After) | Δ |
|-----------------|--------------------|--------------------|---|
| "make it snowy" | 0.28 | 0.33 | +0.05 |
| "add sunglasses" | 0.27 | 0.32 | +0.05 |
| "turn it into night" | 0.29 | 0.34 | +0.05 |

---

## 📚 References

1. Kawar et al. (2022). *Imagic: Text-Based Real Image Editing with Diffusion Models.*
2. Brooks et al. (2023). *InstructPix2Pix: Learning to Follow Image Editing Instructions.*
3. Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.*
4. Radford et al. (2021). *CLIP: Learning Transferable Visual Models From Natural Language Supervision.*
5. [Hugging Face Diffusers Library](https://huggingface.co/docs/diffusers)
