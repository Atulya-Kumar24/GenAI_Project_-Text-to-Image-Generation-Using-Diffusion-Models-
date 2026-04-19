"""
run_experiments.py — Reproduce All Reported Experiments
IIT Indore | GenAI Course Project

Run this script to reproduce the results from Phase 1 and Phase 2.
Results are saved in outputs/ with a summary JSON.
"""

import os
import json
import torch
import argparse
from PIL import Image


# ─── Experiment Configs ───────────────────────────────────────────────────────

PHASE1_PROMPTS = [
    "A futuristic robot walking in a cyberpunk city, neon lights, highly detailed",
    "A cute cat wearing stylish sunglasses sitting on a beach chair",
    "A red sports car parked in snowy mountains, cinematic lighting",
    "An astronaut riding a horse on Mars, surreal digital art",
]

PHASE2_EDITS = [
    {
        "source_prompt": "A red sports car parked in a city",
        "edit_instruction": "make it look like it's snowing",
        "label": "car_snow"
    },
    {
        "source_prompt": "A portrait of a person outdoors",
        "edit_instruction": "add sunglasses",
        "label": "portrait_sunglasses"
    },
    {
        "source_prompt": "A park on a sunny afternoon",
        "edit_instruction": "turn it into night time",
        "label": "park_night"
    },
]


# ─── Phase 1: Text-to-Image ───────────────────────────────────────────────────

def run_phase1(output_dir: str = "outputs/phase1"):
    from src.generate import load_pipeline, run_batch
    
    print("\n" + "="*60)
    print("PHASE 1: Text-to-Image Generation")
    print("="*60)
    
    pipe, device = load_pipeline()
    results = run_batch(pipe, PHASE1_PROMPTS, output_dir, seed=42)
    
    # Compute CLIP scores
    from src.evaluate import load_clip, compute_clip_score
    model, processor, device = load_clip()
    
    scores = []
    for prompt, img_path, image in results:
        score = compute_clip_score(model, processor, image, prompt, device)
        print(f"CLIP Score [{score:.4f}]: {prompt[:50]}")
        scores.append({"prompt": prompt, "image": img_path, "clip_score": score})
    
    avg = sum(s["clip_score"] for s in scores) / len(scores)
    print(f"\nAverage CLIP Score (Phase 1): {avg:.4f}")
    
    return scores


# ─── Phase 2: Image Editing ───────────────────────────────────────────────────

def run_phase2(output_dir: str = "outputs/phase2"):
    """
    Phase 2 first generates source images, then edits them.
    This way the pipeline is fully self-contained.
    """
    from src.generate import load_pipeline, generate_image
    from src.edit_image import load_pipeline as load_edit_pipeline, edit_image
    from src.evaluate import load_clip, compute_clip_score
    
    print("\n" + "="*60)
    print("PHASE 2: Semantic Image Editing (InstructPix2Pix)")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    gen_pipe, device = load_pipeline()
    edit_pipe, _ = load_edit_pipeline(device)
    clip_model, clip_processor, _ = load_clip(device)
    
    results = []
    for exp in PHASE2_EDITS:
        label = exp["label"]
        
        # Generate source image
        print(f"\nGenerating source: '{exp['source_prompt'][:50]}'")
        source_img = generate_image(gen_pipe, exp["source_prompt"], seed=42)
        source_path = os.path.join(output_dir, f"{label}_source.png")
        source_img.save(source_path)
        
        # Edit
        print(f"Editing: '{exp['edit_instruction']}'")
        edited_img = edit_image(edit_pipe, source_img, exp["edit_instruction"], seed=42)
        edited_path = os.path.join(output_dir, f"{label}_edited.png")
        edited_img.save(edited_path)
        
        # Evaluate CLIP before and after
        score_before = compute_clip_score(
            clip_model, clip_processor, source_img, exp["edit_instruction"], device
        )
        score_after = compute_clip_score(
            clip_model, clip_processor, edited_img, exp["edit_instruction"], device
        )
        
        delta = score_after - score_before
        print(f"CLIP: {score_before:.4f} → {score_after:.4f}  (Δ = {delta:+.4f})")
        
        results.append({
            "label": label,
            "source_prompt": exp["source_prompt"],
            "edit_instruction": exp["edit_instruction"],
            "clip_before": score_before,
            "clip_after": score_after,
            "delta": delta,
            "source_image": source_path,
            "edited_image": edited_path
        })
    
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None,
                        help="Run only Phase 1 or Phase 2. Default: both.")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    
    all_results = {}
    
    if args.phase in (None, 1):
        p1 = run_phase1(os.path.join(args.output_dir, "phase1"))
        all_results["phase1"] = p1
    
    if args.phase in (None, 2):
        p2 = run_phase2(os.path.join(args.output_dir, "phase2"))
        all_results["phase2"] = p2
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "experiment_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ All experiments complete. Summary → {summary_path}")


if __name__ == "__main__":
    main()
