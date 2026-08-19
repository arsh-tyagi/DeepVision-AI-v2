"""
ai_model.py
===========
AI Model Handler for DeepVision-AI.

Uses PUBLIC model: Organika/sdxl-detector (ViT-Base-Patch16-224).
No authentication required. Optimized for 4GB VRAM (FP16 on CUDA).
Labels: 0 -> Artificial (AI), 1 -> Real (Handled dynamically via config).
"""

import io
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
from typing import Dict, Tuple, Optional, List, Any

# ---------------------------------------------------------------------------
# Configuration - VERIFIED PUBLIC MODEL
# ---------------------------------------------------------------------------
MODEL_REPO = "Organika/sdxl-detector"  # Public, ViT-Base, ~350MB VRAM FP16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_HALF = DEVICE.type == "cuda"

# ---------------------------------------------------------------------------
# Model Loading (Cached by Streamlit in app.py)
# ---------------------------------------------------------------------------

def load_model() -> Tuple[AutoModelForImageClassification, AutoImageProcessor]:
    """
    Loads the public ViT model and its preprocessor from Hugging Face Hub.
    Returns (model, processor).
    """
    print(f"Loading model from {MODEL_REPO} on {DEVICE}...")
    
    # Load processor (handles resize, normalize, to tensor)
    processor = AutoImageProcessor.from_pretrained(MODEL_REPO)
    
    # Load model
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.float16 if USE_HALF else torch.float32,
        low_cpu_mem_usage=True
    )
    
    model.to(DEVICE).eval()
    
    # Print label mapping for debugging (shows in console)
    print(f"Model Labels (id2label): {model.config.id2label}")
    
    # Warm-up run
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224, device=DEVICE, dtype=torch.float16 if USE_HALF else torch.float32)
        _ = model(dummy)
    
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        
    print("Model loaded successfully.")
    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def predict_image(model: AutoModelForImageClassification, processor: AutoImageProcessor, image: Image.Image) -> Dict:
    """
    Run inference on a single PIL image using the HF processor.
    Dynamically maps logits to 'AI Generated' / 'Real Photograph' using model.config.id2label.
    """
    # Preprocess using the model's specific processor
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    if USE_HALF:
        inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

    # Forward pass
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1).squeeze(0).float().cpu().numpy()

    # --- Dynamic Label Mapping (Robust) ---
    # Model config usually: {0: 'artificial', 1: 'real'} or {0: 'fake', 1: 'real'}
    id2label = model.config.id2label
    
    # Find indices for AI and Real
    ai_idx = None
    real_idx = None
    
    for idx, label in id2label.items():
        label_lower = label.lower()
        if any(kw in label_lower for kw in ["artificial", "fake", "ai", "generated", "synthetic"]):
            ai_idx = idx
        elif any(kw in label_lower for kw in ["real", "photo", "natural", "human"]):
            real_idx = idx
    
    # Fallback: Assume 0=AI, 1=Real if heuristic fails (common for binary classifiers)
    if ai_idx is None: ai_idx = 0
    if real_idx is None: real_idx = 1

    ai_prob = float(probs[ai_idx])
    real_prob = float(probs[real_idx])

    if real_prob > ai_prob:
        verdict = "Real Photograph"
        confidence = real_prob
    else:
        verdict = "AI Generated"
        confidence = ai_prob

    # Clear cache
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "ai_probability": ai_prob,
        "real_probability": real_prob,
        "verdict": verdict,
        "confidence": confidence * 100,
    }


# ---------------------------------------------------------------------------
# EXIF Metadata Extraction (Unchanged)
# ---------------------------------------------------------------------------

def extract_exif(image_bytes: bytes) -> Dict[str, Any]:
    import exifread
    tags = {}
    suspicious_flags = []
    try:
        file_obj = io.BytesIO(image_bytes)
        exif_data = exifread.process_file(file_obj, details=False)
        tag_mapping = {
            "Image Make": "Camera Make", "Image Model": "Camera Model", "Image Software": "Software",
            "Image DateTime": "Date Taken", "EXIF LensModel": "Lens Model", "EXIF FNumber": "Aperture (f-stop)",
            "EXIF ExposureTime": "Exposure Time", "EXIF ISOSpeedRatings": "ISO", "EXIF FocalLength": "Focal Length",
            "GPS GPSLatitude": "GPS Latitude", "GPS GPSLongitude": "GPS Longitude",
            "Image Artist": "Artist", "Image Copyright": "Copyright",
        }
        for exif_tag, friendly_name in tag_mapping.items():
            if exif_tag in exif_data: tags[friendly_name] = str(exif_data[exif_tag])
        
        software = tags.get("Software", "").lower()
        ai_keywords = ["midjourney", "stable diffusion", "dall-e", "dalle", "firefly", "imagen", "stablediffusion", "comfyui", "automatic1111", "forge", "pillow", "pil", "opencv", "adobe firefly", "sdxl"]
        for kw in ai_keywords:
            if kw in software:
                suspicious_flags.append(f"Software tag indicates AI tool: '{tags['Software']}'")
                break
        
        critical = ["Camera Make", "Camera Model", "Lens Model", "Exposure Time", "ISO", "Focal Length"]
        missing = [f for f in critical if f not in tags]
        if len(missing) >= 4: suspicious_flags.append(f"Missing {len(missing)} critical camera EXIF fields (Make, Model, Lens, Exposure, ISO, Focal Length)")
        if "Date Taken" in tags and "Camera Make" not in tags: suspicious_flags.append("Date present but no camera manufacturer info")
        if "Camera Make" in tags and any(p in tags["Camera Make"].lower() for p in ["iphone", "pixel", "galaxy", "xiaomi", "oneplus"]):
            if "GPS Latitude" not in tags and "GPS Longitude" not in tags:
                suspicious_flags.append("Phone camera detected but no GPS data (may be disabled or stripped)")
    except Exception as e:
        tags["Error"] = f"EXIF parsing failed: {str(e)}"
    return {"tags": tags, "suspicious_flags": suspicious_flags, "has_exif": len(tags) > 0 and "Error" not in tags}


# ---------------------------------------------------------------------------
# Descriptive Report Generation (Unchanged Logic)
# ---------------------------------------------------------------------------

def generate_analysis_text(prediction: Dict, exif_result: Dict, image_name: str) -> str:
    ai_pct = prediction["ai_probability"] * 100
    real_pct = prediction["real_probability"] * 100
    verdict = prediction["verdict"]
    confidence = prediction["confidence"]
    lines = [
        f"Analysis Report for: {image_name}", "=" * 50, "",
        f"🤖 AI Probability: {ai_pct:.1f}%",
        f"📷 Real Probability: {real_pct:.1f}%",
        f"🎯 Verdict: {verdict} ({confidence:.1f}% confidence)", "",
        "--- Model Assessment ---"
    ]
    if verdict == "AI Generated":
        lines.append("The Vision Transformer (ViT-Base) model detected patterns consistent with AI-generated imagery: unnatural texture smoothness, inconsistent lighting physics, or statistical anomalies in pixel distributions differing from camera sensor noise.")
    else:
        lines.append("The model classified this as a real photograph. Global attention patterns and patch statistics align with natural camera-captured images.")
    lines.append(""); lines.append("--- Metadata Forensics ---")
    tags = exif_result["tags"]; flags = exif_result["suspicious_flags"]
    if not exif_result["has_exif"]:
        lines.append("⚠️ No EXIF metadata found. Common for social media downloads (stripped) or AI generations.")
    else:
        lines.append("✅ EXIF metadata detected. Key fields:")
        for k in ["Camera Make", "Camera Model", "Lens Model", "Software", "Date Taken"]:
            if k in tags: lines.append(f"   • {k}: {tags[k]}")
        if flags:
            lines.append(""); lines.append("🔍 Suspicious Indicators:")
            for f in flags: lines.append(f"   • {f}")
        else:
            lines.append(""); lines.append("✅ No obvious metadata anomalies detected.")
    lines.append(""); lines.append("--- Combined Assessment ---")
    if verdict == "AI Generated" and flags: lines.append("STRONG INDICATION OF AI GENERATION: Both deep learning model and metadata forensics agree.")
    elif verdict == "AI Generated": lines.append("MODEL INDICATES AI GENERATION: High confidence from ViT, but metadata missing/inconclusive.")
    elif verdict == "Real Photograph" and flags: lines.append("CONFLICTING SIGNALS: Model says Real, but metadata shows anomalies (heavy edit/screenshot/AI with injected metadata).")
    else: lines.append("CONSISTENT REAL PHOTOGRAPH: Both model and metadata suggest genuine camera capture.")
    lines.append(""); lines.append("--- Disclaimer ---")
    lines.append("Automated probabilistic analysis. Not 100% accurate. False positives/negatives occur with heavy compression, screenshots, advanced generative models, or unusual camera settings.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Thumbnail Generation (Unchanged)
# ---------------------------------------------------------------------------

def create_thumbnail(image_bytes: bytes, max_size: Tuple[int, int] = (256, 256)) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    if img.mode in ("RGBA", "LA", "P"): img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75, optimize=True)
    return buf.getvalue()
