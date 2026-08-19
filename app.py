"""
app.py
======
DeepVision-AI Main Streamlit Application.

**Architecture**: Dual-Model Ensemble (ViT-Base + ConvNeXt-Tiny) + Test-Time Augmentation + EXIF Bayesian Fusion.
**Hardware Target**: Lenovo LOQ (i5-12450HX, 16GB RAM, RTX 3050 4GB VRAM) -> ~700MB VRAM FP16.
**Deployment**: Free on Streamlit Community Cloud / Hugging Face Spaces.

Author: B.Tech CSE(AI) 2nd Year Student
Institute: ABESIT Ghaziabad (AKTU Affiliated)
"""

import streamlit as st
import io
import json
from PIL import Image
from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# Local Module Imports (Ensure these files exist in the same directory)
# ---------------------------------------------------------------------------
from database import (
    init_db, register_user, login_user, delete_user,
    save_analysis, get_user_history
)
from ai_model import (
    load_model,                 # Returns List[Dict] (Ensemble Models)
    predict_ensemble,           # Runs ViT + ConvNeXt + TTA
    apply_exif_fusion,          # Bayesian Fusion with Metadata
    extract_exif,
    generate_analysis_text,
    create_thumbnail,
    DEVICE, USE_HALF, AI_THRESHOLD
)
from pdf_generator import generate_report_pdf

# ---------------------------------------------------------------------------
# Page Configuration (Must be first Streamlit command)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DeepVision-AI | Ensemble AI Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/DeepVision-AI",
        "Report a bug": "https://github.com/yourusername/DeepVision-AI/issues",
        "About": "DeepVision-AI - B.Tech CSE(AI) Project, ABESIT Ghaziabad"
    }
)

# ---------------------------------------------------------------------------
# Custom CSS (Clean, Mobile-Responsive, Professional)
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }
    
    /* Verdict Badges */
    .verdict-badge { display: inline-block; padding: 0.5rem 1.2rem; border-radius: 50px; font-weight: 600; font-size: 1.1rem; text-align: center; }
    .verdict-ai { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    .verdict-real { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .verdict-uncertain { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }

    /* Sidebar */
    .sidebar-user { font-size: 0.85rem; color: #6b7280; margin-bottom: 1rem; }
    .stButton > button { border-radius: 8px; font-weight: 500; transition: all 0.2s; }
    .stDownloadButton > button { border-radius: 8px; }
    
    /* Progress Bar Colors (CSS Injection for specific bars) */
    /* Note: Streamlit progress bars are hard to style individually via CSS reliably without keys. 
       We rely on the default green/red logic or custom HTML bars below. */
       
    .prob-container { margin: 0.5rem 0; }
    .prob-label { display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 500; margin-bottom: 0.25rem; }
    .prob-bar-bg { height: 10px; border-radius: 5px; background: #e5e7eb; overflow: hidden; }
    .prob-bar-fill { height: 100%; border-radius: 5px; transition: width 0.6s ease-out; }
    .bar-ai { background: linear-gradient(90deg, #ef4444, #f87171); }
    .bar-real { background: linear-gradient(90deg, #22c55e, #4ade80); }

    .metric-card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem; text-align: center; }
    .metric-value { font-size: 2rem; font-weight: 700; line-height: 1.2; }
    .metric-label { font-size: 0.85rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }
    
    .ensemble-badge { background: #e0e7ff; color: #3730a3; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin-left: 0.5rem; vertical-align: middle; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "authenticated": False, "user_id": None, "username": None,
        "page": "Home", "analysis_result": None,
        "uploaded_image_bytes": None, "uploaded_image_name": None,
        "show_pdf_download": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def logout():
    for key in list(st.session_state.keys()): del st.session_state[key]
    st.rerun()

# ---------------------------------------------------------------------------
# Cached Resources (Ensemble Model Loading)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Dual-Model Ensemble (ViT + ConvNeXt)... ~700MB VRAM (FP16)")
def get_model() -> List[Dict]:
    """Load and cache BOTH models (ViT + ConvNeXt) + Processors."""
    return load_model()  # Returns list of {model, processor, config}

# ---------------------------------------------------------------------------
# Authentication Pages
# ---------------------------------------------------------------------------

def show_auth_page():
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #1e3a5f; margin-bottom: 0.5rem;">🔬 DeepVision-AI</h1>
        <p style="color: #6b7280; font-size: 1.1rem;">Ensemble AI vs Real Image Detection</p>
        <p style="color: #9ca3af; font-size: 0.9rem;"><span class="ensemble-badge">ViT + ConvNeXt + TTA</span> B.Tech CSE(AI) Project • ABESIT Ghaziabad</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔐 Sign In", "📝 Sign Up"])

    with tab1:
        with st.form("login_form", clear_on_submit=False):
            st.markdown("### Welcome Back")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            if st.form_submit_button("Sign In", use_container_width=True, type="primary"):
                if not username or not password: st.error("Please enter both username and password.")
                else:
                    success, msg, user = login_user(username, password)
                    if success:
                        st.session_state.authenticated = True; st.session_state.user_id = user["id"]; st.session_state.username = user["username"]
                        st.success(msg); st.rerun()
                    else: st.error(msg)

    with tab2:
        with st.form("signup_form", clear_on_submit=False):
            st.markdown("### Create Account")
            new_username = st.text_input("Username", placeholder="Choose a username", key="su_username")
            new_email = st.text_input("Email", placeholder="your@email.com", key="su_email")
            new_password = st.text_input("Password", type="password", placeholder="Min 6 characters", key="su_pass")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="su_confirm")
            if st.form_submit_button("Create Account", use_container_width=True, type="primary"):
                if not new_username or not new_email or not new_password: st.error("All fields are required.")
                elif new_password != confirm_password: st.error("Passwords do not match.")
                elif len(new_password) < 6: st.error("Password must be at least 6 characters.")
                else:
                    success, msg = register_user(new_username, new_email, new_password)
                    if success: st.success(msg); st.info("Please switch to the **Sign In** tab to log in.")
                    else: st.error(msg)

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("""<div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #1e3a5f; margin: 0;">🔬 DeepVision-AI</h2>
            <p style="color: #9ca3af; margin: 0; font-size: 0.8rem;">Ensemble AI Detector</p>
            <span class="ensemble-badge" style="margin-top: 0.5rem; display: inline-block;">ViT + ConvNeXt + TTA</span>
        </div>""", unsafe_allow_html=True)
        st.divider()
        st.markdown(f'<div class="sidebar-user">👤 Logged in as <strong>{st.session_state.username}</strong></div>', unsafe_allow_html=True)
        
        nav_options = {"🏠 Home": "Home", "📜 History": "History", "ℹ️ About": "About", "⚙️ Settings": "Settings"}
        for label, page_key in nav_options.items():
            if st.button(label, use_container_width=True, type="primary" if st.session_state.page == page_key else "secondary"):
                st.session_state.page = page_key; st.rerun()
        
        st.divider()
        if st.button("🚪 Logout", use_container_width=True): logout()
        
        # System Info
        st.caption(f"**Device:** {'🚀 GPU (CUDA)' if DEVICE.type == 'cuda' else '💻 CPU'} | **FP16:** {USE_HALF} | **AI Threshold:** {AI_THRESHOLD:.0%}")
        st.markdown("""<div style="text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;">
            <p style="font-size: 0.7rem; color: #9ca3af; margin: 0;">DeepVision-AI v2.0 (Ensemble)<br>B.Tech CSE(AI) 2nd Year<br>ABESIT Ghaziabad (AKTU)</p></div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Page: Home (Main Detection Interface)
# ---------------------------------------------------------------------------

def page_home():
    st.markdown("## 🏠 Home — Ensemble AI Detection")
    
    # Load Models
    models = get_model()
    device_str = "🚀 GPU (CUDA)" if DEVICE.type == "cuda" else "💻 CPU"
    model_names = ", ".join([m["config"]["id"] for m in models])
    st.caption(f"Ensemble: **{model_names}** | Device: {device_str} | FP16: {USE_HALF} | TTA: Enabled")
    st.divider()

    col_left, col_center, col_right = st.columns([1, 3, 1])

    with col_center:
        st.markdown("### 📤 Upload an Image")
        uploaded_file = st.file_uploader("Drag & drop or click to browse", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"], label_visibility="collapsed")

        if uploaded_file:
            image_bytes = uploaded_file.getvalue()
            st.session_state.uploaded_image_bytes = image_bytes
            st.session_state.uploaded_image_name = uploaded_file.name
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, caption=f"Preview: {uploaded_file.name}", use_container_width=True)

            if st.button("🔬 Run Ensemble Scan", use_container_width=True, type="primary"):
                with st.spinner("Running Dual-Model Inference (ViT + ConvNeXt) + TTA + EXIF Fusion..."):
                    try:
                        # 1. Ensemble Inference (Raw Probabilities)
                        raw_pred = predict_ensemble(models, image)
                        
                        # 2. EXIF Extraction
                        exif_result = extract_exif(image_bytes)
                        
                        # 3. Bayesian Fusion (Model + Metadata)
                        prediction = apply_exif_fusion(raw_pred, exif_result)
                        
                        # 4. Generate Report Text
                        analysis_text = generate_analysis_text(prediction, exif_result, uploaded_file.name)
                        
                        # 5. Thumbnail & Save to DB
                        thumb_bytes = create_thumbnail(image_bytes)
                        analysis_id = save_analysis(
                            user_id=st.session_state.user_id, image_name=uploaded_file.name, image_data=thumb_bytes,
                            ai_prob=prediction["ai_probability"], real_prob=prediction["real_probability"],
                            verdict=prediction["verdict"], confidence=prediction["confidence"],
                            exif_summary=json.dumps(exif_result), analysis_text=analysis_text
                        )
                        
                        # Store for UI
                        st.session_state.analysis_result = {
                            "id": analysis_id, "prediction": prediction, "exif_result": exif_result,
                            "analysis_text": analysis_text, "image_bytes": image_bytes, "image_name": uploaded_file.name
                        }
                        st.session_state.show_pdf_download = True
                        st.success("Ensemble Analysis Complete!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Inference Error: {e}")
                        st.exception(e)

        # --- Results Display ---
        if st.session_state.analysis_result:
            res = st.session_state.analysis_result
            pred = res["prediction"]; exif = res["exif_result"]
            st.divider(); st.markdown("## 📊 Ensemble Analysis Results")

            # Top Row: Verdict + Confidence Gauge
            c_verdict, c_gauge = st.columns([2, 1])
            
            with c_verdict:
                verdict_text = pred["verdict"]
                conf = pred["confidence"]
                
                if "AI Generated" in verdict_text and "Low" not in verdict_text:
                    v_class, icon = "verdict-ai", "🤖"
                elif "Low Confidence" in verdict_text or "Ambiguous" in verdict_text:
                    v_class, icon = "verdict-uncertain", "⚠️"
                else:
                    v_class, icon = "verdict-real", "📷"
                
                st.markdown(f'<div class="verdict-badge {v_class}">{icon} {verdict_text} <br><small>{conf:.1f}% Confidence</small></div>', unsafe_allow_html=True)
                
                st.markdown("**Ensemble Probability Breakdown**")
                # Custom HTML Bars for precise coloring
                ai_p = pred["ai_probability"]; real_p = pred["real_probability"]
                st.markdown(f"""
                <div class="prob-container">
                    <div class="prob-label"><span>🤖 AI Generated</span><span>{ai_p*100:.1f}%</span></div>
                    <div class="prob-bar-bg"><div class="prob-bar-fill bar-ai" style="width: {ai_p*100}%"></div></div>
                </div>
                <div class="prob-container">
                    <div class="prob-label"><span>📷 Real Photograph</span><span>{real_p*100:.1f}%</span></div>
                    <div class="prob-bar-bg"><div class="prob-bar-fill bar-real" style="width: {real_p*100}%"></div></div>
                </div>
                """, unsafe_allow_html=True)

            with c_gauge:
                # Confidence Meter
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                conf_color = "#ef4444" if "AI" in verdict_text and "Low" not in verdict_text else ("#f59e0b" if "Low" in verdict_text else "#22c55e")
                st.markdown(f"<div class='metric-label'>Confidence</div><div class='metric-value' style='color:{conf_color}'>{conf:.1f}%</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                st.caption(f"Threshold: {AI_THRESHOLD:.0%} | Models: {len(models)} | TTA: {'On' if True else 'Off'}")

            # EXIF & Analysis Expanders
            exp_col1, exp_col2 = st.columns(2)
            
            with exp_col1:
                with st.expander("📋 EXIF Metadata & Forensics", expanded=True):
                    tags = exif.get("tags", {}); flags = exif.get("suspicious_flags", [])
                    if not exif.get("has_exif"):
                        st.warning("⚠️ No EXIF metadata found.")
                        st.caption("Common for social media downloads (stripped) or AI generations. Fusion treats this as neutral.")
                    else:
                        cols = st.columns(2)
                        fields = [("Camera Make", "Camera Model"), ("Lens Model", "Software"), ("Date Taken", "Aperture (f-stop)"), ("Exposure Time", "ISO"), ("Focal Length", "GPS Latitude")]
                        for i, (l, r) in enumerate(fields):
                            with cols[i % 2]: 
                                st.markdown(f"**{l}:** {tags.get(l, '—')}")
                                st.markdown(f"**{r}:** {tags.get(r, '—')}")
                    
                    if flags:
                        st.markdown("### 🔍 Suspicious Indicators")
                        for f in flags: st.error(f)
                    elif exif.get("has_exif"): st.success("✅ No obvious metadata anomalies detected.")

            with exp_col2:
                with st.expander("📝 Detailed Explanation (Fusion Logic)", expanded=True):
                    st.markdown(res["analysis_text"])

            # PDF Download
            if st.session_state.show_pdf_download:
                pdf_bytes = generate_report_pdf(res["image_bytes"], res["image_name"], pred, exif, res["analysis_text"], st.session_state.username)
                st.download_button("📄 Download PDF Report", pdf_bytes, f"DeepVision_Report_{res['image_name']}.pdf", "application/pdf", use_container_width=True)

            if st.button("🔄 Scan Another Image", use_container_width=True):
                for k in ["analysis_result", "uploaded_image_bytes", "uploaded_image_name", "show_pdf_download"]: st.session_state[k] = None if k != "show_pdf_download" else False
                st.rerun()

# ---------------------------------------------------------------------------
# Page: History
# ---------------------------------------------------------------------------

def page_history():
    st.markdown("## 📜 Analysis History")
    history = get_user_history(st.session_state.user_id, limit=100)
    if not history: st.info("No history yet. Go to **Home** and run your first Ensemble Scan!"); return
    st.caption(f"Showing {len(history)} recent analyses (Ensemble v2.0)")
    
    for rec in history:
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([1, 3, 2, 1.5, 1])
            with c1:
                if rec["image_data"]:
                    try: st.image(Image.open(io.BytesIO(rec["image_data"])), use_container_width=True)
                    except: st.write("🖼️")
                else: st.write("🖼️")
            with c2:
                st.markdown(f"**{rec['image_name']}**")
                v_class = "verdict-ai" if "AI Generated" in rec["verdict"] and "Low" not in rec["verdict"] else ("verdict-uncertain" if "Low" in rec["verdict"] else "verdict-real")
                icon = "🤖" if "AI Generated" in rec["verdict"] else ("⚠️" if "Low" in rec["verdict"] else "📷")
                st.markdown(f'<span class="verdict-badge {v_class}" style="font-size:0.85rem; padding:0.25rem 0.75rem;">{icon} {rec["verdict"]}</span>', unsafe_allow_html=True)
                st.caption(f"Confidence: {rec['confidence']:.1f}%")
            with c3:
                st.markdown(f"**AI:** {rec['ai_probability']*100:.1f}%"); st.progress(rec["ai_probability"])
                st.markdown(f"**Real:** {rec['real_probability']*100:.1f}%"); st.progress(rec["real_probability"])
            with c4: st.caption(f"📅 {rec['created_at'][:16].replace('T', ' ')}")
            with c5:
                if st.button("📄 Report", key=f"rep_{rec['id']}", use_container_width=True):
                    exif_data = json.loads(rec["exif_summary"]) if rec["exif_summary"] else {}
                    pdf_b = generate_report_pdf(rec["image_data"] or b"", rec["image_name"], 
                        {"ai_probability": rec["ai_probability"], "real_probability": rec["real_probability"], "verdict": rec["verdict"], "confidence": rec["confidence"]},
                        exif_data, rec["analysis_text"] or "No analysis stored.", st.session_state.username)
                    st.download_button("⬇️ Download", pdf_b, f"DeepVision_Report_{rec['image_name']}.pdf", "application/pdf", key=f"dl_{rec['id']}", use_container_width=True)

# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------

def page_about():
    st.markdown("## ℹ️ About DeepVision-AI v2.0 (Ensemble)")
    st.markdown("""**DeepVision-AI** detects **AI-generated** vs **Real** photographs using a **Dual-Model Ensemble** with **Test-Time Augmentation (TTA)** and **Metadata Fusion**.
Built as a **B.Tech CSE (AI) 2nd Year** project at **ABESIT Ghaziabad** (AKTU).""")
    st.divider()
    
    t1, t2, t3 = st.columns(3)
    with t1: 
        st.markdown("""**🧠 Ensemble Architecture**
- **Model A (ViT-Base)**: `Organika/sdxl-detector` (~86M). Superior global texture analysis via Self-Attention.
- **Model B (ConvNeXt-Tiny)**: `dima806/deepfake_vs_real_image_detection` (~28M). Robust local feature extraction, resistant to compression.
- **Fusion**: Weighted Average (60/40) + Logit-space TTA (Flip Augmentation).""")
    with t2: 
        st.markdown("""**🔬 Advanced Techniques**
- **Test-Time Augmentation (TTA)**: Averages logits of Original + Horizontal Flip. Reduces variance, +2-3% Acc.
- **Calibrated Thresholding**: 55% AI Threshold (vs 50%) minimizes False Positives on Real photos.
- **EXIF Bayesian Fusion**: Valid Camera Metadata (Make+Model+Lens+Exposure) nudges probability toward Real (-15% AI Prob). AI Software tags nudge toward AI (+20%).""")
    with t3: 
        st.markdown("""**🛠️ Tech Stack**
- **Frontend/Backend**: Streamlit (Pure Python, Mobile Ready)
- **AI/ML**: PyTorch, Hugging Face Transformers, TIMM
- **Database**: SQLite (Auth + History + Thumbnails)
- **Forensics**: Pillow, Exifread
- **Reporting**: fpdf2 (Zero Dependency PDF)
- **Deploy**: Streamlit Cloud / HF Spaces (Free)""")

    st.divider()
    with st.expander("📚 Model Cards & Labels"):
        st.markdown("""
        | Model | Repo | Arch | Params | Input | Labels (id2label) | VRAM (FP16) |
        |---|---|---|---|---|---|---|
        | **ViT (Primary)** | `Organika/sdxl-detector` | ViT-B/16 | 86M | 224² | 0: Artificial, 1: Real | ~350 MB |
        | **CNN (Secondary)** | `dima806/deepfake_vs_real_image_detection` | ConvNeXt-Tiny | 28M | 224² | 0: Fake, 1: Real | ~150 MB |
        | **Total** | **Ensemble** | **Hybrid** | **~114M** | **224²** | **Fused** | **~500-700 MB** |
        
        *Note: Labels mapped dynamically at runtime via `config.id2label`.*""")
    
    st.divider(); st.markdown("### 👨‍💻 Developed By")
    st.markdown("""**Name**: [Your Name Here]  \n**Program**: B.Tech CSE (AI) 2nd Year  \n**Institute**: **ABESIT Ghaziabad** (AKTU)  \n**Session**: 2024–2028""")
    st.divider(); st.caption("DeepVision-AI v2.0 • Educational Research • Not for Legal/Forensic Use")

# ---------------------------------------------------------------------------
# Page: Settings
# ---------------------------------------------------------------------------

def page_settings():
    st.markdown("## ⚙️ Settings")
    st.markdown(f"**Username:** `{st.session_state.username}`  \n**User ID:** `{st.session_state.user_id}`"); st.divider()
    st.markdown("### 🗑️ Danger Zone"); st.warning("Irreversible. All history deleted.")
    with st.form("del_form"):
        confirm = st.text_input("Type username to confirm", placeholder=st.session_state.username)
        if st.form_submit_button("🗑️ Delete My Account", type="secondary"):
            if confirm == st.session_state.username:
                ok, msg = delete_user(st.session_state.user_id)
                if ok: st.success(msg); st.info("Logging out..."); logout()
                else: st.error(msg)
            else: st.error("Username mismatch.")

# ---------------------------------------------------------------------------
# Main Router
# ---------------------------------------------------------------------------

def main():
    init_db(); init_session_state()
    if not st.session_state.authenticated: show_auth_page(); return
    render_sidebar()
    {"Home": page_home, "History": page_history, "About": page_about, "Settings": page_settings}.get(st.session_state.page, page_home)()

if __name__ == "__main__":
    main()
