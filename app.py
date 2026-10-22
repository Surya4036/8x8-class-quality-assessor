"""
8x8 Class Quality Assessment System - Streamlit Interactive Dashboard
AI-Powered Multimodal Quality & Pedagogical Evaluation Platform for Online Live Classes.
"""

import os
import sys
import tempfile
import pathlib
from typing import Dict, Any, Optional, List, Tuple
import streamlit as st
import pandas as pd

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.vtt_parser import VTTParser, VTTAnalysis
from src.video_analyzer import VideoAnalyzer, VideoAnalysis
from src.evaluator import ClassEvaluator, resolve_gemini_api_key, EvaluationResult


# --- Page Configuration ---
st.set_page_config(
    page_title="8x8 Class Quality Assessment System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Custom CSS Styling ---
st.markdown("""
<style>
    /* Main Theme Overrides */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    
    /* Grade Badge Styles */
    .badge-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        font-size: 1.8rem;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .badge-excellent { background-color: #10B981; }
    .badge-good { background-color: #22C55E; }
    .badge-average { background-color: #F59E0B; color: #1E293B; }
    .badge-weak { background-color: #F97316; }
    .badge-poor { background-color: #EF4444; }

    /* Metric Cards */
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card-title {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .metric-card-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
    }

    /* Pillar Cards */
    .pillar-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #3B82F6;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .pillar-title {
        font-weight: 600;
        font-size: 1rem;
        color: #1E293B;
    }
    .pillar-score {
        float: right;
        font-weight: 700;
        font-size: 1.1rem;
        color: #2563EB;
    }

    /* Evidence Timeline Cards */
    .evidence-item {
        background-color: #F8FAFC;
        border-left: 4px solid #0284C7;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        border-radius: 4px;
    }
    .evidence-ts {
        font-weight: 700;
        color: #0284C7;
        font-size: 0.9rem;
    }
    .evidence-pillar {
        background-color: #E0F2FE;
        color: #0369A1;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        margin-left: 0.5rem;
    }
    .evidence-quote {
        font-style: italic;
        color: #334155;
        margin-top: 0.4rem;
        padding-left: 0.8rem;
        border-left: 2px solid #CBD5E1;
    }

    /* List Containers */
    .strength-item {
        background-color: #F0FDF4;
        border-left: 4px solid #22C55E;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        border-radius: 4px;
        color: #14532D;
    }
    .improvement-item {
        background-color: #FFF7ED;
        border-left: 4px solid #F97316;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        border-radius: 4px;
        color: #7C2D12;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---
@st.cache_data
def discover_sample_recordings(base_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Scans class recordings directory and groups paired .vtt and .mp4 files by label.
    """
    recordings_dir = os.path.join(base_dir, "class recordings")
    samples = {}
    if not os.path.exists(recordings_dir):
        return samples

    for root, dirs, files in os.walk(recordings_dir):
        rel_folder = os.path.relpath(root, recordings_dir)
        if rel_folder == ".":
            continue
        
        vtt_files = [f for f in files if f.endswith('.vtt')]
        mp4_files = [f for f in files if f.endswith('.mp4')]

        if vtt_files:
            folder_label = rel_folder.replace('\\', ' / ').replace('/', ' / ').title()
            for vtt_file in vtt_files:
                sample_id = f"[{folder_label}] {vtt_file.replace('-transcript-1.1.vtt', '').replace('.vtt', '')}"
                vtt_path = os.path.join(root, vtt_file)
                
                # Match corresponding MP4 file if available
                mp4_path = None
                if mp4_files:
                    # Find best matching mp4 or take first in folder
                    prefix = vtt_file.split('-')[0].split('.')[0].strip()
                    matching = [m for m in mp4_files if prefix.lower() in m.lower()]
                    if matching:
                        mp4_path = os.path.join(root, matching[0])
                    else:
                        mp4_path = os.path.join(root, mp4_files[0])

                samples[sample_id] = {
                    "label": folder_label,
                    "vtt_path": vtt_path,
                    "mp4_path": mp4_path
                }

    return samples


def get_grade_badge_info(grade: str) -> Tuple[str, str, str]:
    """Returns CSS class, icon, and display string for a grade."""
    m = {
        "Excellent": ("badge-excellent", "🌟", "EXCELLENT"),
        "Good": ("badge-good", "✅", "GOOD"),
        "Average": ("badge-average", "⚖️", "AVERAGE"),
        "Weak": ("badge-weak", "⚠️", "WEAK"),
        "Poor / Quality Issue": ("badge-poor", "🚨", "POOR / QUALITY ISSUE")
    }
    return m.get(grade, ("badge-average", "📊", grade.upper()))


# --- Sidebar Setup ---
st.sidebar.title("🎓 8x8 Class Evaluator")
st.sidebar.markdown("---")

# 1. API Key & Credentials Status
st.sidebar.subheader("🔑 Credentials & API Status")
custom_key = st.sidebar.text_input("Gemini API Key Override (Optional)", type="password", help="Leave blank to use environment key or gcloud ADC.")

resolved_key, key_source = resolve_gemini_api_key(custom_key if custom_key else None)

if resolved_key:
    st.sidebar.success(f"🟢 **API Key Active** ({key_source})")
else:
    st.sidebar.warning("🟡 **API Key Missing**: Operating in Heuristic Analysis Mode.")

model_choice = st.sidebar.selectbox(
    "Target Evaluation Model",
    ["gemini-1.5-flash", "gemini-1.5-pro", "Heuristic Rules Engine"],
    index=0,
    help="gemini-1.5-flash is optimized for sub-10 second evaluation."
)

st.sidebar.markdown("---")

# 2. Input Mode Selection
st.sidebar.subheader("📂 Input Session Data")
input_mode = st.sidebar.radio(
    "Select Input Source",
    ["Upload Files (.vtt / .mp4)", "Load Sample Benchmark Session"],
    index=1
)

uploaded_vtt = None
uploaded_mp4 = None
selected_sample = None
sample_data = None

sample_dict = discover_sample_recordings(BASE_DIR)

if input_mode == "Upload Files (.vtt / .mp4)":
    uploaded_vtt = st.sidebar.file_uploader("Upload Class Transcript (.vtt)", type=["vtt"])
    uploaded_mp4 = st.sidebar.file_uploader("Upload Video Recording (.mp4)", type=["mp4"])
else:
    if sample_dict:
        sample_options = list(sample_dict.keys())
        selected_sample_key = st.sidebar.selectbox("Select Benchmark Sample", sample_options)
        sample_data = sample_dict.get(selected_sample_key)
        if sample_data:
            st.sidebar.info(
                f"**VTT:** `{os.path.basename(sample_data['vtt_path'])}`\n\n"
                f"**MP4:** `{os.path.basename(sample_data['mp4_path']) if sample_data['mp4_path'] else 'None'}`"
            )
    else:
        st.sidebar.warning("No sample files found in `class recordings/` directory.")

run_analysis = st.sidebar.button("🚀 Run Class Assessment", type="primary", use_container_width=True)


# --- Main Dashboard Header ---
st.markdown('<div class="main-header">8x8 Class Quality Assessment System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Multimodal Pedagogical & Visual Quality Evaluation Platform</div>', unsafe_allow_html=True)


# --- Pipeline Execution Logic ---
if run_analysis or "last_result" in st.session_state:
    if run_analysis:
        st.session_state.clear()  # Clear previous execution state
        
        vtt_analysis: Optional[VTTAnalysis] = None
        video_analysis: Optional[VideoAnalysis] = None
        
        with st.spinner("Processing class session inputs..."):
            # 1. Process VTT Transcript
            vtt_content = None
            vtt_path_to_use = None
            vtt_filename = "transcript.vtt"

            if input_mode == "Upload Files (.vtt / .mp4)" and uploaded_vtt is not None:
                vtt_content = uploaded_vtt.getvalue().decode("utf-8", errors="ignore")
                vtt_filename = uploaded_vtt.name
                parser = VTTParser(vtt_content, is_content=True, file_name=vtt_filename)
                vtt_analysis = parser.parse()
            elif input_mode == "Load Sample Benchmark Session" and sample_data:
                vtt_path_to_use = sample_data["vtt_path"]
                parser = VTTParser(vtt_path_to_use)
                vtt_analysis = parser.parse()
            
            if not vtt_analysis:
                st.error("❌ Please upload a valid `.vtt` transcript or select a sample recording to analyze.")
                st.stop()

            # 2. Process MP4 Video (if provided)
            video_path_to_use = None
            temp_mp4_file = None

            if input_mode == "Upload Files (.vtt / .mp4)" and uploaded_mp4 is not None:
                # Write uploaded mp4 to temp file for OpenCV processing
                temp_dir = os.path.join(BASE_DIR, "scratch")
                os.makedirs(temp_dir, exist_ok=True)
                temp_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=temp_dir)
                temp_mp4.write(uploaded_mp4.getvalue())
                temp_mp4.close()
                video_path_to_use = temp_mp4.name
                temp_mp4_file = temp_mp4.name
            elif input_mode == "Load Sample Benchmark Session" and sample_data and sample_data["mp4_path"]:
                video_path_to_use = sample_data["mp4_path"]

            if video_path_to_use and os.path.exists(video_path_to_use):
                video_analyzer = VideoAnalyzer(video_path_to_use)
                video_analysis = video_analyzer.extract_keyframes()

            # 3. Execute Class Evaluation
            use_force_mock = (model_choice == "Heuristic Rules Engine") or (not resolved_key)
            evaluator = ClassEvaluator(
                api_key=resolved_key,
                model_name=model_choice if model_choice in ["gemini-1.5-flash", "gemini-1.5-pro"] else "gemini-1.5-flash"
            )
            
            eval_result: EvaluationResult = evaluator.evaluate(
                vtt_analysis=vtt_analysis,
                video_analysis=video_analysis,
                force_mock=use_force_mock
            )

            # Store in session state for persistence
            st.session_state["vtt_analysis"] = vtt_analysis
            st.session_state["video_analysis"] = video_analysis
            st.session_state["eval_result"] = eval_result
            st.session_state["last_result"] = True

    # Retrieve from session state
    vtt_analysis: VTTAnalysis = st.session_state["vtt_analysis"]
    video_analysis: Optional[VideoAnalysis] = st.session_state.get("video_analysis")
    eval_result: EvaluationResult = st.session_state["eval_result"]

    # --- Render Dashboard Results ---

    # 1. Overall Quality Grade Badge & High-Level Metadata
    col_badge, col_meta = st.columns([1.2, 2])

    with col_badge:
        badge_class, badge_icon, badge_text = get_grade_badge_info(eval_result.overall_grade)
        st.markdown(
            f'''
            <div class="badge-container {badge_class}">
                {badge_icon} {badge_text}
            </div>
            ''',
            unsafe_allow_html=True
        )

    with col_meta:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'''<div class="metric-card">
                    <div class="metric-card-title">Duration</div>
                    <div class="metric-card-value">{vtt_analysis.formatted_duration}</div>
                </div>''',
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f'''<div class="metric-card">
                    <div class="metric-card-title">Words Spoken</div>
                    <div class="metric-card-value">{vtt_analysis.total_word_count:,}</div>
                </div>''',
                unsafe_allow_html=True
            )
        with m3:
            st.markdown(
                f'''<div class="metric-card">
                    <div class="metric-card-title">Speaker Turns</div>
                    <div class="metric-card-value">{vtt_analysis.speaker_turn_count}</div>
                </div>''',
                unsafe_allow_html=True
            )
        with m4:
            st.markdown(
                f'''<div class="metric-card">
                    <div class="metric-card-title">1-on-1 Callouts</div>
                    <div class="metric-card-value">{vtt_analysis.student_callout_count}</div>
                </div>''',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Hard Penalty Capping Rules Banner / Alert Box
    if eval_result.applied_capping_rules:
        for rule in eval_result.applied_capping_rules:
            st.warning(f"🚨 **{rule}**", icon="⚠️")
    else:
        st.success("✅ **Standard Rubric Assessment**: No hard penalty capping rules were triggered for this session.", icon="🛡️")

    st.markdown("---")

    # 3. 5-Pillar Score Metrics & Breakdown
    st.subheader("📊 5-Pillar Rubric Performance Scores")

    pillar_map = {
        "visual_environment": ("🖼️ Visual & Environment", 25, "Virtual background, camera active state, lighting"),
        "warmth_rapport": ("🤗 Warmth & Rapport", 20, "Energetic greeting, new joiner introductions, tone"),
        "student_engagement": ("🙋 Student Engagement", 25, "1-on-1 student name callouts, equal attention"),
        "pedagogical_structure": ("📚 Pedagogy & HW", 20, "Topic videos, clear homework assignment at tail"),
        "language_adaptability": ("🗣️ Language & Adaptability", 10, "Pronunciation clarity, speech speed, regional support")
    }

    cols_p = st.columns(5)
    p_keys = ["visual_environment", "warmth_rapport", "student_engagement", "pedagogical_structure", "language_adaptability"]

    for idx, p_key in enumerate(p_keys):
        p_label, p_weight, p_desc = pillar_map[p_key]
        score = eval_result.pillar_scores.get(p_key, 70.0)
        with cols_p[idx]:
            st.metric(label=f"{p_label} ({p_weight}%)", value=f"{score:.0f}/100")
            st.progress(min(max(float(score) / 100.0, 0.0), 1.0))
            st.caption(p_desc)

    # Pillar Score Chart Visual
    df_scores = pd.DataFrame({
        "Pillar": ["Visual & Env", "Rapport", "Engagement", "Pedagogy & HW", "Language"],
        "Score": [eval_result.pillar_scores.get(k, 70.0) for k in p_keys]
    })
    st.bar_chart(df_scores.set_index("Pillar"), height=220)

    st.markdown("---")

    # 4. Key Strengths & Actionable Coaching Recommendations
    col_str, col_rec = st.columns(2)

    with col_str:
        st.subheader("🌟 Key Strengths")
        if eval_result.strengths:
            for s in eval_result.strengths:
                st.markdown(f'<div class="strength-item">✔️ {s}</div>', unsafe_allow_html=True)
        else:
            st.info("No specific strengths highlighted.")

    with col_rec:
        st.subheader("🎯 Actionable Coaching Feedback")
        if eval_result.areas_for_improvement:
            for imp in eval_result.areas_for_improvement:
                st.markdown(f'<div class="improvement-item">💡 {imp}</div>', unsafe_allow_html=True)
        else:
            st.success("No critical areas for improvement flagged.")

    st.markdown("---")

    # 5. Timestamped Evidence Timeline
    st.subheader("⏱️ Timestamped Evidence Timeline")
    if eval_result.timestamped_evidence:
        for ev in eval_result.timestamped_evidence:
            ts = ev.get("timestamp", "00:00:00")
            pillar = ev.get("pillar", "General")
            obs = ev.get("observation", "")
            quote = ev.get("quote", "")

            quote_html = f'<div class="evidence-quote">"{quote}"</div>' if quote else ''
            
            st.markdown(
                f'''
                <div class="evidence-item">
                    <span class="evidence-ts">⏱️ {ts}</span>
                    <span class="evidence-pillar">{pillar}</span>
                    <div style="margin-top:0.3rem; color:#1E293B;"><b>Observation:</b> {obs}</div>
                    {quote_html}
                </div>
                ''',
                unsafe_allow_html=True
            )
    else:
        st.info("No timestamped evidence items recorded.")

    # 6. Video Keyframe Gallery (if video was processed)
    if video_analysis and video_analysis.keyframes:
        st.markdown("---")
        st.subheader("🖼️ Sampled Keyframes Gallery")
        kf_cols = st.columns(min(len(video_analysis.keyframes), 4))
        for idx, kf in enumerate(video_analysis.keyframes):
            col_target = kf_cols[idx % 4]
            with col_target:
                if kf.image_pil:
                    st.image(
                        kf.image_pil,
                        caption=f"[{kf.window.upper()}] {kf.timestamp_formatted} (Frame #{kf.frame_index})",
                        use_container_width=True
                    )

    # 7. Full Transcript Inspector Expander
    st.markdown("---")
    with st.expander("📄 View Full Clean Transcript & Analysis Details"):
        st.markdown(f"**Tutor Name:** {vtt_analysis.tutor_name or 'Not identified'}")
        st.markdown(f"**Student Names:** {', '.join(vtt_analysis.student_names) if vtt_analysis.student_names else 'None identified'}")
        st.markdown(f"**Greeting Detected:** {vtt_analysis.greeting_detected} | **Homework Detected:** {vtt_analysis.homework_detected}")
        st.text_area("Clean Spoken Transcript", value=vtt_analysis.clean_transcript, height=300)

else:
    # Initial Landing Screen
    st.info("👈 **Get Started**: Select a sample benchmark class from the sidebar or upload `.vtt` / `.mp4` files, then click **Run Class Assessment**.")
    
    st.markdown("""
    ### System Features & Capability Overview
    - **Multimodal Evaluation Engine**: Combines WebVTT text analytics with OpenCV keyframe visual extraction.
    - **5 Core Pedagogical Pillars**: Visual & Environment (25%), Warmth & Rapport (20%), Student Engagement (25%), Pedagogy & HW (20%), Language & Adaptability (10%).
    - **Hard Penalty Capping Rules**: Caps overall grade at `Weak` / `Poor` for missing background/camera-off, and `Average` for missing homework or abrupt openings.
    - **Evidence Citing**: Provides timestamped observations with direct spoken quotes.
    """)
