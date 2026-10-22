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

# Check if a system/server key is pre-configured
system_key, system_source = resolve_gemini_api_key(None)

if system_key:
    st.sidebar.success(f"🟢 **System API Key Active** ({system_source})")
    custom_key = st.sidebar.text_input(
        "Personal API Key Override (Optional)",
        type="password",
        help="System key is active. Leave blank to use pre-configured server secret, or enter a custom key to override."
    )
else:
    st.sidebar.warning("🟡 **No System API Key Configured**")
    custom_key = st.sidebar.text_input(
        "Gemini API Key (Required for AI Evaluation)",
        type="password",
        help="Provide a Gemini API key to enable Multimodal AI evaluation, or switch model to 'Heuristic Rules Engine'."
    )

resolved_key, key_source = resolve_gemini_api_key(custom_key if custom_key else None)

model_choice = st.sidebar.selectbox(
    "Target Evaluation Model",
    ["gemini-1.5-flash", "gemini-1.5-pro", "Heuristic Rules Engine"],
    index=0,
    help="gemini-1.5-flash is optimized for sub-10 second evaluation."
)

st.sidebar.markdown("---")

# 2. Input Session Data
st.sidebar.subheader("📂 Upload Session Data")
uploaded_vtt = st.sidebar.file_uploader("Upload Class Transcript (.vtt)", type=["vtt"], help="Select WebVTT transcript file.")
uploaded_mp4 = st.sidebar.file_uploader("Upload Video Recording (.mp4)", type=["mp4"], help="Optional MP4 video recording for keyframe vision analysis.")

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
            if uploaded_vtt is not None:
                vtt_content = uploaded_vtt.getvalue().decode("utf-8", errors="ignore")
                vtt_filename = uploaded_vtt.name
                parser = VTTParser(vtt_content, is_content=True, file_name=vtt_filename)
                vtt_analysis = parser.parse()
            
            if not vtt_analysis:
                st.error("❌ Please upload a valid `.vtt` class transcript file in the sidebar to analyze.")
                st.stop()

            # 2. Process MP4 Video (if provided)
            video_path_to_use = None
            if uploaded_mp4 is not None:
                temp_dir = os.path.join(BASE_DIR, "scratch")
                os.makedirs(temp_dir, exist_ok=True)
                temp_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=temp_dir)
                temp_mp4.write(uploaded_mp4.getvalue())
                temp_mp4.close()
                video_path_to_use = temp_mp4.name

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
    st.info("👈 **Get Started**: Upload a `.vtt` class transcript and optional `.mp4` video recording in the sidebar, then click **Run Class Assessment**.")
    
    st.markdown("""
    ### System Features & Capability Overview
    - **Multimodal Evaluation Engine**: Combines WebVTT text analytics with OpenCV keyframe visual extraction.
    - **5 Core Pedagogical Pillars**: Visual & Environment (25%), Warmth & Rapport (20%), Student Engagement (25%), Pedagogy & HW (20%), Language & Adaptability (10%).
    - **Hard Penalty Capping Rules**: Caps overall grade at `Weak` / `Poor` for missing background/camera-off, and `Average` for missing homework or abrupt openings.
    - **Evidence Citing**: Provides timestamped observations with direct spoken quotes.
    """)
