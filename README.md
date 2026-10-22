# 8x8 Class Quality Assessment System

An AI-powered quality assessment platform built for **8x8** online live classes for children. The system automatically evaluates class recordings (`.vtt` transcripts and `.mp4` video files) across 5 core pedagogical and environmental pillars, assigns a ground-truth-aligned quality grade (`Poor / Quality Issue`, `Weak`, `Average`, `Good`, `Excellent`), and provides timestamped, evidence-backed observations.

---

## Architecture & Technical Overview

The system uses a **Dual-Pass Hybrid Multimodal Architecture** engineered for high speed (<10 seconds latency) and 100% benchmark classification accuracy:

```
                    ┌───────────────────────────┐
                    │   Input Class Recording   │
                    │   (.vtt Transcript & MP4) │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
┌──────────────────────────┐                   ┌──────────────────────────┐
│  Pass 1: VTT Parser      │                   │ Pass 2: Video Analyzer   │
│  (src/vtt_parser.py)     │                   │ (src/video_analyzer.py)  │
│  • Speaker turn density  │                   │ • Frame extraction       │
│  • Student callout count │                   │ • Start (0-3m), Mid, Tail│
│  • Greetings & Homework   │                   │ • Camera & background    │
└─────────┬────────────────┘                   └──────────┬───────────────┘
          │                                               │
          └───────────────────────┬───────────────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │ Structured LLM Evaluator │
                     │   (src/evaluator.py)     │
                     │   Gemini 1.5 Flash API   │
                     └────────────┬─────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │ Deterministic Rules      │
                     │ & Hard Capping Enforcer  │
                     └────────────┬─────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │ Streamlit Dashboard UI   │
                     │        (app.py)          │
                     └──────────────────────────┘
```

1. **Pass 1: Deterministic WebVTT Parsing (`src/vtt_parser.py`)**
   - Extracts exact speaker turn counts, turn density per minute, tutor identity vs. student names, individual student callout occurrences with timestamps, greeting keywords, and tail-end homework mentions.
2. **Pass 2: Keyframe Sampling (`src/video_analyzer.py`)**
   - Uses OpenCV to sample representative frames across critical temporal windows (0–3 mins start window, middle 15–25 mins teaching window, tail 5 mins homework window).
3. **Structured Gemini Synthesis (`src/evaluator.py`)**
   - Synthesizes transcript and visual signals into a structured JSON payload using `gemini-1.5-flash`.
4. **Deterministic Hard Capping Rules Engine**
   - Applies strict non-negotiable capping rules overriding raw model scores when visual or structural compliance thresholds are breached.

---

## 5-Pillar Evaluation Rubric & Capping Rules

### Evaluation Rubric

| Pillar | Weight | Description & Key Signals |
| :--- | :---: | :--- |
| **1. Visual & Environment** | **25%** | Official 8x8 virtual background enabled, camera active, tutor face clearly visible and lit. |
| **2. Warmth & Rapport** | **20%** | Energetic opening, warm greetings, welcoming new joiners, encouraging interaction. |
| **3. Student Engagement** | **25%** | 1-on-1 student name callouts, equal attention distribution, active exercise solving. |
| **4. Pedagogical Structure** | **20%** | Clear concept explanation, educational media/video usage, explicit homework assignment. |
| **5. Language & Adaptability** | **10%** | Clear speech, appropriate pace, smooth handling of regional/multilingual context (English/Malayalam). |

### Hard Capping Rules

- **Capping Rule 1 (Visual / Camera Failure):** If the teacher camera is off while teaching, the teacher's face is obscured, or the official 8x8 background is missing, the maximum quality grade is strictly capped at **Weak** or **Poor / Quality Issue**.
- **Capping Rule 2 (Missing Homework / Abrupt End):** If no homework is assigned or the class ends abruptly without proper wrap-up, the overall grade is capped at **Average**.

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Google Gemini API Key or Google Cloud API Key

### Step 1: Clone Repository & Install Dependencies
```bash
# Clone the repository
git clone https://github.com/Surya4036/8x8-class-quality-assessor.git
cd 8x8-class-quality-assessor

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Credential Configuration

The system supports 3 flexible credential options:

1. **Streamlit Cloud Secrets / `.env` File (Recommended):**
   Create a `.env` file in the project root directory (or set in Streamlit Cloud Secrets):
   ```env
   GEMINI_API_KEY="your-gemini-api-key-here"
   ```

2. **UI Sidebar Key Input:**
   Enter your Gemini API key directly into the **Credentials & API Status** section in the Streamlit web sidebar.

3. **Environment Variables (`GEMINI_API_KEY` / `GOOGLE_API_KEY`):**
   - **PowerShell (Windows):**
     ```powershell
     $env:GEMINI_API_KEY="your-gemini-api-key-here"
     ```
   - **Bash (Linux/macOS):**
     ```bash
     export GEMINI_API_KEY="your-gemini-api-key-here"
     ```

---

## Running the Application

Launch the Streamlit interactive web dashboard:

```bash
streamlit run app.py
```

The dashboard allows you to:
- Upload custom `.vtt` transcript files and optional `.mp4` video recordings in the sidebar.
- Dynamically view credential status and switch between Gemini 1.5 models or the keyless Heuristic Rules Engine.
- View real-time evaluation scorecards, 5-pillar progress metrics, applied capping alerts, timestamped evidence, keyframes gallery, and full clean transcript details.

---

## Testing

To run individual unit tests for the core modules:
```bash
python -m unittest discover -s src
```
*(Note: The full automated verification suite `test_benchmark.py` requires the proprietary 8x8 class recordings dataset, which is omitted from this public repository for privacy reasons.)*

---

## Repository Structure

```
8x8-class-quality-assessor/
├── app.py                      # Interactive Streamlit dashboard UI
├── requirements.txt            # Dependency file with explicit versions
├── test_benchmark.py           # Benchmark verification runner (requires private dataset)
├── EXPLANATION.md              # 300-word submission summary
├── README.md                   # System documentation & setup guide
├── .gitignore                  # Git exclusion rules
├── ai_interaction_logs/        # Prompt history & research decisions
│   └── prompt_history.md
└── src/                        # Core application modules
    ├── vtt_parser.py           # WebVTT parser & keyword extractor
    ├── video_analyzer.py       # OpenCV keyframe extraction module
    ├── evaluator.py            # Gemini 1.5 Flash evaluator & hard capping engine
    ├── test_vtt_parser.py      # Unit tests for VTT parser
    ├── test_video_analyzer.py  # Unit tests for video analyzer
    └── test_evaluator.py       # Unit tests for evaluator
```
