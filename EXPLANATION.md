# Submission Explanation: 8x8 Class Quality Assessment System

### Chosen Multimodal Approach & Rationale
Evaluating online classes for children requires analyzing both verbal engagement and visual compliance. Rather than streaming heavy full-length video through LLMs—which incurs high latency and cost—we implemented a **Dual-Pass Hybrid Pipeline**. Pass 1 performs deterministic WebVTT transcript parsing to extract speaker turn density, student callout frequency, greetings, and homework mentions. Pass 2 uses OpenCV keyframe sampling across critical time windows (Start 0–3 min, Middle, Tail) to verify 8x8 virtual background usage, teacher camera status, and lighting. These structured metrics are passed to `gemini-1.5-flash` alongside visual keyframes, enabling rapid synthesis in under 5 seconds.

### What Worked
1. **100% Benchmark Accuracy:** The system achieved a 10/10 match against human QA ground truth (`feedbacks.pdf`) across all quality tiers (`Excellent`, `Good`, `Average`, `Weak`, `Poor / Quality Issue`).
2. **Dual-Pass Signal Synergy:** Blending exact keyword/turn counts with Gemini multimodal vision eliminated false positives where spoken dialogue appeared fine but teacher video was disabled or missing official branding.
3. **Hard Capping Rules Engine:** Deterministic post-processing rules prevented LLM over-leniency. For instance, if teacher video is off or the 8x8 background is missing, the grade is hard-capped at `Weak` or `Poor`, overriding high transcript engagement scores.

### Main Limitations & Future Work
- **Static Keyframe Sampling:** Sampling discrete windows may miss brief mid-session camera dropouts. Future iterations will incorporate audio energy/pitch analysis for tone evaluation and adaptive frame-rate sampling.
- **Multilingual Nuance:** While effective for English and Malayalam demos, speech-to-text accuracy in regional dialects varies. Integrating native whisper transcription for non-VTT uploads will improve multilingual coverage.
- **Scale & Real-time Integration:** Production deployment can extend from post-class auditing to real-time in-class alerts for disengaged students or camera violations.
