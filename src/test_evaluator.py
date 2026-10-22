"""
Unit and Integration Tests for Gemini Evaluator and Hard Capping Rules Engine (src/evaluator.py).
"""

import os
import sys
import unittest
from typing import Dict, Any

from src.vtt_parser import VTTParser, VTTAnalysis
from src.video_analyzer import VideoAnalyzer, VideoAnalysis
from src.evaluator import (
    ClassEvaluator,
    EvaluationResult,
    resolve_gemini_api_key,
    GRADE_RANKS
)


class TestGeminiEvaluator(unittest.TestCase):

    def test_api_key_resolution_order(self):
        """Verify API key resolution order (Explicit -> GEMINI_API_KEY -> GOOGLE_API_KEY)."""
        # Save env
        orig_gemini = os.environ.get("GEMINI_API_KEY")
        orig_google = os.environ.get("GOOGLE_API_KEY")

        try:
            # 1. Explicit arg
            key, src = resolve_gemini_api_key("explicit_key_123")
            self.assertEqual(key, "explicit_key_123")
            self.assertEqual(src, "explicit_arg")

            # 2. GEMINI_API_KEY
            os.environ["GEMINI_API_KEY"] = "gemini_env_key"
            os.environ["GOOGLE_API_KEY"] = "google_env_key"
            key, src = resolve_gemini_api_key()
            self.assertEqual(key, "gemini_env_key")
            self.assertEqual(src, "GEMINI_API_KEY")

            # 3. GOOGLE_API_KEY fallback when GEMINI_API_KEY missing
            del os.environ["GEMINI_API_KEY"]
            key, src = resolve_gemini_api_key()
            self.assertEqual(key, "google_env_key")
            self.assertEqual(src, "GOOGLE_API_KEY")

        finally:
            # Restore env
            if orig_gemini:
                os.environ["GEMINI_API_KEY"] = orig_gemini
            elif "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]

            if orig_google:
                os.environ["GOOGLE_API_KEY"] = orig_google
            elif "GOOGLE_API_KEY" in os.environ:
                del os.environ["GOOGLE_API_KEY"]

    def test_capping_rule_1_camera_and_background(self):
        """Rule 1: Camera off / obscured or missing background caps overall grade at Weak or Poor / Quality Issue."""
        evaluator = ClassEvaluator()

        vtt_sample = VTTAnalysis(
            file_path="dummy.vtt",
            file_name="dummy.vtt",
            total_duration_sec=1800.0,
            formatted_duration="30:00",
            total_word_count=1200,
            speaker_turn_count=40,
            speaker_turn_density_per_min=1.3,
            speakers=["Tutor", "Student"],
            tutor_name="Test Tutor",
            student_names=["Alice"],
            student_callout_count=5,
            student_callouts=[],
            greeting_detected=True,
            greeting_count=2,
            greeting_details=[],
            homework_detected=True,
            homework_detected_tail=True,
            homework_count=1,
            homework_tail_count=1,
            homework_details=[],
            clean_transcript="Hello Alice, welcome to class! Here is your homework assignment."
        )

        # Case A: LLM returns Excellent but camera was off / missing background
        capping_flags = {
            "camera_off_or_obscured": True,
            "missing_8x8_background": True,
            "missing_homework": False,
            "abrupt_start_or_end": False
        }
        final_grade, applied_rules = evaluator._enforce_capping_rules(
            raw_grade="Excellent",
            capping_flags=capping_flags,
            vtt_analysis=vtt_sample
        )
        self.assertIn("HARD CAPPING RULE 1 APPLIED", applied_rules[0])
        self.assertIn(final_grade, ["Weak", "Poor / Quality Issue"])

    def test_capping_rule_2_missing_homework(self):
        """Rule 2: Missing homework or abrupt flow caps overall grade at Average."""
        evaluator = ClassEvaluator()

        vtt_no_hw = VTTAnalysis(
            file_path="no_hw.vtt",
            file_name="no_hw.vtt",
            total_duration_sec=1800.0,
            formatted_duration="30:00",
            total_word_count=1200,
            speaker_turn_count=40,
            speaker_turn_density_per_min=1.3,
            speakers=["Tutor"],
            tutor_name="Test Tutor",
            student_names=["Bob"],
            student_callout_count=3,
            student_callouts=[],
            greeting_detected=True,
            greeting_count=1,
            greeting_details=[],
            homework_detected=False,
            homework_detected_tail=False,
            homework_count=0,
            homework_tail_count=0,
            homework_details=[],
            clean_transcript="Hello Bob! Great class today. Bye!"
        )

        # Case B: High quality engagement but missing homework -> Capped at Average
        capping_flags = {
            "camera_off_or_obscured": False,
            "missing_8x8_background": False,
            "missing_homework": True,
            "abrupt_start_or_end": False
        }
        final_grade, applied_rules = evaluator._enforce_capping_rules(
            raw_grade="Good",
            capping_flags=capping_flags,
            vtt_analysis=vtt_no_hw
        )
        self.assertIn("HARD CAPPING RULE 2 APPLIED", applied_rules[0])
        self.assertEqual(final_grade, "Average")

    def test_build_prompt_structure(self):
        """Verify prompt builder formats VTT metrics and keyframe metadata."""
        evaluator = ClassEvaluator()
        parser = VTTParser("class recordings/excellent/B491 Beginner -transcript-1.1.vtt")
        vtt_analysis = parser.parse()

        prompt = evaluator.build_prompt(vtt_analysis)
        self.assertIn("CLASS TRANSCRIPT METRICS", prompt)
        self.assertIn("AYSWARYA JOSEPH", prompt.upper())
        self.assertIn("FULL TRANSCRIPT EXCERPT", prompt)

    def test_heuristic_evaluator_and_serialization(self):
        """Verify fallback heuristic evaluation and JSON dictionary conversion."""
        evaluator = ClassEvaluator()
        parser = VTTParser("class recordings/excellent/B491 Beginner -transcript-1.1.vtt")
        vtt_analysis = parser.parse()

        res = evaluator.evaluate(vtt_analysis, force_mock=True)
        self.assertIsInstance(res, EvaluationResult)
        self.assertIn(res.overall_grade, ["Good", "Excellent", "Average"])
        self.assertIn("visual_environment", res.pillar_scores)
        self.assertIn("warmth_rapport", res.pillar_scores)
        self.assertIn("student_engagement", res.pillar_scores)
        self.assertIn("pedagogical_structure", res.pillar_scores)
        self.assertIn("language_adaptability", res.pillar_scores)

        # Test to_dict serialization
        d = res.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["overall_grade"], res.overall_grade)
        self.assertEqual(d["pillar_scores"], res.pillar_scores)

    def test_ground_truth_samples_evaluation(self):
        """Run heuristic evaluation against sample VTT files from all rating tiers."""
        evaluator = ClassEvaluator()

        samples = [
            ("class recordings/excellent/B491 Beginner -transcript-1.1.vtt", ["Good", "Excellent"]),
            ("class recordings/weak/B482 Intermediate -transcript-1.1.vtt", ["Weak", "Average", "Poor / Quality Issue"]),
            ("class recordings/quality issue/B392 Beginner-transcript-1.1.vtt", ["Weak", "Average", "Poor / Quality Issue"])
        ]

        for path, expected_tier in samples:
            if os.path.exists(path):
                parser = VTTParser(path)
                vtt_analysis = parser.parse()
                res = evaluator.evaluate(vtt_analysis, force_mock=True)
                self.assertIsNotNone(res.overall_grade)
                self.assertTrue(len(res.timestamped_evidence) >= 0)


if __name__ == "__main__":
    unittest.main()
