"""
Automated Ground-Truth Benchmark Test Suite for 8x8 Class Quality Assessment System.
Discovers all benchmark recordings across ground-truth folders ('excellent', 'good', 'average', 'weak', 'quality issue'),
pairs .vtt transcript and .mp4 video files, executes VTTParser, VideoAnalyzer, and ClassEvaluator,
compares predicted grades against human QA ground truth, and asserts >= 90% classification accuracy.
"""

import os
import sys
import unittest
from typing import Dict, List, Tuple, Any

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.vtt_parser import VTTParser
from src.video_analyzer import VideoAnalyzer
from src.evaluator import ClassEvaluator

FOLDER_TO_GROUND_TRUTH = {
    "excellent": "Excellent",
    "good": "Good",
    "average": "Average",
    "weak": "Weak",
    "quality issue": "Poor / Quality Issue"
}


def discover_and_pair_recordings(base_dir: str) -> List[Dict[str, Any]]:
    """
    Discovers all benchmark recordings in base_dir and pairs matching .vtt and .mp4 files.
    """
    recordings = []
    if not os.path.exists(base_dir):
        return recordings

    for folder_name, expected_grade in FOLDER_TO_GROUND_TRUTH.items():
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            continue

        files = os.listdir(folder_path)
        vtt_files = [f for f in files if f.endswith(".vtt")]
        mp4_files = [f for f in files if f.endswith(".mp4")]

        for vtt_file in vtt_files:
            vtt_path = os.path.join(folder_path, vtt_file)
            
            # Extract batch identifier (e.g., 'B472', 'B491', 'B435', 'B552', 'B516', 'B482', 'B558', 'B392', 'B441')
            prefix = vtt_file.split("-transcript")[0].split()[0]
            
            # Match MP4 file sharing prefix
            matching_mp4 = [m for m in mp4_files if prefix.lower() in m.lower()]
            mp4_path = os.path.join(folder_path, matching_mp4[0]) if matching_mp4 else None

            recordings.append({
                "batch_code": prefix,
                "folder": folder_name,
                "expected_grade": expected_grade,
                "vtt_file": vtt_file,
                "vtt_path": vtt_path,
                "mp4_file": matching_mp4[0] if matching_mp4 else "None",
                "mp4_path": mp4_path
            })

    return recordings


def run_benchmark_suite(recordings_dir: str = None) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Runs full evaluation over all discovered benchmark recordings and prints detailed report.
    """
    if recordings_dir is None:
        recordings_dir = os.path.join(PROJECT_ROOT, "class recordings")

    recordings = discover_and_pair_recordings(recordings_dir)
    evaluator = ClassEvaluator()

    results = []
    matches = 0

    print("=" * 105)
    print(" 8x8 CLASS QUALITY ASSESSMENT SYSTEM - GROUND-TRUTH BENCHMARK TEST SUITE")
    print("=" * 105)
    print(f"{'BATCH':<8} | {'FOLDER':<15} | {'EXPECTED (QA)':<20} | {'PREDICTED':<20} | {'MATCH':<6} | {'STATUS':<10}")
    print("-" * 105)

    for rec in recordings:
        # Parse transcript
        vtt_parser = VTTParser(rec["vtt_path"])
        vtt_res = vtt_parser.parse()

        # Extract video keyframes if video present
        video_res = None
        if rec["mp4_path"] and os.path.exists(rec["mp4_path"]):
            video_analyzer = VideoAnalyzer(rec["mp4_path"])
            video_res = video_analyzer.extract_keyframes()

        # Run Class Evaluator (uses live Gemini API if .env key present, otherwise falls back)
        eval_res = evaluator.evaluate(vtt_res, video_res, force_mock=False)

        is_match = (eval_res.overall_grade == rec["expected_grade"])
        if is_match:
            matches += 1

        rec_result = {
            "batch_code": rec["batch_code"],
            "folder": rec["folder"],
            "expected_grade": rec["expected_grade"],
            "predicted_grade": eval_res.overall_grade,
            "match": is_match,
            "vtt_file": rec["vtt_file"],
            "mp4_file": rec["mp4_file"],
            "pillar_scores": eval_res.pillar_scores,
            "applied_capping": eval_res.applied_capping_rules,
            "model_used": eval_res.model_used
        }
        results.append(rec_result)

        match_str = "PASS" if is_match else "FAIL"
        print(f"{rec['batch_code']:<8} | {rec['folder']:<15} | {rec['expected_grade']:<20} | {eval_res.overall_grade:<20} | {str(is_match):<6} | {match_str:<10}")

    total = len(recordings)
    accuracy = (matches / total * 100.0) if total > 0 else 0.0

    print("-" * 105)
    print(f"BENCHMARK ACCURACY SUMMARY: {matches}/{total} Matches ({accuracy:.1f}%)")
    print("=" * 105)

    return accuracy, results


class TestBenchmarkSuite(unittest.TestCase):
    """
    Pytest / Unittest Test Class for Benchmark Integration Verification.
    """

    def test_ground_truth_classification_accuracy(self):
        recordings_dir = os.path.join(PROJECT_ROOT, "class recordings")
        accuracy, results = run_benchmark_suite(recordings_dir)
        
        self.assertGreaterEqual(len(results), 5, "Benchmark suite must discover at least 5 benchmark recordings.")
        self.assertGreaterEqual(
            accuracy, 90.0,
            f"Benchmark classification accuracy {accuracy:.1f}% is below required 90.0% benchmark threshold!"
        )


if __name__ == "__main__":
    accuracy, results = run_benchmark_suite()
    if accuracy < 90.0:
        sys.exit(1)
    else:
        sys.exit(0)
