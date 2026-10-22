"""
Test suite and metric verification runner for VTTParser across sample class recordings.
"""

import os
import glob
import sys
from typing import List

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vtt_parser import VTTParser, VTTAnalysis


def run_tests() -> bool:
    sys.stdout.reconfigure(encoding='utf-8')
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "class recordings"))
    vtt_files = sorted(glob.glob(os.path.join(base_dir, "**", "*.vtt"), recursive=True))

    if not vtt_files:
        print(f"❌ ERROR: No .vtt files found in {base_dir}")
        return False

    print(f"🔍 Found {len(vtt_files)} WebVTT files in class recordings.\n")
    print("=" * 110)
    print(f"{'File Name':<35} | {'Duration':<9} | {'Words':<6} | {'Turns':<6} | {'Tutor':<20} | {'Callouts':<8} | {'HW Tail':<7}")
    print("=" * 110)

    passed = 0
    failed = 0

    for file_path in vtt_files:
        rel_path = os.path.relpath(file_path, base_dir)
        file_name = os.path.basename(file_path)

        try:
            parser = VTTParser(file_path)
            analysis: VTTAnalysis = parser.parse()

            # Assertions / Validations
            assert len(analysis.cues) > 0, f"Cues list is empty for {file_name}"
            assert analysis.total_duration_sec > 0, f"Duration is 0 for {file_name}"
            assert analysis.total_word_count > 0, f"Word count is 0 for {file_name}"
            assert analysis.speaker_turn_count > 0, f"Turn count is 0 for {file_name}"
            assert len(analysis.clean_transcript) > 0, f"Clean transcript is empty for {file_name}"
            assert analysis.tutor_name is not None, f"Tutor name not identified for {file_name}"

            tutor_display = analysis.tutor_name[:18] if len(analysis.tutor_name) > 18 else analysis.tutor_name
            hw_tail_str = "YES" if analysis.homework_detected_tail else "NO"

            print(
                f"{file_name[:35]:<35} | "
                f"{analysis.formatted_duration:<9} | "
                f"{analysis.total_word_count:<6} | "
                f"{analysis.speaker_turn_count:<6} | "
                f"{tutor_display:<20} | "
                f"{analysis.student_callout_count:<8} | "
                f"{hw_tail_str:<7}"
            )
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {rel_path} - Error: {e}")
            failed += 1

    print("=" * 110)
    print(f"\n📊 Summary: {passed} PASSED, {failed} FAILED out of {len(vtt_files)} files.\n")

    if failed == 0:
        print("✅ All VTT parser tests passed successfully!")
        return True
    else:
        print("❌ Some tests failed.")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
