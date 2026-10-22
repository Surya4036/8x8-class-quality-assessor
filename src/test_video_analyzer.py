"""
Unit and Integration Tests for Video Keyframe Sampler (VideoAnalyzer).
Tests keyframe extraction against benchmark .mp4 recordings and missing/corrupted file fallbacks.
"""

import os
import glob
import unittest
from PIL import Image
from video_analyzer import VideoAnalyzer, VideoAnalysis, Keyframe


class TestVideoAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.recordings_dir = r"D:\Humaniti AI Assessment\class recordings"
        # Find all MP4 files recursively
        cls.mp4_files = glob.glob(os.path.join(cls.recordings_dir, "**", "*.mp4"), recursive=True)

    def test_mp4_files_discovered(self):
        """Ensure benchmark mp4 video files are present."""
        self.assertGreater(len(self.mp4_files), 0, "No .mp4 files found in class recordings directory.")

    def test_missing_file_fallback(self):
        """Test fallback when file path does not exist."""
        analyzer = VideoAnalyzer(r"D:\Humaniti AI Assessment\class recordings\non_existent_video.mp4")
        res = analyzer.extract_keyframes()
        self.assertEqual(res.status, "fallback_missing_file")
        self.assertIn("not found", res.error_message)
        self.assertEqual(len(res.keyframes), 0)

    def test_empty_path_fallback(self):
        """Test fallback when video_path is None or empty."""
        analyzer = VideoAnalyzer("")
        res = analyzer.extract_keyframes()
        self.assertEqual(res.status, "fallback_missing_file")
        self.assertEqual(len(res.keyframes), 0)

    def test_keyframe_extraction_on_all_recordings(self):
        """Test keyframe sampling on all benchmark .mp4 recordings."""
        for video_path in self.mp4_files:
            rel_path = os.path.relpath(video_path, self.recordings_dir)
            with self.subTest(video=rel_path):
                analyzer = VideoAnalyzer(video_path)
                result = analyzer.extract_keyframes()

                self.assertEqual(
                    result.status, "success",
                    f"Extraction failed for {rel_path}: {result.error_message}"
                )
                self.assertIsNotNone(result.metadata)
                self.assertGreater(result.metadata.duration_sec, 0)
                self.assertGreater(result.metadata.fps, 0)
                self.assertGreater(result.metadata.total_frames, 0)
                self.assertGreater(result.metadata.width, 0)
                self.assertGreater(result.metadata.height, 0)

                # Check extracted keyframes
                self.assertGreater(result.total_keyframes, 0)
                windows_found = set(kf.window for kf in result.keyframes)
                self.assertIn("start", windows_found)

                # Verify each keyframe properties
                for kf in result.keyframes:
                    self.assertIsInstance(kf.image_pil, Image.Image)
                    self.assertGreater(len(kf.image_bytes), 0)
                    self.assertIsNotNone(kf.base64_jpeg)
                    self.assertGreater(len(kf.base64_jpeg), 0)
                    self.assertGreaterEqual(kf.timestamp_sec, 0)

                # Convert to dict and verify serializability
                d = result.to_dict(include_keyframes_base64=False)
                self.assertEqual(d["status"], "success")
                self.assertEqual(len(d["keyframes"]), result.total_keyframes)

    def test_window_sampling_counts_long_video(self):
        """Test that a long video produces keyframes in Start (3), Middle (2), and Tail (2) windows."""
        # Find a video with duration >= 25 minutes
        long_video = None
        for path in self.mp4_files:
            analyzer = VideoAnalyzer(path)
            res = analyzer.extract_keyframes()
            if res.status == "success" and res.metadata.duration_sec >= 1500:
                long_video = (path, res)
                break

        if long_video:
            path, res = long_video
            start_kfs = [kf for kf in res.keyframes if kf.window == "start"]
            mid_kfs = [kf for kf in res.keyframes if kf.window == "middle"]
            tail_kfs = [kf for kf in res.keyframes if kf.window == "tail"]

            self.assertEqual(len(start_kfs), 3, f"Expected 3 start keyframes for {path}")
            self.assertEqual(len(mid_kfs), 2, f"Expected 2 middle keyframes for {path}")
            self.assertEqual(len(tail_kfs), 2, f"Expected 2 tail keyframes for {path}")
        else:
            self.skipTest("No video >= 25 mins found.")


if __name__ == "__main__":
    unittest.main()
