"""
Video Keyframe Sampler Module for 8x8 Class Quality Assessment System.
Extracts representative keyframes from MP4 recordings at key time windows
(Start, Middle, Tail) using OpenCV and Pillow, without loading full video into RAM.
Provides graceful fallback for missing or corrupted video files.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import os
import io
import base64
import cv2
from PIL import Image


@dataclass
class Keyframe:
    window: str                      # "start", "middle", "tail"
    timestamp_sec: float
    timestamp_formatted: str        # e.g., "00:01:30"
    frame_index: int
    image_pil: Optional[Image.Image] = field(default=None, repr=False)
    image_bytes: Optional[bytes] = field(default=None, repr=False)
    base64_jpeg: Optional[str] = field(default=None, repr=False)

    def to_dict(self, include_base64: bool = True) -> Dict[str, Any]:
        d = {
            "window": self.window,
            "timestamp_sec": round(self.timestamp_sec, 2),
            "timestamp_formatted": self.timestamp_formatted,
            "frame_index": self.frame_index,
        }
        if include_base64 and self.base64_jpeg:
            d["base64_jpeg"] = self.base64_jpeg
        return d


@dataclass
class VideoMetadata:
    file_path: str
    file_name: str
    duration_sec: float
    formatted_duration: str
    fps: float
    total_frames: int
    width: int
    height: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoAnalysis:
    file_path: str
    file_name: str
    status: str                       # "success", "fallback_missing_file", "fallback_corrupt_file"
    error_message: Optional[str] = None
    metadata: Optional[VideoMetadata] = None
    keyframes: List[Keyframe] = field(default_factory=list)
    total_keyframes: int = 0

    def to_dict(self, include_keyframes_base64: bool = False) -> Dict[str, Any]:
        res = {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "status": self.status,
            "error_message": self.error_message,
            "total_keyframes": self.total_keyframes,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "keyframes": [
                kf.to_dict(include_base64=include_keyframes_base64)
                for kf in self.keyframes
            ]
        }
        return res


class VideoAnalyzer:
    """
    Efficient OpenCV/Pillow video keyframe extractor.
    Samples keyframes from Start (0-3 min), Middle (15-25 min), and Tail (40+ min)
    using direct timestamp seeking (cv2.CAP_PROP_POS_MSEC).
    """

    def __init__(self, video_path: Optional[str] = None):
        self.video_path = video_path or ""
        self.file_name = os.path.basename(self.video_path) if self.video_path else ""

    @staticmethod
    def format_seconds(seconds: float) -> str:
        seconds = max(0, int(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def calculate_sample_timestamps(self, duration_sec: float) -> List[Dict[str, Any]]:
        """
        Determines target sample timestamps based on evaluation windows:
        a) Start Window (0-3 mins): 3 keyframes (30s, 90s, 150s)
        b) Middle Window (15-25 mins): 2 keyframes (e.g., 18m, 22m)
        c) Tail Window (40+ mins or final 10%): 2 keyframes
        Adapts dynamically if video duration is shorter than standard class length.
        """
        samples = []

        # --- A) Start Window (3 keyframes) ---
        if duration_sec >= 180.0:
            start_ts = [30.0, 90.0, 150.0]
        else:
            start_ts = [
                duration_sec * 0.15,
                duration_sec * 0.50,
                duration_sec * 0.85
            ]

        for ts in start_ts:
            if ts < duration_sec:
                samples.append({"window": "start", "timestamp_sec": ts})

        # --- B) Middle Window (2 keyframes) ---
        if duration_sec >= 1500.0:  # >= 25 mins
            mid_ts = [18.0 * 60.0, 22.0 * 60.0]
        elif duration_sec >= 600.0:  # 10 - 25 mins
            mid_ts = [duration_sec * 0.40, duration_sec * 0.60]
        elif duration_sec > 180.0:  # 3 - 10 mins
            mid_ts = [duration_sec * 0.45, duration_sec * 0.65]
        else:
            mid_ts = []

        for ts in mid_ts:
            if ts < duration_sec and ts not in [s["timestamp_sec"] for s in samples]:
                samples.append({"window": "middle", "timestamp_sec": ts})

        # --- C) Tail Window (2 keyframes) ---
        if duration_sec >= 2400.0:  # >= 40 mins
            tail_ts = [41.0 * 60.0, 43.0 * 60.0]
        elif duration_sec > 180.0:  # < 40 mins
            tail_ts = [duration_sec * 0.88, duration_sec * 0.95]
        else:
            tail_ts = []

        for ts in tail_ts:
            if ts < duration_sec and ts not in [s["timestamp_sec"] for s in samples]:
                samples.append({"window": "tail", "timestamp_sec": ts})

        return samples

    def extract_keyframes(
        self,
        custom_timestamps: Optional[List[Dict[str, Any]]] = None,
        jpeg_quality: int = 85
    ) -> VideoAnalysis:
        """
        Extracts keyframes from the MP4 video using cv2 direct position seeking.
        Returns a VideoAnalysis object with metadata and PIL/JPEG image representations.
        """
        # 1. Check file existence
        if not self.video_path or not os.path.exists(self.video_path):
            return VideoAnalysis(
                file_path=self.video_path,
                file_name=self.file_name,
                status="fallback_missing_file",
                error_message=f"Video file not found: '{self.video_path}'" if self.video_path else "No video path provided."
            )

        # 2. Open Video Capture
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return VideoAnalysis(
                file_path=self.video_path,
                file_name=self.file_name,
                status="fallback_corrupt_file",
                error_message=f"OpenCV could not open video file: '{self.video_path}'"
            )

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if fps <= 0 or total_frames <= 0:
                return VideoAnalysis(
                    file_path=self.video_path,
                    file_name=self.file_name,
                    status="fallback_corrupt_file",
                    error_message=f"Invalid video metadata (FPS: {fps}, Total Frames: {total_frames})."
                )

            duration_sec = total_frames / fps
            formatted_duration = self.format_seconds(duration_sec)

            metadata = VideoMetadata(
                file_path=self.video_path,
                file_name=self.file_name,
                duration_sec=duration_sec,
                formatted_duration=formatted_duration,
                fps=fps,
                total_frames=total_frames,
                width=width,
                height=height
            )

            # 3. Determine timestamps to sample
            if custom_timestamps:
                targets = custom_timestamps
            else:
                targets = self.calculate_sample_timestamps(duration_sec)

            extracted_keyframes: List[Keyframe] = []

            # 4. Direct Timestamp Seeking and Frame Capture
            for target in targets:
                try:
                    window_name = target.get("window", "custom")
                    ts_sec = target.get("timestamp_sec", 0.0)

                    # Cap ts_sec to duration
                    ts_sec = min(max(0.0, ts_sec), max(0.0, duration_sec - 1.0))

                    # Direct seeking via POS_MSEC
                    pos_msec = ts_sec * 1000.0
                    cap.set(cv2.CAP_PROP_POS_MSEC, pos_msec)
                    ret, frame = cap.read()

                    # Fallback seeking via frame index if POS_MSEC failed
                    if not ret or frame is None:
                        target_frame_idx = max(0, min(int(ts_sec * fps), total_frames - 2))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                        ret, frame = cap.read()

                    if ret and frame is not None:
                        try:
                            actual_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        except Exception:
                            actual_frame_idx = int(ts_sec * fps)

                        # Convert BGR (OpenCV) -> RGB (PIL)
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb_frame)

                        # Encode to JPEG Bytes
                        buf = io.BytesIO()
                        pil_img.save(buf, format="JPEG", quality=jpeg_quality)
                        img_bytes = buf.getvalue()

                        # Base64 string for direct Multimodal API / UI embedding
                        b64_str = base64.b64encode(img_bytes).decode("utf-8")

                        keyframe = Keyframe(
                            window=window_name,
                            timestamp_sec=ts_sec,
                            timestamp_formatted=self.format_seconds(ts_sec),
                            frame_index=actual_frame_idx,
                            image_pil=pil_img,
                            image_bytes=img_bytes,
                            base64_jpeg=b64_str
                        )
                        extracted_keyframes.append(keyframe)
                except cv2.error:
                    continue
                except Exception:
                    continue

            return VideoAnalysis(
                file_path=self.video_path,
                file_name=self.file_name,
                status="success",
                metadata=metadata,
                keyframes=extracted_keyframes,
                total_keyframes=len(extracted_keyframes)
            )

        except Exception as e:
            return VideoAnalysis(
                file_path=self.video_path,
                file_name=self.file_name,
                status="fallback_corrupt_file",
                error_message=f"Error processing video keyframes: {str(e)}"
            )

        finally:
            cap.release()
