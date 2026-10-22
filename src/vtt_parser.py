"""
WebVTT Transcript Parser and Keyword Analyzer for 8x8 Class Quality Assessment System.
Parses .vtt files into structured conversation metrics, spoken text, student callouts,
greeting indicators, and homework assignments.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import re
import os


@dataclass
class VTTCue:
    index: int
    start_time: float
    end_time: float
    formatted_start: str
    formatted_end: str
    speaker: str
    text: str
    word_count: int


@dataclass
class VTTAnalysis:
    file_path: str
    file_name: str
    total_duration_sec: float
    formatted_duration: str
    total_word_count: int
    speaker_turn_count: int
    speaker_turn_density_per_min: float
    speakers: List[str]
    tutor_name: Optional[str]
    student_names: List[str]
    student_callout_count: int
    student_callouts: List[Dict[str, Any]]
    greeting_detected: bool
    greeting_count: int
    greeting_details: List[Dict[str, Any]]
    homework_detected: bool
    homework_detected_tail: bool
    homework_count: int
    homework_tail_count: int
    homework_details: List[Dict[str, Any]]
    clean_transcript: str
    cues: List[VTTCue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res['cues'] = [asdict(c) for c in self.cues]
        return res


class VTTParser:
    """
    Robust WebVTT Parser and Pedagogical Keyword Analyzer.
    """

    GREETING_PATTERNS = [
        r'\bhello\b', r'\bhi\b', r'\bhey\b', r'\bgood\s+evening\b', r'\bgood\s+morning\b',
        r'\bgood\s+afternoon\b', r'\bwelcome\b', r'\bhow\s+are\s+you\b', r'\bnice\s+to\s+meet\b',
        r'\bnamaste\b', r'\bvanakkam\b', r'\bgood\s+day\b'
    ]

    HOMEWORK_PATTERNS = [
        r'\bhomework\b', r'\bhw\b', r'\bassignment\b', r'\bassignments\b',
        r'\bhome\s+work\b', r'\bpractice\s+task\b', r'\bsubmit\b', r'\bnext\s+class\b',
        r'\bexercise\b'
    ]

    def __init__(self, filepath_or_content: str, is_content: bool = False, file_name: str = ""):
        self.raw_content = ""
        self.file_path = ""
        self.file_name = file_name

        if is_content:
            self.raw_content = filepath_or_content
            if not self.file_name:
                self.file_name = "transcript.vtt"
        else:
            self.file_path = filepath_or_content
            if not self.file_name:
                self.file_name = os.path.basename(filepath_or_content)
            self._read_file(filepath_or_content)

    def _read_file(self, filepath: str) -> None:
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        for enc in encodings:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    self.raw_content = f.read()
                return
            except UnicodeDecodeError:
                continue
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            self.raw_content = f.read()

    @staticmethod
    def parse_timestamp_to_seconds(ts_str: str) -> float:
        ts_str = ts_str.strip().replace(',', '.')
        parts = ts_str.split(':')
        try:
            if len(parts) == 3:
                h, m, s = parts
                return float(h) * 3600 + float(m) * 60 + float(s)
            elif len(parts) == 2:
                m, s = parts
                return float(m) * 60 + float(s)
            return float(parts[0])
        except ValueError:
            return 0.0

    @staticmethod
    def format_seconds(seconds: float) -> str:
        seconds = max(0, int(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    @staticmethod
    def clean_text_and_speaker(raw_text: str) -> Tuple[str, Optional[str]]:
        # Remove voice tags <v SpeakerName>text</v>
        speaker_tag = None
        v_match = re.match(r'^<v(?:\.[^\s>]+)?\s+([^>]+)>(.*)', raw_text, re.IGNORECASE | re.DOTALL)
        if v_match:
            speaker_tag = v_match.group(1).strip()
            raw_text = v_match.group(2)
            raw_text = re.sub(r'</v>', '', raw_text, flags=re.IGNORECASE)

        # Strip html tags
        text = re.sub(r'<[^>]+>', '', raw_text)
        # Decode common HTML entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&nbsp;', ' ')
        text = re.sub(r'\s+', ' ', text).strip()

        if not speaker_tag:
            # Check Speaker: text format
            spk_match = re.match(r'^([^:\n]+):\s*(.*)', text)
            if spk_match:
                cand = spk_match.group(1).strip()
                rest = spk_match.group(2).strip()
                if not cand.isdigit() and len(cand) < 40 and not re.match(r'^\d{2}:', cand):
                    speaker_tag = cand
                    text = rest

        return text, speaker_tag

    def parse(self) -> VTTAnalysis:
        blocks = re.split(r'\n\s*\n', self.raw_content)

        cues: List[VTTCue] = []
        speaker_words: Dict[str, int] = {}
        last_speaker = "Unknown"
        cue_idx = 1

        for block in blocks:
            lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
            if not lines:
                continue

            ts_line_idx = -1
            start_ts_str, end_ts_str = None, None
            for idx, line in enumerate(lines):
                if '-->' in line:
                    parts = line.split('-->')
                    start_ts_str = parts[0].strip().split()[0]
                    end_ts_str = parts[1].strip().split()[0]
                    ts_line_idx = idx
                    break

            if ts_line_idx == -1 or not start_ts_str or not end_ts_str:
                continue

            # Gather text lines after timestamp line
            text_lines = []
            for line in lines[ts_line_idx + 1:]:
                if line.startswith('WEBVTT') or line.startswith('NOTE') or line.isdigit():
                    continue
                text_lines.append(line)

            if not text_lines:
                continue

            raw_text = ' '.join(text_lines)
            clean_text, spk = self.clean_text_and_speaker(raw_text)

            if spk:
                current_speaker = spk
            else:
                current_speaker = last_speaker

            if current_speaker != "Unknown":
                last_speaker = current_speaker

            start_sec = self.parse_timestamp_to_seconds(start_ts_str)
            end_sec = self.parse_timestamp_to_seconds(end_ts_str)
            fmt_start = self.format_seconds(start_sec)
            fmt_end = self.format_seconds(end_sec)

            words = len(re.findall(r'\b\w+\b', clean_text))
            if words > 0 or clean_text:
                cues.append(VTTCue(
                    index=cue_idx,
                    start_time=start_sec,
                    end_time=end_sec,
                    formatted_start=fmt_start,
                    formatted_end=fmt_end,
                    speaker=current_speaker,
                    text=clean_text,
                    word_count=words
                ))
                cue_idx += 1
                speaker_words[current_speaker] = speaker_words.get(current_speaker, 0) + words

        # Post-processing Metrics
        total_duration = max([c.end_time for c in cues], default=0.0)
        formatted_duration = self.format_seconds(total_duration)
        total_words = sum([c.word_count for c in cues])

        # Identify Tutor vs Students
        valid_speakers = {k: v for k, v in speaker_words.items() if not k.startswith("Audio shared by") and k != "Unknown"}
        if valid_speakers:
            tutor_name = max(valid_speakers, key=valid_speakers.get)
        else:
            tutor_name = None

        all_speakers = list(speaker_words.keys())
        student_names = [s for s in valid_speakers.keys() if s != tutor_name]

        # Speaker Turn Count & Turn Density
        turn_count = 0
        prev_spk = None
        for c in cues:
            if c.speaker != prev_spk:
                turn_count += 1
                prev_spk = c.speaker

        duration_min = max(0.1, total_duration / 60.0)
        turn_density = round(turn_count / duration_min, 2)

        # Student Callout Analysis
        student_first_names = []
        for s in student_names:
            clean_s = re.sub(r'[^a-zA-Z0-9\s]', '', s).strip()
            fn = clean_s.split()[0] if clean_s else ""
            if len(fn) > 2 and fn.lower() not in ["user", "iphone"]:
                student_first_names.append((s, fn))

        student_callouts = []
        if student_first_names and tutor_name:
            for c in cues:
                # Tutor or media mentioning student name
                if c.speaker == tutor_name or c.speaker.startswith("Audio shared by"):
                    for full_name, fn in student_first_names:
                        pattern = re.compile(r'\b' + re.escape(fn) + r'\b', re.IGNORECASE)
                        if pattern.search(c.text):
                            student_callouts.append({
                                "timestamp": c.formatted_start,
                                "speaker": c.speaker,
                                "student_name": full_name,
                                "first_name": fn,
                                "text": c.text
                            })

        # Greetings Analysis
        greeting_re = re.compile('|'.join(self.GREETING_PATTERNS), re.IGNORECASE)
        greeting_details = []
        for c in cues:
            m = greeting_re.findall(c.text)
            if m:
                greeting_details.append({
                    "timestamp": c.formatted_start,
                    "speaker": c.speaker,
                    "matches": m,
                    "text": c.text
                })

        # Homework Analysis
        hw_re = re.compile('|'.join(self.HOMEWORK_PATTERNS), re.IGNORECASE)
        homework_details = []
        for c in cues:
            m = hw_re.findall(c.text)
            if m:
                homework_details.append({
                    "timestamp": c.formatted_start,
                    "speaker": c.speaker,
                    "matches": m,
                    "text": c.text,
                    "is_tail": c.start_time >= (total_duration * 0.60)
                })

        tail_hw_details = [h for h in homework_details if h["is_tail"]]

        # Clean LLM Ingestion Transcript Format
        formatted_lines = []
        for c in cues:
            formatted_lines.append(f"[{c.formatted_start}] {c.speaker}: {c.text}")
        clean_transcript = "\n".join(formatted_lines)

        return VTTAnalysis(
            file_path=self.file_path,
            file_name=self.file_name,
            total_duration_sec=total_duration,
            formatted_duration=formatted_duration,
            total_word_count=total_words,
            speaker_turn_count=turn_count,
            speaker_turn_density_per_min=turn_density,
            speakers=all_speakers,
            tutor_name=tutor_name,
            student_names=student_names,
            student_callout_count=len(student_callouts),
            student_callouts=student_callouts,
            greeting_detected=len(greeting_details) > 0,
            greeting_count=len(greeting_details),
            greeting_details=greeting_details,
            homework_detected=len(homework_details) > 0,
            homework_detected_tail=len(tail_hw_details) > 0,
            homework_count=len(homework_details),
            homework_tail_count=len(tail_hw_details),
            homework_details=homework_details,
            clean_transcript=clean_transcript,
            cues=cues
        )
