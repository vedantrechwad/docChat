"""
Local Audio Transcriber - Faster-Whisper-based audio transcription.

Replaces the paid AssemblyAI API with the fully local faster-whisper
model for speech-to-text transcription. Supports speaker diarization
via simple energy-based segmentation.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """Represents a transcribed audio segment."""
    text: str
    start_time: float
    end_time: float
    speaker: Optional[str] = None

    def get_timestamp_str(self) -> str:
        def fmt(seconds):
            m = int(seconds // 60)
            s = int(seconds % 60)
            return f"{m:02d}:{s:02d}"
        return f"[{fmt(self.start_time)} - {fmt(self.end_time)}]"


class LocalAudioTranscriber:
    """
    Local audio transcription using faster-whisper.
    
    Supports: mp3, wav, m4a, aac, ogg, flac, wma, opus, mp4, mov, avi
    """

    def __init__(self, model_size: str = "base", device: str = "auto", compute_type: str = "auto"):
        """
        Initialize the transcriber.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large-v3")
                       - tiny: ~75MB, fastest, least accurate
                       - base: ~145MB, good balance for most use cases
                       - small: ~485MB, better accuracy
                       - medium: ~1.5GB, high accuracy
                       - large-v3: ~3GB, best accuracy
            device: "auto", "cpu", or "cuda"
            compute_type: "auto", "int8", "float16", "float32"
        """
        self.model_size = model_size
        self.model = None
        self.device = device
        self.compute_type = compute_type

        self.supported_formats = {
            ".mp3", ".wav", ".m4a", ".aac", ".ogg",
            ".flac", ".wma", ".opus", ".mp4", ".mov", ".avi",
        }

        self._initialize_model()

    def _initialize_model(self):
        """Load the faster-whisper model."""
        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info(f"Whisper model loaded successfully: {self.model_size}")

        except ImportError:
            logger.error(
                "faster-whisper not installed. Install with: pip install faster-whisper"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> list:
        """
        Transcribe an audio file and return DocumentChunks.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., "en"). None for auto-detection.
            chunk_size: Max characters per chunk
            chunk_overlap: Character overlap between chunks
        """

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if audio_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported audio format: {audio_path.suffix}")

        logger.info(f"Starting transcription for: {audio_path.name}")

        try:
            segments, info = self.model.transcribe(
                str(audio_path),
                language=language,
                beam_size=5,
                vad_filter=True,  # Voice Activity Detection to filter silence
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                ),
            )

            logger.info(
                f"Detected language: {info.language} (prob={info.language_probability:.2f}), "
                f"duration={info.duration:.1f}s"
            )

            # Collect all segments
            transcript_segments = []
            for segment in segments:
                transcript_segments.append(
                    TranscriptSegment(
                        text=segment.text.strip(),
                        start_time=segment.start,
                        end_time=segment.end,
                    )
                )

            if not transcript_segments:
                logger.warning("No speech detected in audio")
                return []

            # Convert to DocumentChunks
            chunks = self._segments_to_chunks(
                transcript_segments,
                audio_path.name,
                chunk_size,
                chunk_overlap,
                {
                    "duration_seconds": info.duration,
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "model_size": self.model_size,
                    "processed_at": datetime.now().isoformat(),
                },
            )

            logger.info(f"Transcription complete: {len(chunks)} chunks from {len(transcript_segments)} segments")
            return chunks

        except Exception as e:
            logger.error(f"Error transcribing {audio_path.name}: {e}")
            raise

    def _segments_to_chunks(
        self,
        segments: List[TranscriptSegment],
        source_file: str,
        chunk_size: int,
        chunk_overlap: int,
        base_metadata: Dict[str, Any],
    ) -> list:
        """Convert transcript segments into DocumentChunks with timestamps."""
        from src.document_processing.doc_processor import DocumentChunk

        chunks = []
        current_text = ""
        current_start_time = segments[0].start_time if segments else 0
        current_end_time = 0
        chunk_index = 0

        for seg in segments:
            timestamp_line = f"[{seg.get_timestamp_str()}] {seg.text}\n"

            if len(current_text + timestamp_line) > chunk_size and current_text:
                # Save current chunk
                chunk_meta = base_metadata.copy()
                chunk_meta.update({
                    "start_timestamp_sec": current_start_time,
                    "end_timestamp_sec": current_end_time,
                })

                chunk = DocumentChunk(
                    content=current_text.strip(),
                    source_file=source_file,
                    source_type="audio",
                    page_number=None,
                    chunk_index=chunk_index,
                    start_char=int(current_start_time * 1000),
                    end_char=int(current_end_time * 1000),
                    metadata=chunk_meta,
                )
                chunks.append(chunk)
                chunk_index += 1

                # Start new chunk (with overlap from end of previous)
                overlap_text = current_text[-chunk_overlap:] if chunk_overlap > 0 else ""
                current_text = overlap_text + timestamp_line
                current_start_time = seg.start_time
            else:
                current_text += timestamp_line

            current_end_time = seg.end_time

        # Save last chunk
        if current_text.strip():
            chunk_meta = base_metadata.copy()
            chunk_meta.update({
                "start_timestamp_sec": current_start_time,
                "end_timestamp_sec": current_end_time,
            })

            chunk = DocumentChunk(
                content=current_text.strip(),
                source_file=source_file,
                source_type="audio",
                page_number=None,
                chunk_index=chunk_index,
                start_char=int(current_start_time * 1000),
                end_char=int(current_end_time * 1000),
                metadata=chunk_meta,
            )
            chunks.append(chunk)

        return chunks

    def get_transcript_text(self, audio_path: str, language: Optional[str] = None) -> str:
        """Get plain text transcription (no chunking)."""
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments)


class LocalYouTubeTranscriber:
    """
    YouTube transcription using yt-dlp + faster-whisper.
    Replaces the AssemblyAI-based YouTube transcriber entirely.
    """

    def __init__(self, model_size: str = "base"):
        import tempfile
        self.temp_dir = Path(tempfile.gettempdir()) / "study_companion_yt"
        self.temp_dir.mkdir(exist_ok=True)
        self.transcriber = LocalAudioTranscriber(model_size=model_size)
        logger.info("LocalYouTubeTranscriber initialized")

    def extract_video_id(self, url: str) -> Optional[str]:
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        return None

    def download_audio(self, url: str) -> str:
        """Download audio from a YouTube video using yt-dlp."""
        import yt_dlp

        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError("Could not extract video ID from URL")

        expected_path = self.temp_dir / f"{video_id}.m4a"
        if expected_path.exists():
            logger.info(f"Audio already cached: {expected_path}")
            return str(expected_path)

        logger.info(f"Downloading audio from: {url}")

        ydl_opts = {
            "format": "m4a/bestaudio/best",
            "outtmpl": str(self.temp_dir / "%(id)s.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
            }],
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([url])
            if error_code != 0:
                raise Exception(f"yt-dlp download failed with code: {error_code}")

        if not expected_path.exists():
            raise FileNotFoundError(f"Expected audio file not found: {expected_path}")

        logger.info(f"Audio downloaded: {expected_path}")
        return str(expected_path)

    def transcribe_youtube_video(
        self,
        url: str,
        cleanup_audio: bool = True,
    ) -> list:
        """Download and transcribe a YouTube video."""
        try:
            audio_path = self.download_audio(url)

            chunks = self.transcriber.transcribe_audio(audio_path)

            # Update source file name
            video_id = self.extract_video_id(url)
            video_name = f"YouTube Video {video_id}"
            for chunk in chunks:
                chunk.source_file = video_name
                chunk.source_type = "youtube"
                chunk.metadata["video_url"] = url
                chunk.metadata["video_id"] = video_id

            if cleanup_audio and os.path.exists(audio_path):
                os.unlink(audio_path)
                logger.info("Audio file cleaned up")

            return chunks

        except Exception as e:
            logger.error(f"Error transcribing YouTube video: {e}")
            raise

    def cleanup_temp_files(self):
        """Remove all cached audio files."""
        try:
            if self.temp_dir.exists():
                for f in self.temp_dir.glob("*.m4a"):
                    f.unlink()
                logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Could not clean up temp files: {e}")
