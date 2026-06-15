"""
YouTube Transcript Extractor — Extract subtitles from YouTube videos.

Uses yt-dlp to download auto-generated or manual subtitles (no audio
download needed). Converts subtitle text into DocumentChunks for RAG.
"""

import re
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from src.document_processing.doc_processor import DocumentChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeTranscriptExtractor:
    """Extract transcripts from YouTube videos using yt-dlp subtitles."""

    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "docchat_yt"
        self.temp_dir.mkdir(exist_ok=True)

    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats."""
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        elif "shorts/" in url:
            return url.split("shorts/")[1].split("?")[0]
        return None

    def extract_transcript(
        self,
        url: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[DocumentChunk]:
        """
        Extract transcript from a YouTube video.

        Tries: manual subtitles → auto-generated subtitles → video description fallback.
        """
        import yt_dlp

        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError(f"Could not extract video ID from: {url}")

        logger.info(f"Extracting transcript for YouTube video: {video_id}")

        # First, get video info (title, description)
        info_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }

        with yt_dlp.YoutubeDL(info_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_title = info.get("title", f"YouTube Video {video_id}")
            description = info.get("description", "")

        # Try to download subtitles
        sub_file = self.temp_dir / f"{video_id}.en.vtt"
        sub_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
            "subtitlesformat": "vtt",
            "outtmpl": str(self.temp_dir / "%(id)s"),
        }

        with yt_dlp.YoutubeDL(sub_opts) as ydl:
            ydl.download([url])

        # Find the subtitle file (yt-dlp naming can vary)
        possible_files = list(self.temp_dir.glob(f"{video_id}*.vtt"))
        transcript_text = ""

        if possible_files:
            sub_path = possible_files[0]
            transcript_text = self._parse_vtt(sub_path)
            # Clean up
            for f in possible_files:
                f.unlink(missing_ok=True)
            logger.info(f"Extracted subtitles: {len(transcript_text)} chars")
        elif description:
            # Fallback to description
            transcript_text = f"Video Description:\n\n{description}"
            logger.info("No subtitles found, using video description as fallback")
        else:
            raise ValueError(f"No subtitles or description available for video: {video_id}")

        if not transcript_text.strip():
            raise ValueError(f"Empty transcript for video: {video_id}")

        # Create chunks
        chunks = self._create_chunks(
            transcript_text, video_title, url, video_id,
            chunk_size, chunk_overlap,
        )

        logger.info(f"Created {len(chunks)} chunks from YouTube video: {video_title}")
        return chunks

    def _parse_vtt(self, vtt_path: Path) -> str:
        """Parse a VTT subtitle file into clean text."""
        try:
            content = vtt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = vtt_path.read_text(encoding="latin-1")

        lines = content.split("\n")
        text_lines = []
        seen = set()

        for line in lines:
            line = line.strip()
            # Skip VTT headers, timestamps, and empty lines
            if not line or line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
                continue
            if re.match(r"^\d{2}:\d{2}", line):  # Timestamp line
                continue
            if line.isdigit():  # Sequence number
                continue

            # Remove HTML tags and VTT formatting
            clean = re.sub(r"<[^>]+>", "", line)
            clean = re.sub(r"\{[^}]+\}", "", clean)
            clean = clean.strip()

            if clean and clean not in seen:
                seen.add(clean)
                text_lines.append(clean)

        return " ".join(text_lines)

    def _create_chunks(
        self, text: str, title: str, url: str, video_id: str,
        chunk_size: int, chunk_overlap: int,
    ) -> List[DocumentChunk]:
        """Split transcript text into DocumentChunks."""
        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Break at sentence boundaries
            if end < len(text):
                for sep in [". ", "? ", "! ", "\n"]:
                    pos = text.rfind(sep, start, end)
                    if pos > start + chunk_size * 0.4:
                        end = pos + len(sep)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    source_file=title,
                    source_type="youtube",
                    chunk_index=idx,
                    start_char=start,
                    end_char=end - 1,
                    metadata={
                        "video_url": url,
                        "video_id": video_id,
                        "extracted_at": datetime.now().isoformat(),
                    },
                ))
                idx += 1

            start = end if end >= start + chunk_size else max(end, start + chunk_size - chunk_overlap)

        return chunks
