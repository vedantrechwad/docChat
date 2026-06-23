"""
YouTube Transcript Extractor — Extract subtitles from YouTube videos.
"""

import re
import logging
import tempfile
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from src.document_processing.document_chunk import DocumentChunk
from src.document_processing.chunking_service import ChunkingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeTranscriptExtractor:
    """Extract transcripts from YouTube videos using yt-dlp subtitles."""

    def __init__(self, chunking: Optional[ChunkingService] = None):
        self.temp_dir = Path(tempfile.gettempdir()) / "docchat_yt"
        self.temp_dir.mkdir(exist_ok=True)
        self.chunking = chunking or ChunkingService.from_preset("balanced")

    def set_chunking(self, chunking: ChunkingService) -> None:
        self.chunking = chunking

    def extract_video_id(self, url: str) -> Optional[str]:
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        elif "shorts/" in url:
            return url.split("shorts/")[1].split("?")[0]
        return None

    def extract_transcript(self, url: str) -> List[DocumentChunk]:
        import yt_dlp

        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError(f"Could not extract video ID from: {url}")

        logger.info(f"Extracting transcript for YouTube video: {video_id}")

        info_opts = {"quiet": True, "no_warnings": True, "skip_download": True}

        with yt_dlp.YoutubeDL(info_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_title = info.get("title", f"YouTube Video {video_id}")
            description = info.get("description", "")

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

        possible_files = list(self.temp_dir.glob(f"{video_id}*.vtt"))
        transcript_text = ""

        if possible_files:
            sub_path = possible_files[0]
            transcript_text = self._parse_vtt(sub_path)
            for f in possible_files:
                f.unlink(missing_ok=True)
            logger.info(f"Extracted subtitles: {len(transcript_text)} chars")
        elif description:
            transcript_text = f"Video Description:\n\n{description}"
            logger.info("No subtitles found, using video description as fallback")
        else:
            raise ValueError(f"No subtitles or description available for video: {video_id}")

        if not transcript_text.strip():
            raise ValueError(f"Empty transcript for video: {video_id}")

        chunks = self.chunking.create_chunks(
            text=transcript_text,
            source_file=video_title,
            source_type="youtube",
            additional_metadata={
                "video_url": url,
                "video_id": video_id,
                "extracted_at": datetime.now().isoformat(),
            },
        )

        logger.info(f"Created {len(chunks)} chunks from YouTube video: {video_title}")
        return chunks

    def _parse_vtt(self, vtt_path: Path) -> str:
        try:
            content = vtt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = vtt_path.read_text(encoding="latin-1")

        lines = content.split("\n")
        text_lines = []
        seen = set()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
                continue
            if re.match(r"^\d{2}:\d{2}", line):
                continue
            if line.isdigit():
                continue

            clean = re.sub(r"<[^>]+>", "", line)
            clean = re.sub(r"\{[^}]+\}", "", clean)
            clean = clean.strip()

            if clean and clean not in seen:
                seen.add(clean)
                text_lines.append(clean)

        return " ".join(text_lines)
