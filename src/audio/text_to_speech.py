import logging
import os
import asyncio
from typing import List, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass

try:
    import edge_tts # type: ignore
except ImportError:
    print("edge-tts not installed. Install with: pip install edge-tts>=6.1.12")
    edge_tts = None # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """Represents a single audio segment with metadata"""
    speaker: str
    text: str
    file_path: str


class PodcastTTSGenerator:
    """
    Generates audio files using edge-tts.
    Replaced kokoro to avoid C++ compilation requirements.
    """
    def __init__(self) -> None:
        if edge_tts is None:
            raise ImportError("edge-tts not available. Install with: pip install edge-tts>=6.1.12")
        
        self.speaker_voices: Dict[str, str] = {
            "Speaker 1": "en-US-AriaNeural",
            "Speaker 2": "en-US-GuyNeural",
            "Tutor": "en-US-AriaNeural",
            "Narrator": "en-GB-SoniaNeural",
            "Student": "en-US-GuyNeural"
        }
        
        logger.info("Edge-TTS generator initialized")
    
    def generate_podcast_audio(
        self, 
        script_data: Union[Dict[str, Any], List[Dict[str, str]]],
        output_dir: str = "outputs/podcast_audio",
    ) -> List[str]:
        """
        Takes the script data and generates mp3 files.
        Since edge-tts generates separate files easily, we'll return the list of mp3 paths.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Determine the list of lines to process
        lines: List[Dict[str, str]] = []
        if hasattr(script_data, 'script'):
            lines = getattr(script_data, 'script')
        elif isinstance(script_data, dict) and 'script' in script_data:
            lines = script_data['script']
        elif isinstance(script_data, list):
            lines = script_data
            
        total_lines = len(lines)
        logger.info(f"Generating audio for {total_lines} segments")
        logger.info(f"Output directory: {output_dir}")
        
        output_files: List[str] = []
        
        for i, line_dict in enumerate(lines):
            speaker, dialogue = next(iter(line_dict.items()))
            logger.info(f"Processing segment {i+1}/{total_lines}: {speaker}")
            
            try:
                segment_filename = f"segment_{i+1:03d}_{speaker.replace(' ', '_').lower()}.mp3"
                segment_path = os.path.join(output_dir, segment_filename)
                
                # Run the async generation synchronously
                asyncio.run(self._generate_single_segment(speaker, dialogue, segment_path))
                
                output_files.append(segment_path)
                logger.info(f"✓ Generated segment {i+1}: {segment_filename}")
                
            except Exception as e:
                logger.error(f"✗ Failed to generate segment {i+1}: {str(e)}")
                continue
        
        logger.info(f"Audio generation complete! Generated {len(output_files)} files")
        return output_files
    
    async def _generate_single_segment(self, speaker: str, text: str, output_path: str) -> None:
        voice = self.speaker_voices.get(speaker, "en-US-AriaNeural")
        clean_text = self._clean_text_for_tts(text)

        communicate = edge_tts.Communicate(clean_text, voice)
        await communicate.save(output_path)
    
    def _clean_text_for_tts(self, text: str) -> str:
        clean_text = text.strip()
        clean_text = clean_text.replace("...", ".")
        clean_text = clean_text.replace("!!", "!")
        clean_text = clean_text.replace("??", "?")

        if not clean_text.endswith(('.', '!', '?')):
            clean_text += '.'
        
        return clean_text


if __name__ == "__main__":
    try:
        tts_generator = PodcastTTSGenerator()
        
        sample_script = [
            {"Tutor": "Welcome everyone to our study guide! Today we're exploring artificial intelligence."},
            {"Student": "Thanks for having me! AI is indeed very fascinating."}
        ]
        
        print("Generating podcast audio...")
        output_files = tts_generator.generate_podcast_audio(
            sample_script,
            output_dir="./podcast_output"
        )
        
        print("\nGenerated files:")
        for file_path in output_files:
            print(f"  - {file_path}")
        
        print("\nEdge-TTS test completed successfully!")
        
    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Error: {e}")
