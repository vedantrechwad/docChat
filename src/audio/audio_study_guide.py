"""
Audio Study Guide Generator - Generates study-focused audio scripts.
Uses the LLM Router instead of OpenAI directly. Generates lecture-style
study guides and concise audio summaries.
"""

import logging
import json
from typing import List, Dict, Any
from dataclasses import dataclass

from src.llm.llm_router import LLMRouter, TaskType
from src.document_processing.doc_processor import DocumentProcessor
from src.study_tools import parse_json_from_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioStudyGuide:
    """Represents an audio study guide script."""
    script: List[Dict[str, str]]
    source_document: str
    total_lines: int
    estimated_duration: str
    guide_type: str  # "lecture", "summary", "q_and_a"

    def get_speaker_lines(self, speaker: str) -> List[str]:
        return [item[speaker] for item in self.script if speaker in item]

    def to_json(self) -> str:
        return json.dumps({
            'script': self.script,
            'metadata': {
                'source_document': self.source_document,
                'total_lines': self.total_lines,
                'estimated_duration': self.estimated_duration,
                'guide_type': self.guide_type,
            }
        }, indent=2)


class AudioStudyGuideGenerator:
    """
    Generates audio study guide scripts using the LLM Router.
    
    Supports three modes:
    - "lecture": A focused, educational lecture explaining the material
    - "summary": A concise TL;DR audio summary of key points
    - "q_and_a": A tutor asking questions and providing answers
    """

    def __init__(self, llm_router: LLMRouter) -> None:
        self.llm_router = llm_router
        self.doc_processor = DocumentProcessor()
        logger.info("AudioStudyGuideGenerator initialized")

    def generate_from_text(
        self,
        text_content: str,
        source_name: str = "Document",
        guide_type: str = "lecture",
        target_duration: str = "10 minutes",
    ) -> AudioStudyGuide:
        """Generate a study guide script from text content."""

        logger.info(f"Generating '{guide_type}' audio study guide")

        script_data = self._generate_script(
            text_content, guide_type, target_duration
        )

        guide = AudioStudyGuide(
            script=script_data['script'],
            source_document=source_name,
            total_lines=len(script_data['script']),
            estimated_duration=target_duration,
            guide_type=guide_type,
        )

        logger.info(f"Generated study guide with {guide.total_lines} segments")
        return guide

    def generate_from_document(
        self,
        document_path: str,
        guide_type: str = "lecture",
        target_duration: str = "10 minutes",
    ) -> AudioStudyGuide:
        """Generate a study guide from a document file."""

        chunks = self.doc_processor.process_document(document_path)
        if not chunks:
            raise ValueError("No content extracted from document")

        content = "\n\n".join(c.content for c in chunks)
        source_name = chunks[0].source_file

        return self.generate_from_text(content, source_name, guide_type, target_duration)

    def _generate_script(
        self,
        content: str,
        guide_type: str,
        target_duration: str,
    ) -> Dict[str, Any]:
        """Generate the script using the LLM Router."""

        type_prompts = {
            "lecture": "Create an educational lecture script with a single speaker (\"Tutor\") explaining \n"
                       "the material clearly and thoroughly. The tutor should:\n"
                       "- Start with a brief overview of what will be covered\n"
                       "- Break down complex topics into digestible explanations\n"
                       "- Use examples and analogies to clarify difficult concepts\n"
                       "- Summarize key takeaways at the end",

            "summary": "Create a concise audio summary with a single speaker (\"Narrator\") highlighting \n"
                       "only the most critical points. The narrator should:\n"
                       "- Get straight to the key points without fluff\n"
                       "- Use bullet-point style delivery (short, punchy statements)\n"
                       "- Cover the main ideas in order of importance\n"
                       "- End with 3-5 key takeaways to remember",

            "q_and_a": "Create a study Q&A format with two speakers:\n"
                       "- \"Tutor\" who asks targeted study questions\n"
                       "- \"Student\" who provides clear, accurate answers\n"
                       "The questions should test understanding of key concepts and the answers\n"
                       "should reinforce learning with clear explanations.",
        }

        type_instruction = type_prompts.get(guide_type, type_prompts["lecture"])

        duration_guidelines = {
            "5 minutes": "Keep it concise, focus on 3-4 main points.",
            "10 minutes": "Cover key topics thoroughly with good explanations.",
            "15 minutes": "Comprehensive coverage with detailed discussions.",
            "20 minutes": "In-depth exploration with extensive analysis.",
        }

        duration_guide = duration_guidelines.get(target_duration, duration_guidelines["10 minutes"])

        if guide_type == "q_and_a":
            speakers = '"Tutor" or "Student"'
        else:
            speakers = '"Tutor"' if guide_type == "lecture" else '"Narrator"'

        prompt = (f"Generate an audio study guide script based on the following content.\n\n"
                  f"GUIDE TYPE: {guide_type}\n{type_instruction}\n\n"
                  f"DURATION: {target_duration}\n{duration_guide}\n\n"
                  f"RULES:\n"
                  f"1. Each segment should be 2-4 sentences maximum\n"
                  f"2. Use clear, educational language\n"
                  f"3. Make it engaging and easy to follow when listened to\n"
                  f"4. Maintain professional grammar and punctuation\n"
                  f"5. Focus on helping the listener understand and remember the material\n\n"
                  f"RESPONSE FORMAT:\n"
                  f"Respond with valid JSON containing a 'script' array. Each element should have \n"
                  f"{speakers} as the key and their dialogue as the value.\n\n"
                  f"Example:\n{{\n"
                  f"  \"script\": [\n"
                  f"    {{\"Tutor\": \"Welcome to today's study session. We'll be covering the key concepts from your notes...\"}},\n"
                  f"    {{\"Tutor\": \"Let's start with the first major topic...\"}}\n"
                  f"  ]\n}}\n\n"
                  f"CONTENT:\n{content[:8000]}\n\n"
                  f"Generate the {target_duration} study guide now:")

        try:
            response_obj = self.llm_router.generate(
                prompt=prompt,
                task_type=TaskType.PODCAST_SCRIPT,
                temperature=0.5,
                max_tokens=4000,
            )
            
            response = str(response_obj.content) if response_obj.content is not None else ""

            script_data = parse_json_from_llm(response)

            if 'script' not in script_data or not isinstance(script_data['script'], list):
                raise ValueError("Invalid script format")

            validated = self._validate_script(script_data['script'])
            return {'script': validated}

        except json.JSONDecodeError:
            # Fallback parsing shouldn't fail since we parse inside _parse_json_response
            raise ValueError("Failed to decode JSON from LLM response")
        except Exception as e:
            raise ValueError(f"Error generating script: {str(e)}")

    def _validate_script(self, script: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Validate and clean up the generated script."""
        cleaned: List[Dict[str, str]] = []
        for item in script:
            if not isinstance(item, dict) or len(item) != 1:
                continue

            speaker, dialogue = next(iter(item.items()))
            if not isinstance(speaker, str) or not isinstance(dialogue, str):
                continue
                
            speaker = speaker.strip()
            dialogue = dialogue.strip()

            if not dialogue:
                continue
            if not dialogue.endswith(('.', '!', '?')):
                dialogue += '.'

            cleaned.append({speaker: dialogue})

        if len(cleaned) < 2:
            raise ValueError("Generated script is too short")

        return cleaned
