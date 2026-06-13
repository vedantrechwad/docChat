"""
Study Tools - Quiz, Flashcard, Study Plan, and Concept Map generators.

Uses the LLM Router for model-agnostic generation. These tools form the
core study companion features beyond basic RAG chat.
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from src.llm.llm_router import LLMRouter, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── DATA MODELS ───────────────────────────────────────────────────────────────

@dataclass
class QuizQuestion:
    """A single quiz question."""
    question: str
    question_type: str  # "mcq", "true_false", "short_answer"
    options: List[str] = field(default_factory=list)  # For MCQ
    correct_answer: str = ""
    explanation: str = ""
    difficulty: str = "medium"  # easy, medium, hard
    source_reference: str = ""


@dataclass
class Quiz:
    """A complete quiz."""
    title: str
    questions: List[QuizQuestion]
    source_document: str
    created_at: str = ""
    total_questions: int = 0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.total_questions = len(self.questions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "questions": [
                {
                    "question": q.question,
                    "type": q.question_type,
                    "options": q.options,
                    "correct_answer": q.correct_answer,
                    "explanation": q.explanation,
                    "difficulty": q.difficulty,
                }
                for q in self.questions
            ],
            "source_document": self.source_document,
            "total_questions": self.total_questions,
            "created_at": self.created_at,
        }


@dataclass
class Flashcard:
    """A single flashcard."""
    front: str  # Question / Term
    back: str   # Answer / Definition
    tags: List[str] = field(default_factory=list)
    difficulty: int = 0  # 0-5 for SRS
    next_review: Optional[str] = None
    review_count: int = 0
    ease_factor: float = 2.5  # For SRS algorithm


@dataclass
class FlashcardDeck:
    """A collection of flashcards."""
    title: str
    cards: List[Flashcard]
    source_document: str
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "cards": [
                {
                    "front": c.front,
                    "back": c.back,
                    "tags": c.tags,
                    "difficulty": c.difficulty,
                    "next_review": c.next_review,
                    "review_count": c.review_count,
                    "ease_factor": c.ease_factor,
                }
                for c in self.cards
            ],
            "source_document": self.source_document,
            "total_cards": len(self.cards),
            "created_at": self.created_at,
        }


@dataclass
class StudyPlanDay:
    """A single day in the study plan."""
    day: int
    date: str
    topics: List[str]
    activities: List[str]
    estimated_hours: float
    notes: str = ""


@dataclass
class StudyPlan:
    """A complete study schedule."""
    title: str
    total_days: int
    schedule: List[StudyPlanDay]
    source_document: str
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


def parse_json_from_llm(text: str) -> Dict[str, Any]:
    """Robustly parse JSON from LLM response."""
    text = text.strip()
    
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
        
    start_idx = text.find('{')
    if start_idx == -1:
        start_idx = text.find('[')
        
    if start_idx != -1:
        start_char = text[start_idx]
        end_char = '}' if start_char == '{' else ']'
        
        depth = 0
        in_string = False
        escape = False
        
        for i in range(start_idx, len(text)):
            char = text[i]
            
            if escape:
                escape = False
                continue
                
            if char == '\\':
                escape = True
                continue
                
            if char == '"':
                in_string = not in_string
                continue
                
            if not in_string:
                if char == start_char:
                    depth += 1
                elif char == end_char:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start_idx:i+1])
                        except json.JSONDecodeError:
                            break
                            
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
            
    raise ValueError(f"Could not parse JSON from response: {text[:200]}")


# ─── QUIZ GENERATOR ───────────────────────────────────────────────────────────

class QuizGenerator:
    """Generate quizzes from document content."""

    def __init__(self, llm_router: LLMRouter):
        self.llm_router = llm_router
        logger.info("QuizGenerator initialized")

    def generate_quiz(
        self,
        content: str,
        source_name: str = "Document",
        num_questions: int = 10,
        question_types: List[str] = None,
        difficulty: str = "mixed",
    ) -> Quiz:
        """
        Generate a quiz from document content.
        
        Args:
            content: The document text to generate questions from
            source_name: Name of the source document
            num_questions: Number of questions to generate
            question_types: Types of questions ("mcq", "true_false", "short_answer")
            difficulty: "easy", "medium", "hard", or "mixed"
        """
        if question_types is None:
            question_types = ["mcq", "true_false", "short_answer"]

        types_str = ", ".join(question_types)

        prompt = f"""You are a quiz generator for a study companion app. Generate a quiz based on the following content.

REQUIREMENTS:
- Generate exactly {num_questions} questions
- Question types to include: {types_str}
- Difficulty level: {difficulty}
- Each question must be directly based on the provided content
- For MCQ questions, provide exactly 4 options (A, B, C, D)
- Include a brief explanation for each correct answer

RESPONSE FORMAT (strict JSON):
{{
    "questions": [
        {{
            "question": "What is...?",
            "type": "mcq",
            "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
            "correct_answer": "A) Option 1",
            "explanation": "Because...",
            "difficulty": "medium"
        }},
        {{
            "question": "True or False: ...",
            "type": "true_false",
            "options": ["True", "False"],
            "correct_answer": "True",
            "explanation": "Because...",
            "difficulty": "easy"
        }},
        {{
            "question": "Explain...",
            "type": "short_answer",
            "options": [],
            "correct_answer": "The answer is...",
            "explanation": "Key points: ...",
            "difficulty": "hard"
        }}
    ]
}}

CONTENT:
{content[:6000]}

Generate the quiz now:"""

        try:
            response = self.llm_router.generate(
                prompt=prompt,
                task_type=TaskType.QUIZ_GENERATE,
                temperature=0.3,
                max_tokens=4000,
            )

            quiz_data = parse_json_from_llm(response.content)
            questions = []

            for q_data in quiz_data.get("questions", []):
                questions.append(QuizQuestion(
                    question=q_data.get("question", ""),
                    question_type=q_data.get("type", "mcq"),
                    options=q_data.get("options", []),
                    correct_answer=q_data.get("correct_answer", ""),
                    explanation=q_data.get("explanation", ""),
                    difficulty=q_data.get("difficulty", "medium"),
                    source_reference=source_name,
                ))

            quiz = Quiz(
                title=f"Quiz: {source_name}",
                questions=questions,
                source_document=source_name,
            )

            logger.info(f"Generated quiz with {len(questions)} questions")
            return quiz

        except Exception as e:
            logger.error(f"Error generating quiz: {e}")
            raise

    def grade_answer(self, question: QuizQuestion, user_answer: str) -> Dict[str, Any]:
        """Grade a user's answer against the correct answer."""
        prompt = f"""Grade this answer:

Question: {question.question}
Correct Answer: {question.correct_answer}
Student Answer: {user_answer}

Respond with JSON:
{{
    "is_correct": true/false,
    "score": 0-100,
    "feedback": "Your explanation of why the answer is correct/incorrect"
}}"""

        response = self.llm_router.generate(
            prompt=prompt,
            task_type=TaskType.EXPLAIN,
            temperature=0.1,
            max_tokens=300,
        )

        return parse_json_from_llm(response.content)


# ─── FLASHCARD GENERATOR ──────────────────────────────────────────────────────

class FlashcardGenerator:
    """Generate flashcards from document content."""

    def __init__(self, llm_router: LLMRouter):
        self.llm_router = llm_router
        logger.info("FlashcardGenerator initialized")

    def generate_flashcards(
        self,
        content: str,
        source_name: str = "Document",
        num_cards: int = 20,
        style: str = "concept",  # "concept", "vocabulary", "qa"
    ) -> FlashcardDeck:
        """Generate flashcards from content."""

        prompt = f"""You are a flashcard generator for a study companion. Extract the most important concepts, terms, and facts from the content below and create flashcards.

STYLE: {style}
- "concept": Front = concept name, Back = detailed explanation
- "vocabulary": Front = term, Back = definition with example
- "qa": Front = question, Back = concise answer

Generate exactly {num_cards} flashcards.

RESPONSE FORMAT (strict JSON):
{{
    "cards": [
        {{
            "front": "What is X?",
            "back": "X is... because...",
            "tags": ["topic1", "topic2"]
        }}
    ]
}}

CONTENT:
{content[:6000]}

Generate flashcards now:"""

        try:
            response = self.llm_router.generate(
                prompt=prompt,
                task_type=TaskType.QUIZ_GENERATE,
                temperature=0.3,
                max_tokens=4000,
            )

            data = parse_json_from_llm(response.content)
            cards = []

            for c_data in data.get("cards", []):
                cards.append(Flashcard(
                    front=c_data.get("front", ""),
                    back=c_data.get("back", ""),
                    tags=c_data.get("tags", []),
                    next_review=datetime.now().isoformat(),
                ))

            deck = FlashcardDeck(
                title=f"Flashcards: {source_name}",
                cards=cards,
                source_document=source_name,
            )

            logger.info(f"Generated {len(cards)} flashcards")
            return deck

        except Exception as e:
            logger.error(f"Error generating flashcards: {e}")
            raise

    def review_card(self, card: Flashcard, quality: int) -> Flashcard:
        """
        Update flashcard based on review quality using SM-2 algorithm.
        
        Args:
            card: The flashcard being reviewed
            quality: 0-5 rating (0=forgot, 5=perfect recall)
        """
        card.review_count += 1

        if quality >= 3:
            if card.review_count == 1:
                interval = 1
            elif card.review_count == 2:
                interval = 6
            else:
                interval = int(card.difficulty * card.ease_factor)

            card.ease_factor = max(1.3, card.ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
        else:
            card.review_count = 0
            interval = 1

        card.difficulty = quality
        card.next_review = (datetime.now() + timedelta(days=interval)).isoformat()

        return card


# ─── STUDY PLANNER ─────────────────────────────────────────────────────────────

class StudyPlanner:
    """Generate study plans from syllabus or document content."""

    def __init__(self, llm_router: LLMRouter):
        self.llm_router = llm_router
        logger.info("StudyPlanner initialized")

    def generate_study_plan(
        self,
        content: str,
        source_name: str = "Syllabus",
        total_days: int = 14,
        hours_per_day: float = 3.0,
        exam_date: Optional[str] = None,
    ) -> StudyPlan:
        """Generate a structured study schedule."""

        exam_note = f"\nExam date: {exam_date}. Plan accordingly with revision days before the exam." if exam_date else ""

        prompt = f"""You are a study planner assistant. Create a detailed study schedule based on the following content/syllabus.

PARAMETERS:
- Total study days: {total_days}
- Hours per day: {hours_per_day}{exam_note}

REQUIREMENTS:
- Break down topics into manageable daily chunks
- Include revision and practice days
- Mix different activities (reading, practice problems, review)
- Start with fundamentals, build to complex topics
- Include at least 1 full revision day at the end

RESPONSE FORMAT (strict JSON):
{{
    "schedule": [
        {{
            "day": 1,
            "topics": ["Topic A - Part 1", "Topic B basics"],
            "activities": ["Read chapter 1", "Practice 5 problems", "Create flashcards"],
            "estimated_hours": 3.0,
            "notes": "Focus on understanding fundamentals"
        }}
    ]
}}

CONTENT/SYLLABUS:
{content[:6000]}

Generate the study plan now:"""

        try:
            response = self.llm_router.generate(
                prompt=prompt,
                task_type=TaskType.STUDY_PLAN,
                temperature=0.3,
                max_tokens=4000,
            )

            data = parse_json_from_llm(response.content)
            days = []
            start_date = datetime.now()

            for d_data in data.get("schedule", []):
                day_num = d_data.get("day", len(days) + 1)
                days.append(StudyPlanDay(
                    day=day_num,
                    date=(start_date + timedelta(days=day_num - 1)).strftime("%Y-%m-%d"),
                    topics=d_data.get("topics", []),
                    activities=d_data.get("activities", []),
                    estimated_hours=d_data.get("estimated_hours", hours_per_day),
                    notes=d_data.get("notes", ""),
                ))

            plan = StudyPlan(
                title=f"Study Plan: {source_name}",
                total_days=total_days,
                schedule=days,
                source_document=source_name,
            )

            logger.info(f"Generated study plan with {len(days)} days")
            return plan

        except Exception as e:
            logger.error(f"Error generating study plan: {e}")
            raise


# ─── CONCEPT MAP GENERATOR ────────────────────────────────────────────────────

class ConceptMapGenerator:
    """Generate concept maps from document content."""

    def __init__(self, llm_router: LLMRouter):
        self.llm_router = llm_router
        logger.info("ConceptMapGenerator initialized")

    def generate_concept_map(
        self,
        content: str,
        source_name: str = "Document",
        max_concepts: int = 15,
    ) -> Dict[str, Any]:
        """
        Generate a concept map as a Mermaid.js diagram.
        
        Returns a dict with:
        - "mermaid": The Mermaid.js diagram code
        - "concepts": List of extracted concepts
        - "relationships": List of relationships between concepts
        """

        prompt = f"""You are a concept map generator. Analyze the following content and extract the key concepts and their relationships.

REQUIREMENTS:
- Extract up to {max_concepts} key concepts
- Identify relationships between them (e.g., "is a type of", "causes", "depends on")
- Generate a Mermaid.js flowchart diagram

RESPONSE FORMAT (strict JSON):
{{
    "concepts": ["Concept A", "Concept B", "Concept C"],
    "relationships": [
        {{"from": "Concept A", "to": "Concept B", "label": "causes"}},
        {{"from": "Concept B", "to": "Concept C", "label": "leads to"}}
    ],
    "mermaid": "graph TD\\n    A[Concept A] -->|causes| B[Concept B]\\n    B -->|leads to| C[Concept C]"
}}

CONTENT:
{content[:5000]}

Generate the concept map now:"""

        try:
            response = self.llm_router.generate(
                prompt=prompt,
                task_type=TaskType.CONCEPT_MAP,
                temperature=0.2,
                max_tokens=2000,
            )

            data = parse_json_from_llm(response.content)
            logger.info(f"Generated concept map with {len(data.get('concepts', []))} concepts")
            return data

        except Exception as e:
            logger.error(f"Error generating concept map: {e}")
            raise


# ─── MICRO AI FEATURES ────────────────────────────────────────────────────────

class MicroAITools:
    """
    Small, fast AI features powered by lightweight local models (phi3).
    These provide the "magic" micro-interactions in the study companion.
    """

    def __init__(self, llm_router: LLMRouter):
        self.llm_router = llm_router
        logger.info("MicroAITools initialized")

    def explain_like_im_5(self, term: str, context: str = "") -> str:
        """Explain a complex term in simple language."""
        ctx = f"\nContext: {context}" if context else ""
        prompt = f"Explain '{term}' in one simple sentence that a 5-year-old would understand.{ctx}\nJust give the explanation, nothing else."

        response = self.llm_router.generate(
            prompt=prompt,
            task_type=TaskType.EXPLAIN,
            temperature=0.3,
            max_tokens=100,
        )
        return response.content.strip()

    def generate_mnemonic(self, items: List[str]) -> str:
        """Generate a mnemonic device for a list of items."""
        items_str = ", ".join(items)
        prompt = f"Create a catchy, memorable mnemonic or acronym to remember these items: {items_str}\nJust provide the mnemonic and a brief explanation."

        response = self.llm_router.generate(
            prompt=prompt,
            task_type=TaskType.MNEMONIC,
            temperature=0.7,
            max_tokens=200,
        )
        return response.content.strip()

    def extract_glossary(self, content: str, max_terms: int = 20) -> List[Dict[str, str]]:
        """Extract key terms and definitions from content."""
        prompt = f"""Extract the {max_terms} most important terms and their definitions from this text.

RESPONSE FORMAT (strict JSON):
{{
    "glossary": [
        {{"term": "Term Name", "definition": "Brief definition"}}
    ]
}}

TEXT:
{content[:4000]}

Extract glossary now:"""

        response = self.llm_router.generate(
            prompt=prompt,
            task_type=TaskType.GLOSSARY,
            temperature=0.1,
            max_tokens=2000,
        )

        try:
            text = response.content.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            data = json.loads(text.strip())
            return data.get("glossary", [])
        except Exception:
            return []

    def fix_grammar(self, text: str) -> str:
        """Fix grammar and improve clarity."""
        prompt = f"Fix the grammar and improve clarity of this text. Return only the corrected text:\n\n{text}"

        response = self.llm_router.generate(
            prompt=prompt,
            task_type=TaskType.GRAMMAR,
            temperature=0.1,
            max_tokens=len(text) + 200,
        )
        return response.content.strip()

    def simplify_text(self, text: str) -> str:
        """Simplify complex text for easier understanding."""
        prompt = f"Simplify this text so it's easier to understand. Keep the key information but use simpler words:\n\n{text}"

        response = self.llm_router.generate(
            prompt=prompt,
            task_type=TaskType.EXPLAIN,
            temperature=0.2,
            max_tokens=len(text) + 200,
        )
        return response.content.strip()

    def expand_text(self, text: str, context: str = "") -> str:
        """Expand a brief text with more details."""
        ctx = f"\nAdditional context: {context}" if context else ""
        prompt = f"Expand the following text with more details and explanations:{ctx}\n\n{text}"

        response = self.llm_router.generate(
            prompt=prompt,
            task_type=TaskType.EXPAND,
            temperature=0.4,
            max_tokens=1000,
        )
        return response.content.strip()

    def generate_focus_question(self, content: str) -> Dict[str, str]:
        """Generate a quick True/False focus check question about content."""
        prompt = f"""Generate one simple True/False question about this content to check active recall.

RESPONSE FORMAT (JSON):
{{"question": "...", "answer": "True/False", "explanation": "..."}}

CONTENT:
{content[:2000]}"""

        response = self.llm_router.generate(
            prompt=prompt,
            task_type=TaskType.QUIZ_GENERATE,
            temperature=0.3,
            max_tokens=200,
        )

        try:
            text = response.content.strip()
            if text.startswith("```"):
                text = re.sub(r'^```\w*\n?', '', text)
                text = text.rstrip('`').strip()
            return json.loads(text)
        except Exception:
            return {"question": "Could not generate question", "answer": "", "explanation": ""}
