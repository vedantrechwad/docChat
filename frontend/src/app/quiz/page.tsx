"use client";

import { useState } from 'react';
import styles from './page.module.css';
import { Target, ArrowRight, Loader2, RefreshCw } from 'lucide-react';

type QuizQuestion = {
  question: string;
  type: string;
  options: string[];
  correct_answer: string;
  explanation: string;
};

type Quiz = {
  title: string;
  questions: QuizQuestion[];
};

export default function QuizPage() {
  const [quiz, setQuiz] = useState<Quiz | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [score, setScore] = useState(0);

  const generateQuiz = async () => {
    setIsGenerating(true);
    setQuiz(null);
    setCurrentIndex(0);
    setScore(0);
    setSelectedAnswer(null);
    setShowExplanation(false);

    try {
      const res = await fetch('http://localhost:8000/api/quiz/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_questions: 5,
          question_types: ['mcq']
        })
      });

      if (res.ok) {
        const data = await res.json();
        setQuiz(data);
      } else {
        alert("Failed to generate quiz. Have you uploaded documents?");
      }
    } catch (e) {
      console.error(e);
      alert("Error connecting to backend");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSelect = (opt: string) => {
    if (showExplanation) return;
    setSelectedAnswer(opt);
  };

  const handleNext = () => {
    if (!quiz) return;

    if (!showExplanation) {
      const q = quiz.questions[currentIndex];
      if (selectedAnswer === q.correct_answer) {
        setScore(prev => prev + 1);
      }
      setShowExplanation(true);
    } else {
      setShowExplanation(false);
      setSelectedAnswer(null);
      setCurrentIndex(prev => prev + 1);
    }
  };

  if (!quiz) {
    return (
      <div className={styles.quizContainer} style={{ justifyContent: 'center', alignItems: 'center' }}>
        <Target className="gradient-text" size={64} style={{ marginBottom: '1rem' }} />
        <h1 style={{ marginBottom: '1rem' }}>Test Your Knowledge</h1>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem', textAlign: 'center' }}>
          Generate a custom quiz based on your uploaded documents.
        </p>
        <button className="btn-primary" onClick={generateQuiz} disabled={isGenerating}>
          {isGenerating ? <Loader2 className="animate-spin" /> : "Generate Quiz"}
        </button>
      </div>
    );
  }

  if (currentIndex >= quiz.questions.length) {
    return (
      <div className={styles.quizContainer} style={{ justifyContent: 'center', alignItems: 'center' }}>
        <Target className="gradient-text" size={64} style={{ marginBottom: '1rem' }} />
        <h1 style={{ marginBottom: '1rem' }}>Quiz Completed!</h1>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem', fontSize: '1.25rem' }}>
          You scored {score} out of {quiz.questions.length}
        </p>
        <button className="btn-primary" onClick={generateQuiz} style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <RefreshCw size={18} /> Try Another Quiz
        </button>
      </div>
    );
  }

  const q = quiz.questions[currentIndex];

  return (
    <div className={styles.quizContainer}>
      <div className={styles.header}>
        <h1 className={styles.title}>
          <Target className="gradient-text" size={32} />
          {quiz.title || "Knowledge Check"}
        </h1>
        <div className={styles.progress}>
          Question {currentIndex + 1} of {quiz.questions.length}
        </div>
      </div>

      <div className={styles.quizCard}>
        <div className={styles.questionType}>{q.type.toUpperCase()}</div>
        
        <div className={styles.questionText}>{q.question}</div>

        <div className={styles.optionsList}>
          {q.options?.map((opt: string, i: number) => {
            const isSelected = selectedAnswer === opt;
            const isCorrect = opt === q.correct_answer;
            
            let bgClass = '';
            if (showExplanation) {
              if (isCorrect) bgClass = 'border-green-500 bg-green-500/10';
              else if (isSelected) bgClass = 'border-red-500 bg-red-500/10';
            } else if (isSelected) {
              bgClass = styles.selected;
            }

            return (
              <div 
                key={i} 
                className={`${styles.option} ${bgClass}`}
                onClick={() => handleSelect(opt)}
                style={{
                  borderColor: showExplanation && isCorrect ? '#10b981' : showExplanation && isSelected ? '#ef4444' : undefined,
                  background: showExplanation && isCorrect ? 'rgba(16,185,129,0.1)' : showExplanation && isSelected ? 'rgba(239,68,68,0.1)' : undefined
                }}
              >
                <div className={styles.optionRadio} style={{
                  borderColor: showExplanation && isCorrect ? '#10b981' : showExplanation && isSelected ? '#ef4444' : undefined,
                  background: isSelected || (showExplanation && isCorrect) ? (isCorrect ? '#10b981' : '#ef4444') : undefined
                }}></div>
                {opt}
              </div>
            );
          })}
        </div>

        {showExplanation && (
          <div style={{ marginTop: '2rem', padding: '1rem', background: 'var(--bg-secondary)', borderRadius: '8px', borderLeft: '4px solid var(--accent-primary)' }}>
            <strong>Explanation:</strong> {q.explanation}
          </div>
        )}

        <div className={styles.actions}>
          <button 
            className="btn-primary" 
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
            onClick={handleNext}
            disabled={!selectedAnswer}
          >
            {showExplanation ? "Next Question" : "Check Answer"} <ArrowRight size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}
