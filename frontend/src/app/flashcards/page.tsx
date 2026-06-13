"use client";

import { useState } from 'react';
import styles from './page.module.css';
import { Layers, RotateCw, Plus, Brain, Loader2 } from 'lucide-react';

type Flashcard = {
  front: string;
  back: string;
};

type FlashcardDeck = {
  title: string;
  cards: Flashcard[];
};

export default function FlashcardsPage() {
  const [deck, setDeck] = useState<FlashcardDeck | null>(null);
  const [isFlipped, setIsFlipped] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [topic, setTopic] = useState('');
  const [customFront, setCustomFront] = useState('');
  const [customBack, setCustomBack] = useState('');

  const generateFlashcards = async () => {
    setIsGenerating(true);
    setDeck(null);
    setCurrentIndex(0);
    setIsFlipped(false);

    try {
      const res = await fetch('http://localhost:8000/api/flashcards/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_cards: 10,
          style: 'concept',
          topic: topic.trim() || undefined
        })
      });

      if (res.ok) {
        const data = await res.json();
        setDeck(data);
      } else {
        alert("Failed to generate flashcards. Have you uploaded documents?");
      }
    } catch (e) {
      console.error(e);
      alert("Error connecting to backend");
    } finally {
      setIsGenerating(false);
    }
  };

  const addCustomCard = () => {
    if (!customFront.trim() || !customBack.trim()) return;
    
    const newCard: Flashcard = {
      front: customFront.trim(),
      back: customBack.trim()
    };
    
    if (deck) {
      setDeck({
        ...deck,
        cards: [...deck.cards, newCard]
      });
    } else {
      setDeck({
        title: "Custom Deck",
        cards: [newCard]
      });
    }
    setCustomFront('');
    setCustomBack('');
  };

  const handleNextCard = () => {
    setIsFlipped(false);
    setTimeout(() => setCurrentIndex(prev => prev + 1), 150); // wait for flip animation
  };

  if (!deck) {
    return (
      <div className={styles.flashcardContainer} style={{ justifyContent: 'center', alignItems: 'center' }}>
        <Layers className="gradient-text" size={64} style={{ marginBottom: '1rem' }} />
        <h1 style={{ marginBottom: '1rem' }}>Spaced Repetition Flashcards</h1>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', textAlign: 'center' }}>
          Generate a custom SM-2 flashcard deck based on your uploaded documents.
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', width: '100%', maxWidth: '400px', marginBottom: '2rem' }}>
          <input 
            type="text" 
            placeholder="Specific Topic (Optional)..." 
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            style={{ padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border-color)', background: 'var(--bg-panel)', color: 'var(--text-primary)', width: '100%' }}
            disabled={isGenerating}
          />
          <button className="btn-primary" onClick={generateFlashcards} disabled={isGenerating}>
            {isGenerating ? <Loader2 className="animate-spin" style={{ margin: 'auto' }} /> : "Generate Flashcards"}
          </button>
        </div>

        <div style={{ width: '100%', maxWidth: '400px', borderTop: '1px solid var(--border-color)', paddingTop: '2rem' }}>
          <h2 style={{ marginBottom: '1rem', fontSize: '1.2rem', textAlign: 'center' }}>Or Create Custom Card</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <input 
              type="text" 
              placeholder="Front (Question/Term)..." 
              value={customFront}
              onChange={(e) => setCustomFront(e.target.value)}
              style={{ padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border-color)', background: 'var(--bg-panel)', color: 'var(--text-primary)' }}
            />
            <textarea 
              placeholder="Back (Answer/Definition)..." 
              value={customBack}
              onChange={(e) => setCustomBack(e.target.value)}
              style={{ padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border-color)', background: 'var(--bg-panel)', color: 'var(--text-primary)', minHeight: '80px', resize: 'vertical' }}
            />
            <button className="btn-primary" onClick={addCustomCard} style={{ background: 'var(--bg-surface)', color: 'var(--text-primary)', border: '1px solid var(--border-color)' }}>
              Add to Deck
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (currentIndex >= deck.cards.length) {
    return (
      <div className={styles.flashcardContainer} style={{ justifyContent: 'center', alignItems: 'center' }}>
        <Layers className="gradient-text" size={64} style={{ marginBottom: '1rem' }} />
        <h1 style={{ marginBottom: '1rem' }}>Review Session Complete!</h1>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem', fontSize: '1.25rem' }}>
          You&apos;ve reviewed all {deck.cards.length} cards.
        </p>
        <button className="btn-primary" onClick={generateFlashcards} style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <Plus size={16} /> Generate New Deck
        </button>
      </div>
    );
  }

  const currentCard = deck.cards[currentIndex];

  return (
    <div className={styles.flashcardContainer}>
      <div className={styles.header}>
        <h1 className={styles.title}>
          <Layers className="gradient-text" size={32} />
          {deck.title || "Flashcard Review"}
        </h1>
        <div className={styles.progress}>
          {currentIndex + 1} / {deck.cards.length} Cards
        </div>
        <button 
          className="btn-primary" 
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.5rem 1rem' }}
          onClick={generateFlashcards}
          disabled={isGenerating}
        >
          {isGenerating ? <Loader2 size={16} className="animate-spin" /> : <><Plus size={16} /> Generate More</>}
        </button>
      </div>

      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem', width: '100%', maxWidth: '600px' }}>
        <input 
          type="text" 
          placeholder="New Front..." 
          value={customFront}
          onChange={(e) => setCustomFront(e.target.value)}
          style={{ padding: '0.5rem', borderRadius: '6px', border: '1px solid var(--border-color)', background: 'var(--bg-panel)', color: 'var(--text-primary)', flex: 1 }}
        />
        <input 
          type="text" 
          placeholder="New Back..." 
          value={customBack}
          onChange={(e) => setCustomBack(e.target.value)}
          style={{ padding: '0.5rem', borderRadius: '6px', border: '1px solid var(--border-color)', background: 'var(--bg-panel)', color: 'var(--text-primary)', flex: 1 }}
        />
        <button className="btn-primary" onClick={addCustomCard} style={{ padding: '0.5rem 1rem' }}>
          Add Card
        </button>
      </div>

      <div className={styles.cardArea}>
        <div 
          className={`${styles.flashcard} ${isFlipped ? styles.flipped : ''}`}
          onClick={() => setIsFlipped(!isFlipped)}
        >
          <div className={styles.cardInner}>
            {/* Front */}
            <div className={styles.cardFront}>
              <div className={styles.cardLabel}>Question</div>
              <div className={styles.cardContent}>
                {currentCard.front}
              </div>
              <div className={styles.flipHint}>
                <RotateCw size={14} /> Click to flip
              </div>
            </div>

            {/* Back */}
            <div className={styles.cardBack}>
              <div className={styles.cardLabel}>Answer</div>
              <div className={styles.cardContent}>
                {currentCard.back}
              </div>
              <div className={styles.flipHint}>
                <Brain size={14} /> Rate your recall below
              </div>
            </div>
          </div>
        </div>

        {/* SRS Rating Actions */}
        <div className={styles.actions}>
          <button className={`${styles.rateButton} ${styles.rateHard}`} onClick={handleNextCard}>
            Again
            <span>&lt; 1m</span>
          </button>
          <button className={`${styles.rateButton} ${styles.rateHard}`} onClick={handleNextCard}>
            Hard
            <span>12h</span>
          </button>
          <button className={`${styles.rateButton} ${styles.rateGood}`} onClick={handleNextCard}>
            Good
            <span>1d</span>
          </button>
          <button className={`${styles.rateButton} ${styles.rateEasy}`} onClick={handleNextCard}>
            Easy
            <span>4d</span>
          </button>
        </div>
      </div>
    </div>
  );
}
