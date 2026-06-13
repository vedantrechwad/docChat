"use client";

import styles from './page.module.css';
import { Mic, Play, Pause, SkipBack, SkipForward, Volume2, FileText, Settings2, Loader2, Plus } from 'lucide-react';
import { useState } from 'react';

type ScriptLine = Record<string, string>;

type AudioGuideData = {
  title: string;
  guide_type: string;
  duration: string;
  script: ScriptLine[];
};

export default function AudioPage() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [topic, setTopic] = useState('');
  const [guide, setGuide] = useState<AudioGuideData | null>(null);

  const generateGuide = async () => {
    setIsGenerating(true);
    setGuide(null);
    try {
      const res = await fetch('http://localhost:8000/api/audio/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          guide_type: 'lecture',
          target_duration: '5 minutes',
          topic: topic.trim() || undefined
        })
      });

      if (res.ok) {
        const data = await res.json();
        setGuide(data);
      } else {
        alert("Failed to generate audio guide. Have you uploaded documents?");
      }
    } catch (e) {
      console.error(e);
      alert("Error connecting to backend");
    } finally {
      setIsGenerating(false);
    }
  };

  if (!guide) {
    return (
      <div className={styles.audioContainer} style={{ justifyContent: 'center', alignItems: 'center' }}>
        <Mic className="gradient-text" size={64} style={{ marginBottom: '1rem' }} />
        <h1 style={{ marginBottom: '1rem' }}>Audio Study Guides</h1>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', textAlign: 'center' }}>
          Generate a custom lecture or summary based on your documents.
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
          <button className="btn-primary" onClick={generateGuide} disabled={isGenerating}>
            {isGenerating ? <Loader2 className="animate-spin" style={{ margin: 'auto' }} /> : "Generate Audio Guide"}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.audioContainer}>
      <div className={styles.header}>
        <h1 className={styles.title}>
          <Mic className="gradient-text" size={32} />
          {guide.title}
        </h1>
        <button className="btn-primary" style={{ padding: '0.5rem 1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }} onClick={() => setGuide(null)}>
          <Plus size={16} /> New Guide
        </button>
      </div>

      <div className={styles.playerCard}>
        <div className={styles.trackInfo}>
          <div className={styles.trackTitle}>{guide.title}</div>
          <div className={styles.trackMeta}>AI Generated {guide.guide_type} • {guide.duration}</div>
        </div>

        <div className={styles.progressContainer}>
          <div className={styles.progressBar}>
            <div className={styles.progressFill} style={{ width: isPlaying ? '45%' : '0%' }}></div>
          </div>
          <div className={styles.timeInfo}>
            <span>00:00</span>
            <span>{guide.duration}</span>
          </div>
        </div>

        <div className={styles.controls}>
          <button className={styles.subControl}>
            <Volume2 size={20} />
          </button>
          <button className={styles.subControl}>
            <SkipBack size={20} />
          </button>
          
          <button 
            className={styles.mainControl}
            onClick={() => setIsPlaying(!isPlaying)}
          >
            {isPlaying ? <Pause size={28} /> : <Play size={28} style={{ marginLeft: '4px' }} />}
          </button>
          
          <button className={styles.subControl}>
            <SkipForward size={20} />
          </button>
          <button className={styles.subControl}>
            <Settings2 size={20} />
          </button>
        </div>
      </div>

      <div className={styles.transcriptSection}>
        <div className={styles.transcriptTitle}>
          <FileText size={18} /> Interactive Transcript
        </div>
        <div className={styles.transcriptContent}>
          {guide.script.map((lineObj, i) => {
            const speaker = Object.keys(lineObj)[0];
            const text = lineObj[speaker];
            return (
              <div key={i} style={{ marginBottom: '1rem' }}>
                <strong style={{ color: 'var(--primary)', marginRight: '0.5rem' }}>{speaker}:</strong>
                <span>{text}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
