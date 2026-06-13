"use client";

import { useState } from 'react';
import styles from './page.module.css';
import { Target, Clock, Download, Loader2 } from 'lucide-react';

type StudyPlanDay = {
  day: number;
  date: string;
  topics: string[];
  activities: string[];
  estimated_hours: number;
  notes?: string;
};

type StudyPlan = {
  title: string;
  total_days: number;
  schedule: StudyPlanDay[];
};

export default function PlannerPage() {
  const [examDate, setExamDate] = useState('');
  const [studyHours, setStudyHours] = useState(2);
  const [isGenerating, setIsGenerating] = useState(false);
  const [plan, setPlan] = useState<StudyPlan | null>(null);

  const generatePlan = async () => {
    setIsGenerating(true);
    setPlan(null);

    try {
      const res = await fetch('http://localhost:8000/api/study-plan/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          exam_date: examDate || null,
          total_days: 14,
          hours_per_day: studyHours,
        })
      });

      if (res.ok) {
        const data = await res.json();
        setPlan(data);
      } else {
        alert("Failed to generate plan. Have you uploaded documents?");
      }
    } catch (e) {
      console.error(e);
      alert("Error connecting to backend");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className={styles.plannerContainer}>
      <div className={styles.header}>
        <h1 className={styles.title}>
          <Target className="gradient-text" size={32} />
          AI Study Planner
        </h1>
        {plan && (
          <div className={styles.controls}>
            <button className="iconButton glass-panel">
              <Download size={18} />
            </button>
          </div>
        )}
      </div>

      <div className={styles.scheduleGrid}>
        
        {/* Configuration Panel */}
        <div className={styles.setupPanel}>
          <h2 className={styles.setupTitle}>Generate Plan</h2>
          
          <div className={styles.formGroup}>
            <label className={styles.label}>Source Material</label>
            <select className={styles.select} defaultValue="all">
              <option value="all">All Library Sources</option>
            </select>
          </div>

          <div className={styles.formGroup}>
            <label className={styles.label}>Exam Date (Optional)</label>
            <input 
              type="date" 
              className={styles.input} 
              value={examDate}
              onChange={e => setExamDate(e.target.value)}
            />
          </div>

          <div className={styles.formGroup}>
            <label className={styles.label}>Study Hours / Day</label>
            <input 
              type="number" 
              className={styles.input} 
              value={studyHours}
              onChange={e => setStudyHours(parseInt(e.target.value) || 1)}
              min={1} 
              max={12} 
            />
          </div>

          <button 
            className={`btn-primary ${styles.generateBtn}`} 
            onClick={generatePlan}
            disabled={isGenerating}
          >
            {isGenerating ? <Loader2 className="animate-spin" /> : "Generate Schedule"}
          </button>
        </div>

        {/* Timeline View */}
        <div className={styles.timelinePanel}>
          {!plan && !isGenerating && (
            <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
              Configure your preferences and click Generate Schedule to create your personalized study plan.
            </div>
          )}

          {isGenerating && (
            <div style={{ display: 'flex', height: '100%', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', gap: '1rem' }}>
              <Loader2 size={48} className="animate-spin gradient-text" />
              <div>Analyzing your documents and creating a schedule...</div>
            </div>
          )}
          
          {plan && plan.schedule.map((day, i) => (
            <div key={i} className={styles.dayCard} style={{ opacity: i === 0 ? 1 : 0.8 }}>
              <div className={styles.dayHeader}>
                <div className={styles.dayTitle}>Day {day.day}: {day.topics.join(', ')}</div>
              </div>
              
              <div className={styles.taskList}>
                {day.activities.map((activity, j) => (
                  <div key={j} className={styles.taskItem}>
                    <div className={styles.checkbox}></div>
                    <div className={styles.taskContent}>
                      <div className={styles.taskTitle}>{activity}</div>
                      <div className={styles.taskDesc}>{day.notes || day.date}</div>
                    </div>
                    <div className={styles.taskTime}>
                      <Clock size={14} /> {day.estimated_hours}h
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

      </div>
    </div>
  );
}
