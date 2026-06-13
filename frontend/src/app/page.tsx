import styles from './page.module.css';
import { BookOpen, Map, Layers, Target, FileText, ArrowRight, Clock, MoreHorizontal } from 'lucide-react';
import Link from 'next/link';

export default function Home() {
  return (
    <div className={styles.dashboard}>
      <div className={styles.welcomeSection}>
        <h1 className={styles.title}>
          Welcome back, <span className="gradient-text">Student</span>
        </h1>
        <p className={styles.subtitle}>Ready to continue your learning journey?</p>
      </div>

      <div className={styles.grid}>
        {/* Chat & Sources Card */}
        <Link href="/sources" className="glass-panel card hover">
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <div className={styles.iconWrapper}>
                <BookOpen size={24} />
              </div>
              <h2 className={styles.cardTitle}>Sources & Chat</h2>
            </div>
            <p className={styles.cardDescription}>
              Upload your documents, notes, and lectures. Chat with your customized AI assistant to query your specific materials.
            </p>
            <div className={styles.cardAction}>
              Open Library <ArrowRight size={16} />
            </div>
          </div>
        </Link>

        {/* Canvas Card */}
        <Link href="/canvas" className="glass-panel card hover">
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <div className={styles.iconWrapper}>
                <Map size={24} />
              </div>
              <h2 className={styles.cardTitle}>Infinite Canvas</h2>
            </div>
            <p className={styles.cardDescription}>
              A spatial workspace to visually organize notes, extract concepts, and connect ideas with AI assistance.
            </p>
            <div className={styles.cardAction}>
              Open Canvas <ArrowRight size={16} />
            </div>
          </div>
        </Link>

        {/* Flashcards Card */}
        <Link href="/flashcards" className="glass-panel card hover">
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <div className={styles.iconWrapper}>
                <Layers size={24} />
              </div>
              <h2 className={styles.cardTitle}>Flashcards (SRS)</h2>
            </div>
            <p className={styles.cardDescription}>
              Auto-generated flashcards from your sources with Spaced Repetition System (SM-2) for maximum retention.
            </p>
            <div className={styles.cardAction}>
              Start Review <ArrowRight size={16} />
            </div>
          </div>
        </Link>

        {/* Study Planner Card */}
        <Link href="/planner" className="glass-panel card hover">
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <div className={styles.iconWrapper}>
                <Target size={24} />
              </div>
              <h2 className={styles.cardTitle}>Study Planner</h2>
            </div>
            <p className={styles.cardDescription}>
              Upload a syllabus and get an AI-generated, optimized daily study schedule leading up to your exam.
            </p>
            <div className={styles.cardAction}>
              View Plan <ArrowRight size={16} />
            </div>
          </div>
        </Link>
      </div>

      <div className={styles.recentSection}>
        <div className={styles.sectionTitle}>
          <h3>Recent Materials</h3>
          <button className="btn-primary" style={{ padding: '0.5rem 1rem', fontSize: '0.9rem' }}>
            Upload New
          </button>
        </div>
        
        <div className={styles.recentList}>
          {/* Mock Recent Items */}
          {[
            { title: "Advanced Calculus Chapter 4", type: "PDF Document", time: "2 hours ago" },
            { title: "Quantum Physics Lecture Audio", type: "Audio Transcript", time: "Yesterday" },
            { title: "Machine Learning Concept Map", type: "Canvas Workspace", time: "3 days ago" },
          ].map((item, i) => (
            <div key={i} className={`glass-panel ${styles.recentItem}`}>
              <div className={styles.itemInfo}>
                <div className={styles.itemIcon}>
                  <FileText size={20} />
                </div>
                <div>
                  <div className={styles.itemTitle}>{item.title}</div>
                  <div className={styles.itemMeta}>
                    {item.type} • <Clock size={12} style={{ display: 'inline', marginLeft: '4px', marginRight: '2px' }} /> {item.time}
                  </div>
                </div>
              </div>
              <div className={styles.itemAction}>
                <MoreHorizontal size={20} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
