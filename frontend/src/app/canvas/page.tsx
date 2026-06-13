"use client";

import { useEffect, useState } from 'react';
import styles from './page.module.css';
import { Sparkles, FileText, Download, Share2 } from 'lucide-react';
import { Tldraw, createShapeId } from '@tldraw/tldraw';
import '@tldraw/tldraw/tldraw.css';
import { useChatCanvas } from '../context/ChatCanvasContext';

export default function CanvasPage() {
  const { pendingCanvasItems, clearCanvasItems } = useChatCanvas();
  const [editor, setEditor] = useState<any>(null);

  useEffect(() => {
    if (editor && pendingCanvasItems.length > 0) {
      const newShapes = pendingCanvasItems.map((textStr, i) => {
        const text = String(textStr);
        const richText = {
          type: "doc",
          content: text.split("\n").map(line => 
            line ? { type: "paragraph", content: [{ type: "text", text: line }] } : { type: "paragraph" }
          )
        };
        
        return {
          id: createShapeId(),
          type: 'note',
          x: 200 + (i * 50),
          y: 200 + (i * 50),
          props: { richText, size: 's', font: 'sans', align: 'middle' }
        };
      });
      editor.createShapes(newShapes);
      clearCanvasItems();
    }
  }, [editor, pendingCanvasItems, clearCanvasItems]);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 600 }}>Study Canvas</h1>
          <p style={{ color: 'var(--text-secondary)' }}>Visually organize and connect your notes with AI.</p>
        </div>
      </div>

      <div className={styles.canvasContainer}>
        {/* The actual Tldraw infinite canvas */}
        <Tldraw onMount={(editor) => setEditor(editor)} />

        {/* Floating Export Menu */}
        <div className={styles.exportMenu}>
          <button className="iconButton glass-panel">
            <Share2 size={18} />
          </button>
          <button className="iconButton glass-panel">
            <Download size={18} />
          </button>
        </div>

        {/* Floating AI Toolbar */}
        <div className={styles.aiFloatingMenu}>
          <button className={`${styles.aiButton} ${styles.primaryAiButton}`}>
            <Sparkles size={16} />
            Ask AI
          </button>
          
          <div style={{ width: '1px', background: 'var(--border-color)', margin: '0 0.5rem' }}></div>
          
          <button className={styles.aiButton}>
            <FileText size={16} />
            Summarize Selection
          </button>
          <button className={styles.aiButton}>
            Expand
          </button>
          <button className={styles.aiButton}>
            Simplify
          </button>
        </div>
      </div>
    </div>
  );
}
