"use client";

import { useState, useEffect, useRef } from 'react';
import styles from './page.module.css';
import { BookOpen, UploadCloud, FileText, MonitorPlay, Globe, FileAudio, Bot, User, Send, Settings2, Loader2, Map } from 'lucide-react';
import { useChatCanvas } from '../context/ChatCanvasContext';

type Source = {
  name: string;
  type: string;
  size: string;
  chunks: number;
};

type Citation = {
  source_file?: string;
  source_name?: string;
};

type ChatMessage = {
  role: 'assistant' | 'user';
  content: string;
  sources?: Citation[];
};

export default function SourcesPage() {
  const [sources, setSources] = useState<Source[]>([]);
  const { pendingCanvasItems, addCanvasItem, clearCanvasItems, chatMessages: messages, setChatMessages: setMessages } = useChatCanvas();
  const [inputValue, setInputValue] = useState('');
  const [urlInputValue, setUrlInputValue] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [selection, setSelection] = useState<{ text: string, x: number, y: number } | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const fetchSources = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/sources');
      if (res.ok) {
        const data = await res.json();
        setSources(data.sources || []);
      }
    } catch {
      setSources([]);
    }
  };

  useEffect(() => {
    const loadSources = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/sources');
        if (res.ok) {
          const data: { sources?: Source[] } = await res.json();
          setSources(data.sources || []);
        }
      } catch {
        setSources([]);
      }
    };

    void loadSources();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  useEffect(() => {
    const handleMouseUp = () => {
      const activeSelection = window.getSelection();
      if (activeSelection && activeSelection.toString().trim() && !isTyping) {
        const range = activeSelection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        
        // Show popover slightly above the selection
        setSelection({
          text: activeSelection.toString().trim(),
          x: rect.left + (rect.width / 2),
          y: rect.top - 10
        });
      } else {
        // Delay hiding slightly to allow clicks on the popover to register before it disappears
        setTimeout(() => setSelection(null), 150);
      }
    };

    document.addEventListener('mouseup', handleMouseUp);
    return () => document.removeEventListener('mouseup', handleMouseUp);
  }, [isTyping]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    
    setIsUploading(true);
    const formData = new FormData();
    for (let i = 0; i < e.target.files.length; i++) {
      formData.append('files', e.target.files[i]);
    }

    try {
      const res = await fetch('http://localhost:8000/api/sources/upload', {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        await fetchSources();
      } else {
        alert("Upload failed. Make sure the backend is running and pipeline is initialized.");
      }
    } catch (error) {
      console.error("Upload error", error);
      alert("Error connecting to backend.");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleUrlUpload = async () => {
    if (!urlInputValue.trim()) return;
    setIsUploading(true);
    try {
      const res = await fetch('http://localhost:8000/api/sources/url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ urls: [urlInputValue.trim()] }),
      });
      if (res.ok) {
        await fetchSources();
        setUrlInputValue('');
      } else {
        alert("URL upload failed. Make sure it's a valid webpage.");
      }
    } catch (error) {
      console.error("URL upload error", error);
      alert("Error connecting to backend.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleYouTubeUpload = async () => {
    if (!urlInputValue.trim()) return;
    setIsUploading(true);
    try {
      const res = await fetch('http://localhost:8000/api/sources/youtube', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: urlInputValue.trim() }),
      });
      if (res.ok) {
        await fetchSources();
        setUrlInputValue('');
      } else {
        alert("YouTube upload failed. Make sure it's a valid YouTube URL.");
      }
    } catch (error) {
      console.error("YouTube upload error", error);
      alert("Error connecting to backend.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleSendMessage = async () => {    if (!inputValue.trim()) return;

    const userMsg: ChatMessage = { role: 'user', content: inputValue };
    setMessages(prev => [...prev, userMsg]);
    setInputValue('');
    setIsTyping(true);

    try {
      const res = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMsg.content, max_chunks: 8 }),
      });

      if (res.ok) {
        const data = await res.json();
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: data.response,
          sources: data.sources_used
        }]);
      } else {
        setMessages(prev => [...prev, { role: 'assistant', content: "Sorry, I encountered an error. Have you initialized the settings and uploaded documents?" }]);
      }
    } catch (error) {
      console.error("Chat error", error);
      setMessages(prev => [...prev, { role: 'assistant', content: "Error connecting to the backend API." }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const getSourceIcon = (type: string) => {
    if (type.toLowerCase().includes('audio')) return <FileAudio size={20} />;
    if (type.toLowerCase().includes('youtube')) return <MonitorPlay size={20} />;
    if (type.toLowerCase().includes('web') || type.toLowerCase().includes('url')) return <Globe size={20} />;
    return <FileText size={20} />;
  };

  return (
    <div className={styles.sourcesContainer}>
      
      {/* Left Panel: Sources */}
      <div className={styles.leftPanel}>
        <div className={styles.panelHeader}>
          <div className={styles.panelTitle}>
            <BookOpen size={20} />
            My Sources ({sources.length})
          </div>
          <button className="iconButton" style={{ padding: '0.25rem', background: 'transparent' }} onClick={fetchSources}>
            <Settings2 size={18} />
          </button>
        </div>

        <input 
          type="file" 
          multiple 
          ref={fileInputRef} 
          style={{ display: 'none' }} 
          onChange={handleFileUpload} 
        />
        <div 
          className={styles.uploadArea} 
          onClick={() => fileInputRef.current?.click()}
          style={{ cursor: isUploading ? 'not-allowed' : 'pointer', opacity: isUploading ? 0.7 : 1 }}
        >
          {isUploading ? <Loader2 size={32} className={`${styles.uploadIcon} animate-spin`} /> : <UploadCloud size={32} className={styles.uploadIcon} />}
          <div style={{ fontWeight: 500, marginBottom: '0.25rem' }}>{isUploading ? 'Uploading...' : 'Upload Document'}</div>
          <div className={styles.uploadText}>PDF, Word, TXT, or Audio/Video</div>
        </div>

        <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <input 
            type="text" 
            placeholder="Paste Web URL or YouTube Link..." 
            value={urlInputValue}
            onChange={(e) => setUrlInputValue(e.target.value)}
            style={{ padding: '0.5rem', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'transparent', color: 'var(--text-primary)' }}
            disabled={isUploading}
          />
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button 
              onClick={handleUrlUpload} 
              disabled={isUploading || !urlInputValue.trim()}
              style={{ flex: 1, padding: '0.4rem', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-panel)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px' }}>
              <Globe size={14} /> Web Page
            </button>
            <button 
              onClick={handleYouTubeUpload} 
              disabled={isUploading || !urlInputValue.trim()}
              style={{ flex: 1, padding: '0.4rem', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-panel)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px' }}>
              <MonitorPlay size={14} /> YouTube
            </button>
          </div>
        </div>

        <div className={styles.sourcesList}>          {sources.length === 0 && !isUploading && (
            <div style={{ textAlign: 'center', color: 'var(--text-muted)', marginTop: '2rem', fontSize: '0.9rem' }}>
              No sources uploaded yet.
            </div>
          )}
          {sources.map((src, i) => (
            <div key={i} className={`glass-panel ${styles.sourceItem}`}>
              <div className={styles.sourceIcon}>
                {getSourceIcon(src.type)}
              </div>
              <div className={styles.sourceDetails}>
                <div className={styles.sourceName}>{src.name}</div>
                <div className={styles.sourceMeta}>{src.type} • {src.size} • {src.chunks} chunks</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Right Panel: Chat Interface */}
      <div className={`glass-panel ${styles.rightPanel}`}>
        <div className={styles.chatHeader}>
          <Bot size={24} className="gradient-text" />
          <div style={{ fontWeight: 600 }}>Study Companion AI</div>
        </div>

        <div className={styles.chatMessages}>
          {messages.map((msg, i) => (
            <div key={i} className={`${styles.message} ${msg.role === 'user' ? styles.user : ''}`}>
              <div className={`${styles.avatar} ${msg.role === 'user' ? styles.userAvatar : styles.aiAvatar}`}>
                {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
              </div>
              <div className={styles.messageContent}>
                <div style={{ whiteSpace: 'pre-wrap' }}>
                  {msg.sources && msg.sources.length > 0 ? (
                    msg.content.split(/(\[\d+\])/g).map((part, idx) => {
                      const match = part.match(/\[(\d+)\]/);
                      if (match) {
                        const sourceIdx = parseInt(match[1]) - 1;
                        const source = msg.sources![sourceIdx];
                        if (source) {
                          return (
                            <span 
                              key={idx} 
                              title={source.content || "Source text"}
                              style={{ 
                                backgroundColor: 'var(--primary-color, #4a90e2)', 
                                color: 'white', 
                                borderRadius: '4px', 
                                padding: '2px 6px', 
                                fontSize: '0.75rem', 
                                cursor: 'help',
                                margin: '0 2px'
                              }}
                            >
                              {match[1]}
                            </span>
                          );
                        }
                      }
                      return <span key={idx}>{part}</span>;
                    })
                  ) : (
                    msg.content
                  )}
                </div>
                
                {msg.role === 'assistant' && (
                  <div style={{ marginTop: '0.75rem' }}>
                    <button 
                      onClick={() => { addCanvasItem(msg.content); alert("Added to Canvas Queue! Go to Canvas to see it."); }}
                      className="iconButton" 
                      style={{ fontSize: '0.8rem', display: 'inline-flex', alignItems: 'center', gap: '6px', background: 'var(--bg-panel)', padding: '6px 12px', borderRadius: '6px', border: '1px solid var(--border-color)', cursor: 'pointer' }}>
                      <Map size={14} className="gradient-text" /> Send to Canvas
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
          {isTyping && (
            <div className={styles.message}>
              <div className={`${styles.avatar} ${styles.aiAvatar}`}>
                <Bot size={20} />
              </div>
              <div className={styles.messageContent}>
                <div className="flex gap-1" style={{ display: 'flex', gap: '4px', alignItems: 'center', height: '24px' }}>
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{width: '6px', height: '6px', borderRadius: '50%', background: 'var(--text-muted)'}}></span>
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{width: '6px', height: '6px', borderRadius: '50%', background: 'var(--text-muted)', animationDelay: '0.2s'}}></span>
                  <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{width: '6px', height: '6px', borderRadius: '50%', background: 'var(--text-muted)', animationDelay: '0.4s'}}></span>
                </div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className={styles.chatInputArea}>
          <div className={styles.inputWrapper}>
            <input 
              type="text" 
              className={styles.chatInput} 
              placeholder="Ask a question about your documents..." 
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isTyping}
            />
            <button className={styles.sendButton} onClick={handleSendMessage} disabled={isTyping || !inputValue.trim()}>
              <Send size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Text Selection Popover */}
      {selection && (
        <div style={{
          position: 'fixed',
          top: selection.y,
          left: selection.x,
          transform: 'translate(-50%, -100%)',
          backgroundColor: 'var(--bg-panel)',
          border: '1px solid var(--border-color)',
          borderRadius: '8px',
          boxShadow: 'var(--shadow-lg)',
          padding: '6px',
          display: 'flex',
          gap: '4px',
          zIndex: 1000,
          backdropFilter: 'blur(10px)'
        }}
        onMouseDown={(e) => e.stopPropagation()} // Prevent closing when clicking the menu
        >
          <button 
            className="iconButton" 
            style={{ fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '4px', padding: '4px 8px', background: 'var(--bg-surface)' }}
            onClick={(e) => {
              e.stopPropagation();
              addCanvasItem(selection.text);
              alert("Added to Canvas Queue! Go to Canvas to see it.");
              setSelection(null);
              window.getSelection()?.removeAllRanges();
            }}
          >
            <Map size={14} className="gradient-text" /> Send to Canvas
          </button>
          <div style={{ width: '1px', backgroundColor: 'var(--border-color)', margin: '0 4px' }}></div>
          <button 
            className="iconButton" 
            style={{ fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '4px', padding: '4px 8px', background: 'var(--bg-surface)' }}
            onClick={(e) => {
              e.stopPropagation();
              setInputValue(`Explain this: "${selection.text}"`);
              setSelection(null);
              window.getSelection()?.removeAllRanges();
            }}
          >
            <Bot size={14} className="gradient-text" /> Ask AI
          </button>
        </div>
      )}
    </div>
  );
}
