"use client";

import { useState, useEffect } from 'react';
import styles from './page.module.css';
import { Cloud, HardDrive, CheckCircle, ServerCog } from 'lucide-react';

type ModelMode = 'local' | 'cloud';
type CloudProvider = 'groq' | 'openai' | 'gemini';

export default function SettingsPage() {
  const [modelMode, setModelMode] = useState<ModelMode>('local');
  const [cloudProvider, setCloudProvider] = useState<CloudProvider>('gemini');
  const [primaryModel, setPrimaryModel] = useState('llama3');
  const [fastModel, setFastModel] = useState('phi3');
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [isSaving, setIsSaving] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');

  // Fetch available models on load
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/models');
        if (res.ok) {
          const data = await res.json();
          setAvailableModels(data.models || []);
        }
      } catch {
        setAvailableModels([]);
      }
    };
    fetchModels();
  }, []);

  const handleSave = async () => {
    setIsSaving(true);
    setStatusMessage('Saving settings...');
    
    try {
      const res = await fetch('http://localhost:8000/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_mode: modelMode,
          cloud_provider: cloudProvider,
          primary_model: primaryModel,
          fast_model: fastModel,
        }),
      });

      if (res.ok) {
        setStatusMessage('Settings saved successfully!');
        setTimeout(() => setStatusMessage(''), 3000);
      } else {
        const err = await res.json();
        setStatusMessage(`Error: ${err.detail}`);
      }
    } catch (error) {
      console.error(error);
      setStatusMessage('Error: Could not connect to backend.');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className={styles.settingsContainer}>
      <div className={styles.header}>
        <h1 className={styles.title}>Settings</h1>
        <p className={styles.subtitle}>Configure your AI models, cloud fallbacks, and local resources.</p>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>
          <ServerCog className="gradient-text" size={24} />
          Model Runtime
        </h2>
        <p className={styles.description} style={{ marginBottom: '1.5rem' }}>
          Choose whether generation should prefer your local Ollama models or a cloud provider configured through the backend environment.
        </p>

        <div className={styles.modeGrid}>
          <button
            type="button"
            className={`${styles.modeCard} ${modelMode === 'local' ? styles.modeCardActive : ''}`}
            onClick={() => setModelMode('local')}
          >
            <HardDrive size={22} />
            <span>Local Ollama</span>
            <small>Private, offline-friendly, runs on your machine.</small>
          </button>
          <button
            type="button"
            className={`${styles.modeCard} ${modelMode === 'cloud' ? styles.modeCardActive : ''}`}
            onClick={() => setModelMode('cloud')}
          >
            <Cloud size={22} />
            <span>Cloud API</span>
            <small>Faster hosted models, using keys from your `.env` file.</small>
          </button>
        </div>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>
          <HardDrive className="gradient-text" size={24} />
          Local Models (Ollama)
          <span className={styles.badge}>Free & Private</span>
        </h2>
        
        <div className={styles.formGroup}>
          <label className={styles.label}>Primary Model (Complex Tasks)</label>
          <p className={styles.description}>Used for RAG chat, study planning, and detailed summaries.</p>
          <select 
            className={styles.select} 
            value={primaryModel}
            onChange={(e) => setPrimaryModel(e.target.value)}
          >
            <option value="llama3">Llama 3 (8B) - Recommended</option>
            <option value="llama3:70b">Llama 3 (70B) - Requires 64GB+ RAM</option>
            <option value="mistral">Mistral (7B)</option>
            <option value="gemma">Gemma (7B)</option>
            {availableModels.filter(m => !["llama3", "llama3:70b", "mistral", "gemma"].includes(m)).map(m => (
              <option key={m} value={m}>{m} (Detected)</option>
            ))}
          </select>
        </div>

        <div className={styles.formGroup}>
          <label className={styles.label}>Fast Model (Micro Features)</label>
          <p className={styles.description}>Used for quick actions like ELI5, grammar fixes, and flashcard generation.</p>
          <select 
            className={styles.select} 
            value={fastModel}
            onChange={(e) => setFastModel(e.target.value)}
          >
            <option value="phi3">Phi-3 Mini (3.8B) - Recommended</option>
            <option value="qwen2:0.5b">Qwen 2 (0.5B) - Ultra fast</option>
            <option value="llama3">Llama 3 (8B)</option>
            {availableModels.filter(m => !["phi3", "qwen2:0.5b", "llama3"].includes(m)).map(m => (
              <option key={m} value={m}>{m} (Detected)</option>
            ))}
          </select>
        </div>

        <div className={styles.statusIndicator}>
          <div className={styles.statusDot}></div>
          <span style={{ fontWeight: 500 }}>Ollama Server is running at http://localhost:11434</span>
        </div>
      </div>

      <div className={styles.section}>
        <h2 className={styles.sectionTitle}>
          <Cloud className="gradient-text" size={24} />
          Cloud Provider
        </h2>
        <p className={styles.description} style={{ marginBottom: '1.5rem' }}>
          API keys are intentionally not entered in the UI. Add them to `.env` as `GROQ_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY`; missing keys are ignored until you add them.
        </p>

        <div className={styles.formGroup}>
          <label className={styles.label}>Preferred Cloud Provider</label>
          <select
            className={styles.select}
            value={cloudProvider}
            onChange={(e) => setCloudProvider(e.target.value as CloudProvider)}
            disabled={modelMode !== 'cloud'}
          >
            <option value="gemini">Gemini 2.5 Flash</option>
            <option value="groq">Groq Llama 3.3 70B</option>
            <option value="openai">OpenAI GPT-4o Mini</option>
          </select>
          <p className={styles.description} style={{ marginTop: '0.75rem' }}>
            If the selected cloud key is absent, the backend keeps running and falls back to local models.
          </p>
        </div>
      </div>

      <div className={styles.actions} style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '1rem' }}>
        {statusMessage && <span style={{ color: 'var(--accent-primary)', fontWeight: 500 }}>{statusMessage}</span>}
        <button 
          className="btn-primary" 
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
          onClick={handleSave}
          disabled={isSaving}
        >
          <CheckCircle size={18} />
          {isSaving ? 'Saving...' : 'Save Settings'}
        </button>
      </div>
    </div>
  );
}
