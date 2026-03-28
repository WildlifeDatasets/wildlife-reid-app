import React, { useState, useEffect, useRef } from 'react';
import { 
  Database, Image as ImageIcon, CheckCircle2, AlertCircle, 
  Settings, Download, Search, ChevronLeft, Save, Play, 
  ArrowRight, Layers, FileUp, Loader2 
} from 'lucide-react';
import './App.css';

const API_BASE = 'http://localhost:8000';

type Identity = {
  id: string;
  count: number;
  representative_image: string;
  index: number;
};

type ImageData = {
  index: number;
  filename: string;
  id: string;
};

type ValidationData = {
  query: ImageData;
  neighbors: {
    index: number;
    id: string;
    filename: string;
    score: number;
  }[];
};

type ErrorCase = {
  index: number;
  filename: string;
  id: string;
};

function App() {
  const [view, setView] = useState<'identities' | 'details' | 'validation' | 'errors'>('identities');
  const [identities, setIdentities] = useState<Identity[]>([]);
  const [selectedIdentity, setSelectedIdentity] = useState<string | null>(null);
  const [identityImages, setIdentityImages] = useState<ImageData[]>([]);
  const [validationData, setValidationData] = useState<ValidationData | null>(null);
  const [errors, setErrors] = useState<ErrorCase[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchText, setSearchText] = useState('');
  const [embeddingProgress, setEmbeddingProgress] = useState(0);
  const [isEmbeddingReady, setIsEmbeddingReady] = useState(false);
  const [status, setStatus] = useState('Ready');
  const [newIdInput, setNewIdInput] = useState('');

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchIdentities();
    checkEmbeddingStatus();
    const interval = setInterval(checkEmbeddingStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const fetchIdentities = async (search = '') => {
    try {
      const resp = await fetch(`${API_BASE}/api/identities?search=${search}`);
      const data = await resp.json();
      setIdentities(data);
    } catch (err) {
      console.error(err);
      setStatus('Failed to fetch identities');
    }
  };

  const checkEmbeddingStatus = async () => {
    try {
      const resp = await fetch(`${API_BASE}/api/tasks/embeddings-status`);
      const data = await resp.json();
      setEmbeddingProgress(data.progress);
      setIsEmbeddingReady(data.ready);
    } catch (err) {
      console.error(err);
    }
  };

  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setSearchText(val);
    fetchIdentities(val);
  };

  const handleLoadMetadata = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.[0]) return;
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    setStatus('Uploading metadata...');
    try {
      const resp = await fetch(`${API_BASE}/api/load-metadata`, {
        method: 'POST',
        body: formData,
      });
      const data = await resp.json();
      setStatus(`Loaded ${data.count} images`);
      fetchIdentities();
    } catch (err) {
      setStatus('Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const openIdentity = async (id: string) => {
    setSelectedIdentity(id);
    setView('details');
    try {
      const resp = await fetch(`${API_BASE}/api/identities/${id}/images`);
      const data = await resp.json();
      setIdentityImages(data);
    } catch (err) {
      console.error(err);
    }
  };

  const openValidation = async (index: number) => {
    if (!isEmbeddingReady) {
      setStatus('Please generate embeddings first!');
      return;
    }
    setLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/api/images/${index}/validation`);
      if (!resp.ok) {
        const error = await resp.json();
        throw new Error(error.detail || 'Failed to load validation data');
      }
      const data = await resp.json();
      setValidationData(data);
      setNewIdInput(data.query.id);
      setView('validation');
    } catch (err: any) {
      console.error(err);
      setStatus(err.message || 'Failed to load validation data');
    } finally {
      setLoading(false);
    }
  };

  const updateId = async () => {
    if (!validationData) return;
    try {
      await fetch(`${API_BASE}/api/images/${validationData.query.index}/id`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ new_id: newIdInput }),
      });
      setStatus('ID updated successfully');
      // Refresh current view if needed
      if (selectedIdentity) openIdentity(selectedIdentity);
    } catch (err) {
      console.error(err);
      setStatus('Update failed');
    }
  };

  const startEmbeddings = async () => {
    try {
      await fetch(`${API_BASE}/api/tasks/generate-embeddings`, { method: 'POST' });
      setStatus('Embedding generation started');
    } catch (err) {
      console.error(err);
    }
  };

  const detectErrors = async () => {
    setLoading(true);
    setStatus('Detecting potential errors...');
    try {
      const resp = await fetch(`${API_BASE}/api/tasks/detect-errors`, { method: 'POST' });
      const data = await resp.json();
      setErrors(data);
      setView('errors');
      setStatus(`Found ${data.length} potential errors`);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const runSplit = async () => {
    try {
      const resp = await fetch(`${API_BASE}/api/tasks/split-data`, { method: 'POST' });
      const data = await resp.json();
      setStatus(`Split complete: ${data.train_count} Train, ${data.test_count} Test`);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="sidebar-header">
          <Layers className="logo-icon" />
          <h1>SALAMANDER<span>RE-ID</span></h1>
        </div>

        <nav className="sidebar-nav">
          <button 
            className={`nav-item ${view === 'identities' || view === 'details' ? 'active' : ''}`}
            onClick={() => setView('identities')}
          >
            <Database size={20} />
            Identities
          </button>
          <button 
            className={`nav-item ${view === 'errors' ? 'active' : ''}`}
            onClick={detectErrors}
          >
            <AlertCircle size={20} />
            Potential Errors
          </button>
        </nav>

        <div className="sidebar-section">
          <p className="section-label">ACTIONS</p>
          <button className="action-btn" onClick={() => fileInputRef.current?.click()}>
            <FileUp size={18} />
            Load Metadata
          </button>
          <input 
            type="file" 
            ref={fileInputRef} 
            style={{ display: 'none' }} 
            onChange={handleLoadMetadata}
            accept=".csv,.xlsx"
          />
          
          <button className="action-btn" onClick={startEmbeddings}>
            <Play size={18} />
            Embeddings
          </button>
          
          <button className="action-btn" onClick={runSplit}>
            <Settings size={18} />
            Split Train/Test
          </button>
          
          <a href={`${API_BASE}/api/export`} className="action-btn export-link">
            <Download size={18} />
            Export Results
          </a>
        </div>

        <div className="sidebar-footer">
          <div className="status-indicator">
            <span className={`status-dot ${status.includes('failed') ? 'error' : 'success'}`}></span>
            {status}
          </div>
          {embeddingProgress > 0 && embeddingProgress < 100 && (
            <div className="progress-container">
              <div className="progress-bar" style={{ width: `${embeddingProgress}%` }}></div>
            </div>
          )}
        </div>
      </aside>

      <main className="main-content">
        <header className="content-header">
          <div className="header-left">
            {view === 'details' && (
              <button className="back-btn" onClick={() => setView('identities')}>
                <ChevronLeft size={20} />
              </button>
            )}
            <h2>
              {view === 'identities' && 'Identities Database'}
              {view === 'details' && `Identity: ${selectedIdentity}`}
              {view === 'validation' && 'Validation Mode'}
              {view === 'errors' && 'Potential Errors'}
            </h2>
          </div>
          
          <div className="header-right">
            {view === 'identities' && (
              <div className="search-box">
                <Search size={18} />
                <input 
                  type="text" 
                  placeholder="Search Identity ID..." 
                  value={searchText}
                  onChange={handleSearch}
                />
              </div>
            )}
          </div>
        </header>

        <div className="view-container">
          {loading ? (
            <div className="loading-overlay">
              <Loader2 className="spinner" />
              <p>Processing data...</p>
            </div>
          ) : (
            <>
              {view === 'identities' && (
                <div className="grid-container">
                  {identities.map((id) => (
                    <div className="card identity-card" key={id.id} onClick={() => openIdentity(id.id)}>
                      <div className="card-img-wrapper">
                        <img src={`${API_BASE}/api/images/serve/${id.representative_image}`} alt={id.id} loading="lazy" />
                      </div>
                      <div className="card-info">
                        <h3>{id.id}</h3>
                        <p>{id.count} images</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {view === 'details' && (
                <div className="grid-container">
                  {identityImages.map((img) => (
                    <div className="card img-card" key={img.index} onClick={() => openValidation(img.index)}>
                      <div className="card-img-wrapper">
                        <img src={`${API_BASE}/api/images/serve/${img.filename}`} alt={img.filename} loading="lazy" />
                      </div>
                      <div className="card-info">
                        <p className="filename">{img.filename}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {view === 'errors' && (
                <div className="error-list">
                  {errors.map((err) => (
                    <div className="error-item" key={err.index} onClick={() => openValidation(err.index)}>
                      <div className="error-img">
                        <img src={`${API_BASE}/api/images/serve/${err.filename}`} alt="" />
                      </div>
                      <div className="error-details">
                        <div className="error-meta">
                          <span className="error-tag">MISMATCH</span>
                          <h4>Image {err.index}</h4>
                        </div>
                        <p>ID: <strong>{err.id}</strong> — Match threshold exceeded with another ID.</p>
                      </div>
                      <ArrowRight size={20} />
                    </div>
                  ))}
                </div>
              )}

              {view === 'validation' && validationData && (
                <div className="validation-view">
                  <div className="validation-main">
                    <div className="query-section">
                      <h3>Query Image</h3>
                      <div className="query-card">
                        <img src={`${API_BASE}/api/images/serve/${validationData.query.filename}`} alt="Query" />
                        <div className="query-controls">
                          <div className="id-field">
                            <label>Current Identity ID</label>
                            <input 
                              type="text" 
                              value={newIdInput} 
                              onChange={(e) => setNewIdInput(e.target.value)} 
                            />
                          </div>
                          <button className="save-btn" onClick={updateId}>
                            <Save size={18} />
                            Save New ID
                          </button>
                        </div>
                      </div>
                    </div>

                    <div className="matches-section">
                      <h3>Closest Matches</h3>
                      <div className="matches-grid">
                        {validationData.neighbors.map((n) => (
                          <div className="match-card" key={n.index}>
                            <div className="match-img">
                              <img src={`${API_BASE}/api/images/serve/${n.filename}`} alt="" />
                            </div>
                            <div className="match-info">
                              <span className="match-id">{n.id}</span>
                              <span className="match-score">Score: {n.score.toFixed(4)}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
