'use client';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Leaf, LoaderCircle, RefreshCw } from 'lucide-react';
import { active } from '../lib/api';
import { useStudio } from '../lib/store';
import { navigation, titles } from '../lib/constants';
import { Overview } from './pages/Overview';
import { Generate } from './pages/Generate';
import { Gallery } from './pages/Gallery';
import { Settings } from './pages/Settings';

const SPOTLIGHT = '.stat, .feature-card, .gallery-card, .showcase-card';

// Feeds the cursor position to the hovered card so its border can light up under the pointer.
function spotlight(event: React.PointerEvent<HTMLElement>) {
  if (event.pointerType !== 'mouse') return;
  const card = (event.target as HTMLElement).closest<HTMLElement>(SPOTLIGHT);
  if (!card) return;
  const box = card.getBoundingClientRect();
  card.style.setProperty('--mx', `${event.clientX - box.left}px`);
  card.style.setProperty('--my', `${event.clientY - box.top}px`);
}

export default function Studio({ page }: { page: string }) {
  const health = useStudio(s => s.health);
  const jobs = useStudio(s => s.jobs);
  const error = useStudio(s => s.error);
  const refresh = useStudio(s => s.refresh);
  const [refreshing, setRefreshing] = useState(false);
  const activeJob = jobs.find(active);
  const running = !!activeJob;

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => void refresh(), 2500);
    return () => clearInterval(timer);
  }, [running, refresh]);

  // Cleanly route legacy links to studio
  const activePage = (page === 'dataset' || page === 'training') ? 'generate' : page;
  const pageLabel = navigation.find(n => n.id === activePage)?.label || 'Studio';
  const online = !!health && !error;

  const jobLabel = !activeJob ? ''
    : activeJob.kind === 'training' ? 'Training adapter'
      : activeJob.config?.mode === 'video' ? 'Rendering video' : 'Rendering image';
  const jobPercent = activeJob && activeJob.total > 0
    ? Math.min(100, Math.round((activeJob.current / activeJob.total) * 100))
    : null;

  async function manualRefresh() {
    setRefreshing(true);
    try { await refresh(); } finally { setRefreshing(false); }
  }

  return (
    <>
      <div className="ambient" aria-hidden="true">
        <i className="orb-a" /><i className="orb-b" /><i className="orb-c" />
      </div>

      <div className="shell">
        <aside className="sidebar">
          <Link href="/" className="brand" aria-label="NatureGen home">
            <span className="brand-icon"><Leaf size={19} /></span>
            naturegen
          </Link>
          <span className="brand-badge">Image &amp; Video</span>

          <div className="nav-label">WORKSPACE</div>
          <nav aria-label="Primary">
            {navigation.map(({ id, label, icon: Icon }) => (
              <Link
                key={id}
                href={`/${id}`}
                className={activePage === id ? 'selected' : ''}
                aria-current={activePage === id ? 'page' : undefined}
                title={label}
              >
                <Icon size={18} />
                {label}
                {activePage === id && <span className="nav-dot" />}
              </Link>
            ))}
          </nav>

          <div className="sidebar-bottom">
            <div className="model-card">
              <div className="status-indicator-row">
                <span className={`dot ${health?.generation_model ? 'green' : ''}`} />
                <span className="status-text">
                  {activeJob ? 'Working' : health?.generation_model ? 'System Ready' : 'Connecting…'}
                </span>
              </div>
              <strong className="status-headline">Photorealistic Engine</strong>
              {activeJob ? (
                <div className="sidebar-job" aria-live="polite">
                  <div>
                    <span>{jobLabel}</span>
                    <b>{jobPercent === null ? '…' : `${jobPercent}%`}</b>
                  </div>
                  <progress value={activeJob.current || 0} max={activeJob.total || 1} />
                </div>
              ) : (
                <p className="status-sub">High-fidelity nature generation &amp; fluid motion animation.</p>
              )}
            </div>
          </div>
        </aside>

        <div className="main-wrap">
          <header className="topbar">
            <div className="breadcrumbs">
              <span>NatureGen</span>
              <span className="breadcrumb-separator">/</span>
              <strong>{pageLabel}</strong>
            </div>
            <div className="topbar-actions">
              {activeJob && (
                <Link href="/generate" className="job-chip" title="Open the Studio to follow progress">
                  <LoaderCircle size={13} className="spin" />
                  <span>{jobLabel}</span>
                  {jobPercent !== null && <b>{jobPercent}%</b>}
                </Link>
              )}
              <div className="engine-status-pill">
                <span className={`dot ${online ? 'green' : ''}`} />
                <span>{online ? 'Online' : 'Syncing'}</span>
              </div>
              <button
                className="icon-button"
                aria-label="Refresh workspace"
                title="Refresh workspace status"
                disabled={refreshing}
                onClick={() => void manualRefresh()}
              >
                <RefreshCw size={14} className={refreshing ? 'spin' : ''} />
              </button>
            </div>
          </header>

          <main onPointerMove={spotlight}>
            <div className="page-heading" key={`heading-${activePage}`}>
              <div>
                <div className="eyebrow">{pageLabel}</div>
                <h1>{titles[activePage]?.[0] || 'Creative Studio'}</h1>
                <p>{titles[activePage]?.[1] || 'Create photorealistic nature imagery and cinematic videos.'}</p>
              </div>
            </div>

            {error && <div className="notice error" role="alert">{error}</div>}
            {health?.runtime_error && (
              <div className="notice error" role="alert">{health.runtime_error}</div>
            )}

            <div className="page" key={activePage}>
              {activePage === 'dashboard' && <Overview />}
              {activePage === 'generate' && <Generate />}
              {activePage === 'history' && <Gallery />}
              {activePage === 'settings' && <Settings />}
            </div>

            <footer>
              <span>NatureGen — Photorealistic Nature Imagery &amp; Cinematic Video Generation.</span>            </footer>
          </main>
        </div>
      </div>
    </>
  );
}
