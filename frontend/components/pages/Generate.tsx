'use client';
import {
  ArrowDownToLine, Check, Clock, Copy, Film, History, Lightbulb, LoaderCircle,
  Shuffle, SlidersHorizontal, Sparkles, Video, Volume2, X,
} from 'lucide-react';
import { useEffect, useState } from 'react';
import { active, api, Generation, VideoGeneration, imageURL, Job, message } from '../../lib/api';
import { useStudio } from '../../lib/store';
import { STYLE_OPTIONS } from '../../lib/constants';
import { Empty } from '../shared/Empty';
import { JobProgress } from '../shared/JobProgress';

const RING_RADIUS = 58;
const RING_LENGTH = 2 * Math.PI * RING_RADIUS;
const fill = (value: number, min: number, max: number) =>
  ({ '--fill': `${((value - min) / (max - min)) * 100}%` }) as React.CSSProperties;

export function Generate() {
  const state = useStudio();
  const [busy, setBusy] = useState(false);
  const [prompting, setPrompting] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);
  const mode = state.mode;
  const setMode = (m: 'image' | 'video') => state.set({ mode: m });
  const [currentImage, setCurrentImage] = useState<Generation | null>(null);
  const [currentVideo, setCurrentVideo] = useState<VideoGeneration | null>(null);
  const [generatingJobId, setGeneratingJobId] = useState<string | null>(null);
  const [activeStyle, setActiveStyle] = useState<string | null>(null);
  const [promptIdeas, setPromptIdeas] = useState<string[]>([]);

  // Check URL query param ?mode=video or hash #video on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      const m = params.get('mode');
      if (m === 'video' || window.location.hash === '#video') {
        state.set({ mode: 'video' });
      } else if (m === 'image' || window.location.hash === '#image') {
        state.set({ mode: 'image' });
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const activeJob = state.jobs.find(j => j.kind === 'generation' && active(j));
  const job = activeJob || (generatingJobId ? state.jobs.find(j => j.id === generatingJobId) : undefined);

  // If a job is active on mount (e.g. reload while running), track it
  useEffect(() => {
    if (activeJob && !generatingJobId) {
      setGeneratingJobId(activeJob.id);
    }
  }, [activeJob, generatingJobId]);

  // When tracked job completes, show the new media
  useEffect(() => {
    if (!generatingJobId) return;
    const target = state.jobs.find(j => j.id === generatingJobId);
    if (target?.state === 'completed' && target.result && 'video_url' in target.result) {
      setCurrentImage(null);
      setCurrentVideo(target.result as VideoGeneration);
      setGeneratingJobId(null);
    } else if (target?.state === 'completed' && target.result && 'image_url' in target.result) {
      setCurrentVideo(null);
      setCurrentImage(target.result as Generation);
      setGeneratingJobId(null);
    } else if (target?.state === 'failed') {
      setError(target.error || 'Generation failed. Please try again.');
      setGeneratingJobId(null);
    } else if (target?.state === 'cancelled' || target?.state === 'stopped') {
      setError('Generation was cancelled.');
      setGeneratingJobId(null);
    }
  }, [state.jobs, generatingJobId]);

  async function generatePromptIdeas() {
    setPrompting(true);
    setError('');
    try {
      const response = await api.post<{ prompts: string[] }>('/api/prompts/generate', {
        mode,
        idea: "",
        avoid: [...promptIdeas, state.prompt].filter(Boolean).slice(-20),
        style: activeStyle || '',
        count: 4,
      }, { timeout: 45000 });
      const prompts = (response.data?.prompts || []).filter(prompt => typeof prompt === 'string' && prompt.length >= 3);
      if (!prompts.length) throw new Error('Could not retrieve prompt ideas. Please try again.');
      setPromptIdeas(prompts);
      state.set({ prompt: prompts[0].slice(0, 500) });
    } catch (e) {
      setError(message(e));
    } finally {
      setPrompting(false);
    }
  }

  function randomizeSeed() {
    state.set({ seed: Math.floor(Math.random() * 4294967295) });
  }

  async function generate() {
    setBusy(true);
    setError('');
    setCurrentImage(null);
    setCurrentVideo(null);
    try {
      const res = mode === 'video'
        ? await api.post<Job>('/api/generate-video', {
            prompt: state.prompt,
            seed: state.seed,
          })
        : await api.post<Job>('/api/generate', {
            prompt: state.prompt,
            seed: state.seed,
            steps: state.steps,
            guidance: state.guidance,
            resolution: 512,
          });
      if (res.data?.id) {
        setGeneratingJobId(res.data.id);
      }
      await state.refresh();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }

  async function copyPrompt(prompt: string) {
    try {
      await navigator.clipboard.writeText(prompt);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setError('Clipboard unavailable.');
    }
  }

  const working = active(job);
  const cannotGenerate = busy || working ||
    state.prompt.trim().length < 3 ||
    !state.health?.generation_model ||
    !!state.health?.runtime_error;

  const jobIsVideo = activeJob?.config?.mode === 'video';
  const jobTotal = activeJob ? activeJob.total || (jobIsVideo ? 20 : state.steps) : 0;
  const jobFraction = activeJob && activeJob.total > 0
    ? Math.min(1, (activeJob.current || 0) / activeJob.total)
    : null;

  return (
    <div className="create-grid">
      <section className="panel">
        {/* Mode Switcher */}
        <div className="media-mode-toggle" role="tablist" aria-label="Creation Mode" data-mode={mode}>
          <span className="mode-thumb" aria-hidden="true" />
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'image'}
            className={`media-mode-btn ${mode === 'image' ? 'active' : ''}`}
            onClick={() => { setMode('image'); setCurrentImage(null); setCurrentVideo(null); }}
            disabled={working}
          >
            <Sparkles size={15} /> Text to Image
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'video'}
            className={`media-mode-btn ${mode === 'video' ? 'active' : ''}`}
            onClick={() => { setMode('video'); setCurrentImage(null); setCurrentVideo(null); }}
            disabled={working}
          >
            <Video size={15} /> Text to Video
          </button>
        </div>

        {/* Prompt Input */}
        <div className="label-row">
          <label htmlFor="prompt">Scene Description</label>
        </div>
        <div className="prompt-box">
          <textarea
            id="prompt"
            maxLength={500}
            rows={5}
            placeholder={mode === 'video'
              ? 'Describe an organic landscape or water scene in motion (e.g. bioluminescent mountain stream cascading over granite rocks at dusk)...'
              : 'Describe your nature scene in detail: lighting, atmosphere, time of day, botanical flora, water reflections, or wildlife...'}
            value={state.prompt}
            onChange={e => state.set({ prompt: e.target.value })}
            onKeyDown={e => {
              if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !cannotGenerate) {
                e.preventDefault();
                void generate();
              }
            }}
          />
          <div className="prompt-box-footer">
            <button
              type="button"
              className="inspire-button"
              onClick={() => void generatePromptIdeas()}
              disabled={prompting || busy || working}
            >
              {prompting ? <LoaderCircle size={13} className="spin" /> : <Sparkles size={13} />}
              {prompting ? 'Generating ideas…' : 'Inspire Prompt'}
            </button>
            <span className={`prompt-count ${state.prompt.length >= 450 ? 'near' : ''}`}>
              {state.prompt.length}/500
            </span>
          </div>
        </div>
        <div className="field-hint">
          <span>Include atmospheric lighting, weather, and ecosystem details for best results.</span>
        </div>

        {/* Style Presets */}
        <div className="label-row" style={{ marginTop: 18 }}>
          <span className="label">Aesthetic Style</span>
        </div>
        <div className="style-presets-grid">
          {STYLE_OPTIONS.map(style => (
            <button
              key={style}
              type="button"
              className={`style-preset-chip ${activeStyle === style ? 'active' : ''}`}
              aria-pressed={activeStyle === style}
              onClick={() => setActiveStyle(activeStyle === style ? null : style)}
              disabled={prompting || working}
            >
              {activeStyle === style && <Check size={13} />}
              {style}
            </button>
          ))}
        </div>

        {promptIdeas.length > 0 && (
          <>
            <div className="label-row" style={{ marginTop: 18 }}>
              <span className="label">Inspiration Ideas</span>
            </div>
            <div className="prompt-chips">
              {promptIdeas.map((prompt, index) => (
                <button key={`${index}-${prompt}`} type="button" onClick={() => state.set({ prompt })}>
                  <Lightbulb size={14} />
                  <span>{prompt}</span>
                </button>
              ))}
            </div>
          </>
        )}

        <hr />

        {/* Settings */}
        <div className="panel-heading">
          <SlidersHorizontal size={15} />
          <h3>Studio Parameters</h3>
        </div>

        <div className="form-grid">
          {mode === 'image' ? (
            <>
              <div className="field">
                <div className="label-row">
                  <label htmlFor="steps">Sampling Steps</label>
                  <span className="value-readout">{state.steps}</span>
                </div>
                <input
                  id="steps"
                  type="range"
                  min={4}
                  max={50}
                  step={1}
                  value={state.steps}
                  style={fill(state.steps, 4, 50)}
                  onChange={e => state.set({ steps: +e.target.value })}
                />
                <div className="range-scale"><span>Faster</span><span>Finer detail</span></div>
              </div>
              <div className="field">
                <div className="label-row">
                  <label htmlFor="guidance">Prompt Fidelity</label>
                  <span className="value-readout">{state.guidance.toFixed(1)}</span>
                </div>
                <input
                  id="guidance"
                  type="range"
                  min={1}
                  max={12}
                  step={0.5}
                  value={state.guidance}
                  style={fill(state.guidance, 1, 12)}
                  onChange={e => state.set({ guidance: +e.target.value })}
                />
                <div className="range-scale"><span>Freer</span><span>Stricter</span></div>
              </div>
            </>
          ) : (
            <div className="field span-2">
              <div className="video-specs-pill">
                <span className="spec-chip"><Film size={14} /> 480 Frames @ 8 FPS</span>
                <span className="spec-chip"><Clock size={14} /> 60s Seamless Motion</span>
                <span className="spec-chip"><Volume2 size={14} /> Ambient Nature Sound</span>
              </div>
            </div>
          )}

          <div className="field span-2">
            <div className="label-row">
              <label htmlFor="seed">Random Seed</label>
            </div>
            <div className="input-affix">
              <input
                id="seed"
                type="number"
                min={0}
                max={4294967295}
                value={state.seed}
                onChange={e => state.set({ seed: +e.target.value })}
              />
              <button
                type="button"
                className="icon-button"
                aria-label="Randomize seed"
                title="Randomize seed"
                onClick={randomizeSeed}
              >
                <Shuffle size={15} />
              </button>
            </div>
          </div>
        </div>

        {error && <div role="alert" className="notice error">{error}</div>}

        <button
          className="button primary large full"
          style={{ marginTop: 24 }}
          disabled={cannotGenerate}
          onClick={() => void generate()}
        >
          {busy || working
            ? <LoaderCircle className="spin" size={17} />
            : mode === 'video' ? <Video size={17} /> : <Sparkles size={17} />}
          {working
            ? (mode === 'video' ? 'Generating Video Sequence…' : 'Generating High-Res Image…')
            : (mode === 'video' ? 'Generate Video Sequence' : 'Generate High-Res Image')}
          {!busy && !working && (
            <span className="kbd-group" aria-hidden="true"><kbd>Ctrl</kbd><kbd>↵</kbd></span>
          )}
        </button>

        <JobProgress job={job} />
        {working && (
          <div style={{ marginTop: 10 }}>
            <button
              className="button secondary full"
              onClick={async () => {
                if (job?.id) await api.post(`/api/jobs/${job.id}/cancel`);
                await state.refresh();
              }}
            >
              <X size={15} /> Cancel Generation
            </button>
          </div>
        )}
      </section>

      {/* Media Canvas Stage */}
      <section className="panel canvas-panel">
        <div className="canvas-title">
          <span>CANVAS VIEWPORT</span>
          <div>
            {(currentImage || currentVideo) && (
              <button
                type="button"
                className="text-button"
                onClick={() => { setCurrentImage(null); setCurrentVideo(null); }}
              >
                Clear
              </button>
            )}
            <span className="pill">
              {(currentVideo || mode === 'video') && <Film size={11} />}
              {currentVideo
                ? 'VIDEO · 480 FRAMES'
                : mode === 'video'
                  ? 'VIDEO OUTPUT'
                  : '512 × 512 HD'}
            </span>
          </div>
        </div>

        <div className={`canvas ${activeJob ? 'is-busy' : ''}`}>
          <span className="canvas-corner tl" /><span className="canvas-corner tr" />
          <span className="canvas-corner bl" /><span className="canvas-corner br" />

          {activeJob ? (
            <div className="canvas-generating" aria-live="polite">
              <div className={`progress-ring ${jobFraction === null ? 'indeterminate' : ''}`}>
                <svg viewBox="0 0 132 132" aria-hidden="true">
                  <defs>
                    <linearGradient id="ring-gradient" x1="0" y1="0" x2="1" y2="1">
                      <stop offset="0%" stopColor="#e3fba6" />
                      <stop offset="100%" stopColor="#6fe3cf" />
                    </linearGradient>
                  </defs>
                  <circle className="ring-track" cx="66" cy="66" r={RING_RADIUS} />
                  <circle
                    className="ring-value"
                    cx="66"
                    cy="66"
                    r={RING_RADIUS}
                    strokeDasharray={RING_LENGTH}
                    strokeDashoffset={RING_LENGTH * (1 - (jobFraction ?? 0.25))}
                  />
                </svg>
                {jobFraction !== null && (
                  <span>{Math.round(jobFraction * 100)}<small>%</small></span>
                )}
              </div>
              <div className="canvas-generating-badge">
                {jobIsVideo ? 'Rendering Video Sequence…' : 'Rendering Image…'}
              </div>
              <p className="canvas-generating-phase">
                {activeJob.phase?.replace(/-/g, ' ') || (jobIsVideo ? 'Animating natural flow vectors…' : 'Sampling diffusion model…')}
              </p>
              <span className="canvas-generating-steps">{activeJob.current || 0} / {jobTotal} steps</span>
            </div>
          ) : currentVideo ? (
            <video
              className="generated-video"
              src={imageURL(currentVideo.video_url)}
              controls
              autoPlay
              loop
              playsInline
            />
          ) : currentImage ? (
            <img
              className="generated-large"
              src={imageURL(currentImage.image_url)}
              alt={currentImage.prompt}
            />
          ) : (
            <Empty title="Canvas Ready">
              Enter a nature description on the left and start generation to render high-fidelity {mode === 'video' ? 'video clips' : 'nature imagery'}.
              {(mode === 'video' ? state.videoHistory.length : state.history.length) > 0 && (
                <div style={{ marginTop: 18 }}>
                  <button
                    type="button"
                    className="button secondary small"
                    onClick={() => {
                      if (mode === 'video' && state.videoHistory[0]) setCurrentVideo(state.videoHistory[0]);
                      else if (state.history[0]) setCurrentImage(state.history[0]);
                    }}
                  >
                    <History size={13} /> View Last {mode === 'video' ? 'Video' : 'Image'}
                  </button>
                </div>
              )}
            </Empty>
          )}
        </div>

        {currentVideo && (
          <div className="image-details">
            <p>{currentVideo.prompt}</p>
            <div className="meta-chips">
              <span>Seed {currentVideo.seed}</span>
              <span>{currentVideo.frames} frames</span>
              <span>{currentVideo.fps} FPS</span>
              <span>H.264 MP4</span>
            </div>
            <div className="button-row">
              <a className="button secondary" href={imageURL(currentVideo.video_url)} download>
                <ArrowDownToLine size={14} /> Download MP4
              </a>
              <button type="button" className="button secondary" onClick={() => void copyPrompt(currentVideo.prompt)}>
                {copied ? <Check size={14} /> : <Copy size={13} />} {copied ? 'Copied' : 'Copy Prompt'}
              </button>
            </div>
          </div>
        )}

        {currentImage && (
          <div className="image-details">
            <p>{currentImage.prompt}</p>
            <div className="meta-chips">
              <span>{currentImage.resolution} × {currentImage.resolution} px</span>
              <span>Seed {currentImage.seed}</span>
              <span>{currentImage.steps} steps</span>
            </div>
            <div className="button-row">
              <a
                className="button secondary"
                href={imageURL(currentImage.image_url)}
                target="_blank"
                rel="noreferrer"
                download
              >
                <ArrowDownToLine size={14} /> Download Image
              </a>
              <button type="button" className="button secondary" onClick={() => void copyPrompt(currentImage.prompt)}>
                {copied ? <Check size={14} /> : <Copy size={13} />} {copied ? 'Copied' : 'Copy Prompt'}
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
