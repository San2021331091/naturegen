'use client';
import { useState, useEffect } from 'react';
import { BrainCircuit, Check, Gauge, Images, LoaderCircle, Play, SlidersHorizontal, Sparkles, Square } from 'lucide-react';
import { active, api, message } from '../../lib/api';
import { useStudio } from '../../lib/store';
import { Stat } from '../shared/Stat';

const TRAIN_STAGES = [
  { phase: 'loading-model', label: 'Load model', detail: 'Tiny-SD weights into memory' },
  { phase: 'filling image pool', label: 'Load photos', detail: 'Download and encode image-caption pairs' },
  { phase: 'training-lora', label: 'Train', detail: 'LoRA optimizer steps on the loaded pairs' },
];
const TRAIN_STATE_LABELS: Record<string, string> = {
  queued: 'Queued', running: 'Running', stopping: 'Stopping…',
  completed: 'Completed', failed: 'Failed', stopped: 'Stopped',
};
const PHOTO_SECONDS = 5.5;
const STEP_SECONDS = 7;
const paceFactor = (resolution?: number) => (resolution === 384 ? 0.5 : 1);

function duration(seconds: number) {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  const minutes = Math.round(seconds / 60);
  if (minutes < 1) return 'under a minute';
  return minutes < 60 ? `${minutes} min` : `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

export function Training() {
  const jobs = useStudio(s => s.jobs);
  const health = useStudio(s => s.health);
  const refresh = useStudio(s => s.refresh);
  const [pairs, setPairs] = useState(200);
  const [steps, setSteps] = useState(600);
  const [learningRate, setLearningRate] = useState(0.00001);
  const [seed, setSeed] = useState(42);
  const [resume, setResume] = useState(false);
  const [resolution, setResolution] = useState(512);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [mark, setMark] = useState<{ id: string; phase: string; current: number; at: number } | null>(null);

  const runs = jobs
    .filter(j => j.kind === 'training' && j.config?.target_model === 'tiny-sd')
    .sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
  const job = runs.find(active) || runs[0];
  const running = active(job);
  const blocked = jobs.some(active);
  const stageIndex = running ? TRAIN_STAGES.findIndex(s => s.phase === job.phase) : -1;
  const lora = health?.generation_lora;
  const shown = running
    ? { pairs: job.config?.buffer_size ?? pairs, steps: job.config?.steps ?? steps,
        learningRate: job.config?.learning_rate ?? learningRate, seed: job.config?.seed ?? seed,
        resolution: job.config?.resolution ?? 512 }
    : { pairs, steps, learningRate, seed, resolution };

  useEffect(() => {
    if (!job || !running) return;
    if (!mark || mark.id !== job.id || mark.phase !== (job.phase || '')) {
      setMark({ id: job.id, phase: job.phase || '', current: job.current || 0, at: job.updated_at || 0 });
    }
  }, [job, running, mark]);

  let remaining: number | null = null;
  if (running && mark && mark.id === job.id && mark.phase === job.phase && job.current > mark.current) {
    const perItem = ((job.updated_at || 0) - mark.at) / (job.current - mark.current);
    remaining = perItem * (job.total - job.current);
    if (job.phase === 'filling image pool') {
      const stepsLeft = (job.config?.steps || 0) - (job.config?.resume ? health?.generation_lora?.steps || 0 : 0);
      remaining += Math.max(0, stepsLeft) * STEP_SECONDS * paceFactor(job.config?.resolution);
    }
  }

  const canResume = !!lora?.available;
  const continuing = running ? !!job.config?.resume : canResume && resume;
  const resumeDone = !running && continuing && steps <= (lora?.steps || 0);
  const valid = !resumeDone && pairs >= 4 && pairs <= 1024 && steps >= 1 && steps <= 10000
    && learningRate >= 0.000001 && learningRate <= 0.001 && seed >= 0 && seed <= 4294967295;

  const loadPairs = continuing ? Math.min(pairs, lora?.images_seen || pairs) : pairs;
  const newSteps = continuing ? Math.max(0, steps - (lora?.steps || 0)) : steps;
  const photoSeconds = PHOTO_SECONDS * paceFactor(resolution);
  const stepSeconds = STEP_SECONDS * paceFactor(resolution);

  async function start() {
    setBusy(true);
    setError('');
    try {
      await api.post('/api/training/tiny-sd-lora', { steps, buffer_size: pairs, learning_rate: learningRate, seed, resume: canResume && resume, resolution });
      await refresh();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }

  async function stop() {
    setBusy(true);
    setError('');
    try {
      await api.post('/api/training/stop');
      await refresh();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }

  const finished = job?.state === 'completed';
  const stateLabel = !job ? 'No runs yet' : TRAIN_STATE_LABELS[job.state] || 'Interrupted';

  return (
    <div className="create-grid">
      <section className="panel">
        <div className="panel-heading">
          <SlidersHorizontal size={16} />
          <h3>New Training Run</h3>
        </div>
        <p className="muted" style={{ marginTop: 0 }}>
          Streams nature photos with their captions from Pexels and Unsplash and trains a small LoRA adapter on Tiny-SD.
          The Studio uses the newest adapter automatically.
        </p>

        <label className="train-resume">
          <input type="checkbox" checked={continuing} disabled={running || !canResume}
            onChange={e => setResume(e.target.checked)} />
          <span>
            Continue the saved adapter
            <small>
              {canResume
                ? `Keeps its ${lora.steps} steps and re-loads its saved pairs (${lora.images_seen} seen) instead of starting fresh.`
                : 'Available once an adapter has been saved.'}
            </small>
          </span>
        </label>

        <div className="form-grid">
          <label>
            {continuing ? 'Pairs to re-load (newest first)' : 'Image-caption pairs'}
            <input type="number" min={4} max={1024} value={shown.pairs} disabled={running}
              onChange={e => setPairs(+e.target.value)} />
          </label>
          <label>
            {continuing ? 'Total steps to reach' : 'Training steps'}
            <input type="number" min={1} max={10000} value={shown.steps} disabled={running}
              onChange={e => setSteps(+e.target.value)} />
          </label>
          <label>
            Learning rate
            <input type="number" min={0.000001} max={0.001} step={0.000001} value={shown.learningRate} disabled={running}
              onChange={e => setLearningRate(+e.target.value)} />
          </label>
          <label>
            Seed
            <input type="number" min={0} max={4294967295} value={shown.seed} disabled={running}
              onChange={e => setSeed(+e.target.value)} />
          </label>
          <label style={{ gridColumn: 'span 2' }}>
            Training image size
            <select value={shown.resolution} disabled={running} onChange={e => setResolution(+e.target.value)}>
              <option value={512}>512 × 512 — full detail</option>
              <option value={384}>384 × 384 — about twice as fast, slightly softer results</option>
            </select>
          </label>
        </div>
        <div className="field-hint">
          <span>4–1024 pairs · 1–10,000 steps. Raise steps with pairs so each pair is seen several times.</span>
        </div>

        <div className="notice" style={{ marginTop: 14 }}>
          {running
            ? 'These are the settings of the run in progress. Stop it to change them and start a new run.'
            : <>Estimated time on this machine: about {duration(loadPairs * photoSeconds + newSteps * stepSeconds)}{' '}
              ({duration(loadPairs * photoSeconds)} loading photos, {duration(newSteps * stepSeconds)} training{continuing ? ` ${newSteps} more steps` : ''}).</>}
          {' '}Image and video generation wait while a run is active.
        </div>

        {error && <div role="alert" className="notice error">{error}</div>}

        {running ? (
          <button className="button secondary full" style={{ marginTop: 18 }}
            disabled={busy || job.state === 'stopping'} onClick={() => void stop()}>
            {job.state === 'stopping' ? <LoaderCircle className="spin" size={16} /> : <Square size={15} />}
            {job.state === 'stopping' ? 'Stopping…' : 'Stop and Save Progress'}
          </button>
        ) : (
          <button className="button primary full" style={{ marginTop: 18 }}
            disabled={busy || blocked || !valid || !!health?.runtime_error} onClick={() => void start()}>
            {busy ? <LoaderCircle className="spin" size={16} /> : <Play size={15} />}
            {blocked ? 'Another job is running' : 'Start Training'}
          </button>
        )}
        {!valid && (
          <p className="error-text">
            {resumeDone
              ? `The saved adapter already has ${lora?.steps} steps. Set total steps higher to continue it.`
              : 'Check the values: one of them is outside its allowed range.'}
          </p>
        )}
      </section>

      <section className="panel">
        <div className="canvas-title" style={{ marginBottom: 12 }}>
          <span>TRAINING STAGE</span>
          <span className="pill">{stateLabel}</span>
        </div>

        <ol className="train-stages">
          {TRAIN_STAGES.map((stage, index) => {
            const done = running ? index < stageIndex : finished;
            const current = running && index === stageIndex;
            return (
              <li key={stage.phase} className={`train-stage ${done ? 'done' : ''} ${current ? 'current' : ''}`}>
                <span className="stage-mark">
                  {done ? <Check size={13} /> : current ? <LoaderCircle className="spin" size={13} /> : index + 1}
                </span>
                <div>
                  {stage.label}
                  <small>{stage.detail}</small>
                </div>
                {current && job.total > 0 && stage.phase !== 'loading-model' && (
                  <span className="stage-count">{job.current} / {job.total}</span>
                )}
              </li>
            );
          })}
        </ol>

        {running && (
          <div className="job-progress" aria-live="polite" style={{ marginTop: 0 }}>
            <div>
              <strong>{TRAIN_STAGES[stageIndex]?.label || 'Starting'}</strong>
              <span>{remaining === null ? 'Measuring pace…' : `About ${duration(remaining)} left`}</span>
            </div>
            <progress value={job.phase === 'loading-model' ? 0 : job.current || 0} max={job.total || 1} />
          </div>
        )}

        {job ? (
          <>
            <div className="train-metrics">
              <Stat icon={<Images size={16} />} label="Pairs Seen"
                value={job.images_seen ?? 0} detail={`Pool of ${job.config?.buffer_size ?? '—'}`} />
              <Stat icon={<Gauge size={16} />} label="Step"
                value={job.global_step ?? 0} detail={`of ${job.config?.steps ?? '—'}`} />
              <Stat icon={<Sparkles size={16} />} label="Loss"
                value={typeof job.loss === 'number' ? job.loss.toFixed(4) : '—'} detail="Latest step, lower is better" />
            </div>
            {job.error && <p className="error-text">{job.error}</p>}
          </>
        ) : (
          <p className="muted">No training run has been started from this backend yet.</p>
        )}

        <hr />
        <div className="panel-heading">
          <BrainCircuit size={16} />
          <h3>Adapter In Use</h3>
        </div>
        {lora?.available ? (
          <dl>
            <dt>Trained on</dt>
            <dd>{lora.images_seen} image-caption pairs</dd>
            <dt>Steps</dt>
            <dd>{lora.steps}</dd>
            <dt>Sources</dt>
            <dd>{lora.providers.join(', ') || '—'}</dd>
            <dt>Last saved</dt>
            <dd>{lora.trained_at ? new Date(lora.trained_at * 1000).toLocaleString() : '—'}</dd>
          </dl>
        ) : (
          <p className="muted">No adapter saved yet. The Studio is using the base Tiny-SD model.</p>
        )}
      </section>
    </div>
  );
}
