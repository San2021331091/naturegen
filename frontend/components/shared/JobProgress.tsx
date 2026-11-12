'use client';
import { Job } from '../../lib/api';

export function JobProgress({ job }: { job?: Job }) {
  if (!job) return null;
  const isVideo = job.config?.mode === 'video';
  const loadingModel = ['loading-model', 'loading-video-model'].includes(job.phase || '');
  const phaseLabel = loadingModel
    ? 'Initializing AI model'
    : job.phase === 'generating-keyframe'
      ? 'Generating image frame'
      : job.phase === 'segmenting-motion'
        ? 'Finding water and foliage'
        : job.phase === 'animating-regions'
          ? 'Animating selected regions'
          : job.phase === 'transitioning-images'
            ? 'Creating image transitions'
            : job.phase === 'adding-audio'
              ? 'Adding selected sounds'
              : job.phase === 'generating-video'
                ? 'Rendering video'
                : job.phase || 'Processing';

  const progressLabel = loadingModel
    ? 'Loading model into memory...'
    : isVideo
      ? `Progress ${(job.current || 0)} / ${(job.total || 20)} · Generating Video`
      : `Step ${(job.current || 0)} / ${(job.total || 20)} · Generating Image`;

  return (
    <div className="job-progress" aria-live="polite">
      <div>
        <strong>{phaseLabel}</strong>
        <span>{progressLabel}</span>
      </div>
      <progress value={job.current || 0} max={job.total || 1} />
      {job.error && <p className="error-text">{job.error}</p>}
    </div>
  );
}
