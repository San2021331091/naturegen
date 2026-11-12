'use client';
import { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { ArrowDownToLine, ChevronLeft, ChevronRight, X } from 'lucide-react';
import { Generation, VideoGeneration, imageURL } from '../../lib/api';

export type MediaItem =
  | (Generation & { mediaType: 'image' })
  | (VideoGeneration & { mediaType: 'video' });

export function Lightbox({ items, index, onIndex, onClose }: {
  items: MediaItem[];
  index: number;
  onIndex: (index: number) => void;
  onClose: () => void;
}) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const item = items[index];
  const last = items.length - 1;

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    closeRef.current?.focus();
    return () => {
      document.body.style.overflow = overflow;
      previous?.focus?.();
    };
  }, []);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') onClose();
      else if (event.key === 'ArrowLeft' && index > 0) onIndex(index - 1);
      else if (event.key === 'ArrowRight' && index < last) onIndex(index + 1);
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [index, last, onIndex, onClose]);

  // The item can disappear underneath an open viewer (deleted, list refreshed).
  useEffect(() => { if (!item) onClose(); }, [item, onClose]);
  if (!item) return null;

  const isVideo = item.mediaType === 'video';
  const url = imageURL(isVideo ? item.video_url : item.image_url);

  return createPortal(
    <div
      className="lightbox"
      role="dialog"
      aria-modal="true"
      aria-label="Creation viewer"
      onClick={event => { if (event.target === event.currentTarget) onClose(); }}
    >
      <span className="lightbox-counter">{index + 1} / {items.length}</span>
      <button ref={closeRef} type="button" className="lightbox-close" aria-label="Close viewer" onClick={onClose}>
        <X size={18} />
      </button>

      <button
        type="button"
        className="lightbox-nav prev"
        aria-label="Previous creation"
        disabled={index === 0}
        onClick={() => onIndex(index - 1)}
      >
        <ChevronLeft size={20} />
      </button>

      <div className="lightbox-stage" key={`${item.mediaType}-${item.id}`}>
        {isVideo
          ? <video src={url} controls autoPlay loop playsInline aria-label={item.prompt} />
          : <img src={url} alt={item.prompt} />}
        <div className="lightbox-caption">
          <p>{item.prompt}</p>
          <div className="meta-chips">
            <span>{new Date(item.created_at * 1000).toLocaleDateString()}</span>
            {isVideo
              ? <><span>{item.frames} frames</span><span>{item.fps} fps</span></>
              : <><span>{item.resolution} × {item.resolution}</span><span>{item.steps} steps</span></>}
            <span>Seed {item.seed}</span>
          </div>
          <a className="button secondary small" style={{ marginTop: 14 }} href={url} download>
            <ArrowDownToLine size={14} /> Download {isVideo ? 'MP4' : 'image'}
          </a>
        </div>
      </div>

      <button
        type="button"
        className="lightbox-nav next"
        aria-label="Next creation"
        disabled={index === last}
        onClick={() => onIndex(index + 1)}
      >
        <ChevronRight size={20} />
      </button>
    </div>,
    document.body,
  );
}
