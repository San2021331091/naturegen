'use client';
import Link from 'next/link';
import { ArrowDownToLine, Image as ImageIcon, Maximize2, RefreshCw, Sparkles, Trash2, Video } from 'lucide-react';
import { useCallback, useState } from 'react';
import { api, imageURL, message } from '../../lib/api';
import { useStudio } from '../../lib/store';
import { Empty } from '../shared/Empty';
import { Lightbox, MediaItem } from '../shared/Lightbox';

export function Gallery({ compact = false }: { compact?: boolean }) {
  const history = useStudio(s => s.history);
  const videoHistory = useStudio(s => s.videoHistory);
  const refresh = useStudio(s => s.refresh);
  const set = useStudio(s => s.set);
  const [filter, setFilter] = useState<'all' | 'images' | 'videos'>('all');
  const [error, setError] = useState('');
  const [viewing, setViewing] = useState<number | null>(null);
  const closeViewer = useCallback(() => setViewing(null), []);

  const allItems: MediaItem[] = [
    ...history.map(item => ({ ...item, mediaType: 'image' as const })),
    ...videoHistory.map(item => ({ ...item, mediaType: 'video' as const })),
  ].sort((a, b) => b.created_at - a.created_at);

  const filteredItems = filter === 'images'
    ? allItems.filter(i => i.mediaType === 'image')
    : filter === 'videos'
      ? allItems.filter(i => i.mediaType === 'video')
      : allItems;

  const displayItems = compact ? allItems.slice(0, 3) : filteredItems;

  if (!allItems.length) {
    return (
      <div className="panel">
        <Empty title="No Creations Yet">
          Use the Studio to generate your first photorealistic nature image or cinematic video.
          <div style={{ marginTop: 18 }}>
            <Link href="/generate" className="button primary">
              <Sparkles size={14} /> Open Studio
            </Link>
          </div>
        </Empty>
      </div>
    );
  }

  return (
    <>
      {error && <p className="notice error" role="alert">{error}</p>}

      {!compact && (
        <div>
          <div className="gallery-filter-bar" role="group" aria-label="Filter creations">
            <button
              type="button"
              className={`filter-pill ${filter === 'all' ? 'active' : ''}`}
              aria-pressed={filter === 'all'}
              onClick={() => setFilter('all')}
            >
              All Creations <span className="filter-count">{allItems.length}</span>
            </button>
            <button
              type="button"
              className={`filter-pill ${filter === 'images' ? 'active' : ''}`}
              aria-pressed={filter === 'images'}
              onClick={() => setFilter('images')}
            >
              <ImageIcon size={14} /> Images Only <span className="filter-count">{history.length}</span>
            </button>
            <button
              type="button"
              className={`filter-pill ${filter === 'videos' ? 'active' : ''}`}
              aria-pressed={filter === 'videos'}
              onClick={() => setFilter('videos')}
            >
              <Video size={14} /> Videos Only <span className="filter-count">{videoHistory.length}</span>
            </button>
          </div>
        </div>
      )}

      {!displayItems.length && (
        <div className="panel">
          <Empty title={filter === 'videos' ? 'No Videos Yet' : 'No Images Yet'}>
            Nothing matches this filter. Switch back to all creations or make something new in the Studio.
          </Empty>
        </div>
      )}

      <div className="gallery">
        {displayItems.map((item, index) => {
          const isVideo = item.mediaType === 'video';
          const mediaUrl = imageURL(isVideo ? item.video_url : item.image_url);

          return (
            <article className="gallery-card" key={`${item.mediaType}-${item.id}`}>
              <div className="gallery-media">
                <span className={`media-card-badge ${isVideo ? 'video' : 'image'}`}>
                  {isVideo ? <><Video size={11} /> Video</> : <><ImageIcon size={11} /> Image</>}
                </span>

                {isVideo ? (
                  <>
                    <video
                      src={mediaUrl}
                      controls
                      preload="metadata"
                      playsInline
                      aria-label={item.prompt}
                    />
                    <button
                      type="button"
                      className="gallery-expand"
                      aria-label="View larger"
                      onClick={() => setViewing(index)}
                    >
                      <Maximize2 size={14} />
                    </button>
                  </>
                ) : (
                  <button
                    type="button"
                    className="gallery-open"
                    aria-label="View larger"
                    onClick={() => setViewing(index)}
                  >
                    <img src={mediaUrl} alt={item.prompt} loading="lazy" />
                    <span className="gallery-expand" aria-hidden="true"><Maximize2 size={14} /></span>
                  </button>
                )}
              </div>

              <div className="gallery-body">
                <p title={item.prompt}>{item.prompt}</p>
                <small>
                  {new Date(item.created_at * 1000).toLocaleDateString()} · {isVideo
                    ? `${item.frames} frames · ${item.fps} fps`
                    : `${item.resolution} × ${item.resolution} · Seed ${item.seed}`}
                </small>

                <div className="button-row">
                  <a className="icon-button" aria-label="Download" title="Download" href={mediaUrl} download>
                    <ArrowDownToLine size={16} />
                  </a>
                  <Link
                    className="icon-button"
                    aria-label="Reuse in Studio"
                    title="Reuse in Studio"
                    href={isVideo ? "/generate?mode=video" : "/generate"}
                    onClick={() => {
                      set({
                        mode: isVideo ? 'video' : 'image',
                        prompt: item.prompt,
                        seed: item.seed,
                        steps: isVideo ? 20 : Math.min(50, Math.max(4, item.steps || 20)),
                        guidance: isVideo ? 7.5 : (item.guidance || 7.5),
                      });
                    }}
                  >
                    <RefreshCw size={15} />
                  </Link>
                  <button
                    type="button"
                    className="icon-button danger"
                    aria-label="Delete"
                    title="Delete"
                    onClick={async () => {
                      try {
                        await api.delete(`/api/${isVideo ? 'video-history' : 'history'}/${item.id}`);
                        await refresh();
                      } catch (e) {
                        setError(message(e));
                      }
                    }}
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
            </article>
          );
        })}
      </div>

      {viewing !== null && (
        <Lightbox items={displayItems} index={viewing} onIndex={setViewing} onClose={closeViewer} />
      )}
    </>
  );
}
