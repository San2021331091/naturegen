'use client';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { ArrowRight, Compass, Film, Image as ImageIcon, ShieldCheck, Sparkles, Video } from 'lucide-react';
import { useStudio } from '../../lib/store';
import { Stat } from '../shared/Stat';
import { Gallery } from './Gallery';

const SHOWCASE_PROMPTS = [
  {
    title: 'Bioluminescent Alpine Cascade',
    prompt: 'Bioluminescent mountain waterfall at twilight with floating glowing spores, lush mossy rocks, cinematic volumetric mist, 35mm photography',
    tag: 'Cinematic · 512 × 512',
    style: 'Cinematic',
    scene: 'scene-cascade',
  },
  {
    title: 'Emerald Forest Sunbeams',
    prompt: 'Ancient redwood forest canopy illuminated by morning god rays, dew-covered ferns, realistic volumetric lighting, National Geographic style',
    tag: 'Photorealistic · 512 × 512',
    style: 'Photorealistic',
    scene: 'scene-forest',
  },
  {
    title: 'Volcanic Obsidian Coastline',
    prompt: 'Black sand volcanic beach in Iceland at dusk, dramatic crashing seafoam waves, basalt columns, mood mist, aerial drone perspective',
    tag: 'Aerial Drone · 512 × 512',
    style: 'Aerial Drone',
    scene: 'scene-coast',
  },
];

const ROTATE_MS = 6000;

export function Overview() {
  const history = useStudio(s => s.history);
  const videoHistory = useStudio(s => s.videoHistory);
  const set = useStudio(s => s.set);
  const [slide, setSlide] = useState(0);
  const [paused, setPaused] = useState(false);
  const [autoplay, setAutoplay] = useState(false);

  // Rotate the prompt ideas, unless the visitor prefers reduced motion.
  useEffect(() => {
    setAutoplay(!window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  }, []);
  const playing = autoplay && !paused;
  useEffect(() => {
    if (!playing) return;
    const timer = setTimeout(() => setSlide(s => (s + 1) % SHOWCASE_PROMPTS.length), ROTATE_MS);
    return () => clearTimeout(timer);
  }, [playing, slide]);

  const featured = SHOWCASE_PROMPTS[slide];

  return (
    <>
      <section className="hero">
        <div className="hero-copy">
          <div className="hero-badge"><Sparkles size={13} /> Photorealistic Nature Synthesis</div>
          <h2>Breathtaking Nature Visuals &amp; <em>Living Motion.</em></h2>
          <p>
            Transform descriptive text into photorealistic nature imagery and fluid cinematic video sequences in seconds.
          </p>
          <div className="hero-actions">
            <Link
              href="/generate"
              className="button primary large"
              onClick={() => set({ mode: 'image' })}
            >
              <Sparkles size={16} /> Create Image
            </Link>
            <Link
              href="/generate?mode=video"
              className="button secondary large"
              onClick={() => set({ mode: 'video' })}
            >
              <Video size={16} /> Generate Video
            </Link>
          </div>
          <span className="hero-caption">
            <ShieldCheck size={14} /> High-fidelity 512 × 512 images &amp; seamless MP4 video loops.
          </span>
        </div>

        <div className="hero-showcase">
          <div
            className="showcase-card"
            onMouseEnter={() => setPaused(true)}
            onMouseLeave={() => setPaused(false)}
            onFocus={() => setPaused(true)}
            onBlur={() => setPaused(false)}
          >
            <div className="showcase-art">
              {SHOWCASE_PROMPTS.map((item, index) => (
                <i key={item.scene} className={`${item.scene} ${index === slide ? 'on' : ''}`} aria-hidden="true" />
              ))}
              <div className="showcase-art-tags">
                <span className="showcase-pill">{featured.tag}</span>
                <span className="showcase-badge">Featured Style</span>
              </div>
              <h3>{featured.title}</h3>
            </div>

            <div className="showcase-content">
              <p className="showcase-prompt">“{featured.prompt}”</p>
              <div className="showcase-footer">
                <div className="showcase-actions">
                  <Link
                    href="/generate"
                    className="button light small"
                    onClick={() => set({ mode: 'image', prompt: featured.prompt })}
                  >
                    <Sparkles size={13} /> Try in Studio
                  </Link>
                  <Link
                    href="/generate?mode=video"
                    className="button secondary small"
                    onClick={() => set({ mode: 'video', prompt: featured.prompt })}
                  >
                    <Video size={13} /> Animate to Video
                  </Link>
                </div>
                <div className={`showcase-dots ${playing ? 'playing' : ''}`}>
                  {SHOWCASE_PROMPTS.map((item, index) => (
                    <button
                      key={`${item.scene}-${index === slide ? slide : 'idle'}`}
                      type="button"
                      className={index === slide ? 'on' : ''}
                      aria-label={`Show prompt idea: ${item.title}`}
                      aria-current={index === slide}
                      onClick={() => setSlide(index)}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="stats-grid">
        <Stat
          icon={<ImageIcon size={17} />}
          label="Images Generated"
          value={history.length}
          detail="High-resolution nature photos"
        />
        <Stat
          icon={<Video size={17} />}
          label="Videos Created"
          value={videoHistory.length}
          detail="Fluid natural motion sequences"
        />
        <Stat
          icon={<Sparkles size={17} />}
          label="Image Studio"
          value="Active"
          detail="512 × 512 High Definition"
        />
        <Stat
          icon={<Film size={17} />}
          label="Video Motion"
          value="Active"
          detail="Smooth 60s ambient loops"
        />
      </div>

      <div className="section-title">
        <div>
          <h2>Creative Capabilities</h2>
          <p>Everything you need to produce stunning natural imagery and video assets.</p>
        </div>
      </div>

      <div className="features-grid">
        <article className="feature-card">
          <div className="feature-card-top">
            <div className="feature-card-icon"><ImageIcon size={21} /></div>
            <span>01</span>
          </div>
          <h3>Photorealistic Imagery</h3>
          <p>Render true-to-life flora, water reflections, atmospheric mist, and geological formations with natural lighting and depth of field.</p>
        </article>
        <article className="feature-card">
          <div className="feature-card-top">
            <div className="feature-card-icon"><Video size={21} /></div>
            <span>02</span>
          </div>
          <h3>Fluid Nature Motion</h3>
          <p>Bring still nature compositions to life with organic, seamless movement across water surfaces, drifting clouds, and swaying canopies.</p>
        </article>
        <article className="feature-card">
          <div className="feature-card-top">
            <div className="feature-card-icon"><Compass size={21} /></div>
            <span>03</span>
          </div>
          <h3>Prompt Inspiration</h3>
          <p>Instantly expand initial concepts into rich, atmospheric nature descriptions crafted specifically for photorealistic visual composition.</p>
        </article>
      </div>

      <div className="section-title">
        <div>
          <h2>Recent Creations</h2>
          <p>Review your recent generated images and motion clips.</p>
        </div>
        <Link href="/history">View all creations <ArrowRight size={14} /></Link>
      </div>
      <div>
        <Gallery compact />
      </div>
    </>
  );
}
