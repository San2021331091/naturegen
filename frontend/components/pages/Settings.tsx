'use client';
import { Film, ShieldCheck, Sparkles } from 'lucide-react';
import { useStudio } from '../../lib/store';

export function Settings() {
  const health = useStudio(s => s.health);

  return (
    <div className="two-columns">
      <section className="panel">
        <div className="panel-heading">
          <Sparkles size={16} />
          <h2>Studio Engine</h2>
        </div>
        <dl>
          <dt>Image Synthesis</dt>
          <dd>High-Resolution Photorealistic Diffusion</dd>
          <dt>Video Animation</dt>
          <dd>Semantic Flow &amp; Regional Motion Engine</dd>
          <dt>Hardware Acceleration</dt>
          <dd className="highlight">
            {health?.generation_device?.toUpperCase() === 'CUDA' ? 'GPU Accelerated (NVIDIA CUDA)' : 'Hardware Optimized'}
          </dd>
          <dt>Default Image Output</dt>
          <dd>512 × 512 HD</dd>
          <dt>Default Video Duration</dt>
          <dd>60s Continuous Loop (480 Frames @ 8 FPS)</dd>
        </dl>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <Film size={16} />
          <h2>Export &amp; Media Specifications</h2>
        </div>
        <dl>
          <dt>Image Container</dt>
          <dd>Lossless PNG (sRGB Color Space)</dd>
          <dt>Video Container</dt>
          <dd>H.264 MP4 (Web &amp; Mobile Compatible)</dd>
          <dt>Audio Track</dt>
          <dd>Stereo Nature Soundscape (22.05 kHz)</dd>
          <dt>Storage Location</dt>
          <dd>Local Device Storage</dd>
        </dl>

        <div className="notice" style={{ marginTop: 20, marginBottom: 0 }}>
          <ShieldCheck size={16} style={{ display: 'inline', verticalAlign: '-3px', marginRight: 6 }} />
          <strong>100% Local &amp; Private.</strong> All creations are rendered on your machine and stored locally on your device.
        </div>
      </section>
    </div>
  );
}
