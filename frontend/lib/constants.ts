import { Compass, Sparkles, Film, Settings2 } from 'lucide-react';

export const navigation = [
  { id: 'dashboard', label: 'Explore', icon: Compass },
  { id: 'generate', label: 'Studio', icon: Sparkles },
  { id: 'history', label: 'Creations', icon: Film },
  { id: 'settings', label: 'Settings', icon: Settings2 },
];

export const titles: Record<string, [string, string]> = {
  dashboard: ['Creative Showcase', 'Explore curated nature visuals and start your next high-fidelity creation.'],
  generate: ['Creative Studio', 'Craft photorealistic nature images and fluid cinematic video sequences.'],
  history: ['Your Creations', 'Browse, inspect, and export your generated imagery and video clips.'],
  settings: ['Studio Settings', 'Manage generation preferences, hardware acceleration, and output profiles.'],
};

export const STYLE_OPTIONS = [
  'Photorealistic', 'Cinematic', 'Aerial Drone',
  'Nature Documentary', 'Macro Wildlife', 'Moody Mist',
];
