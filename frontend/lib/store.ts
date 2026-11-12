import { create } from 'zustand';
import { api, Health, Stats, Job, Generation, VideoGeneration, message } from './api';

interface State {
  health: Health | null; stats: Stats | null; jobs: Job[]; history: Generation[]; videoHistory: VideoGeneration[];
  error: string; prompt: string; seed: number; steps: number; guidance: number; sampler: string;
  mode: 'image' | 'video';
  set: (values: Partial<State>) => void;
  refresh: () => Promise<void>;
}
export const useStudio = create<State>((set) => ({
  health: null, stats: null, jobs: [], history: [], videoHistory: [], error: '',
  prompt: '', seed: 42, steps: 20, guidance: 7.5, sampler: 'default',
  mode: 'image',
  set: values => set(values),
  refresh: async () => {
    try {
      const [health, stats, jobs, history, videoHistory] = await Promise.all([
        api.get<Health>('/api/health'), api.get<Stats>('/api/dataset/stats'),
        api.get<Job[]>('/api/jobs'), api.get<Generation[]>('/api/history'),
        api.get<VideoGeneration[]>('/api/video-history'),
      ]);
      set({ health: health.data, stats: stats.data, jobs: jobs.data, history: history.data, videoHistory: videoHistory.data, error: '' });
    } catch (error) { set({ error: message(error) }); }
  },
}));
