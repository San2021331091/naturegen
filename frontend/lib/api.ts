import axios from 'axios';

export const baseURL = (process.env.NEXT_PUBLIC_API_URL ?? (process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000')).replace(/\/$/, '');
export const api = axios.create({ baseURL, timeout: 30000 });
export const imageURL = (path: string) => `${baseURL}${path}`;
export function message(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) return detail.map((d: { msg: string }) => d.msg).join('; ');
    return error.response ? `Request failed (${error.response.status}).` : (process.env.NODE_ENV === 'production' ? 'Cannot reach the studio engine. Please try again in a moment.' : 'Cannot reach the backend. Start FastAPI on port 8000.');
  }
  return error instanceof Error ? error.message : 'Something went wrong.';
}
export interface Health { status: string; checkpoint_available: boolean; device: string; generation_device: string; torch_version: string | null; runtime_error: string | null; generation_model: string; generation_model_loaded: boolean; video_model: string; video_model_loaded: boolean; generation_lora?: { available: boolean; steps: number; images_seen: number; providers: string[]; trained_at: number | null }; }
export interface Generation { id: string; prompt: string; image_url: string; created_at: number; seed: number; steps: number; resolution: number; guidance: number; sampler: string; }
export interface VideoGeneration { id: string; prompt: string; video_url: string; created_at: number; seed: number; steps: number; frames: number; fps: number; width: number; height: number; model_id: string; audio?: { description: string; audio_style?: string; transition_effect_count?: number }; transition?: string | null; source_image_count?: number; }
export interface Job { id: string; kind: string; state: string; current: number; total: number; created_at?: number; updated_at?: number; config?: { mode?: string; steps?: number; buffer_size?: number; learning_rate?: number; seed?: number; resume?: boolean; resolution?: number; target_model?: string; method?: string }; error?: string; loss?: number; validation_loss?: number; epoch?: number; global_step?: number; images_seen?: number; phase?: string; result?: Generation | VideoGeneration; }
export interface Stats { total: number; available: number; invalid: number; approved: number; categories: Record<string, number>; sources: Record<string, number>; splits: Record<string, number>; training_readiness: { ready: boolean; train_pairs: number; validation_pairs: number; message: string | null }; }
export interface Pair { id: string; caption: string; image_url: string; source: string; creator: string; attribution_url: string; license_note: string; status: string; review_status: string; split: string | null; match_signals?: { coverage?: number }; }
export interface Metric { id: string; run_id: string; epoch: number; loss: number | null; validation_loss: number; }
export const active = (job?: Job): job is Job => !!job && ['queued', 'running', 'stopping'].includes(job.state);
