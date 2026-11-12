const DEFAULT_SPACE_URL = 'https://sanloxai-springspace.hf.space';
const REQUEST_TIMEOUT_MS = 120_000;

type RouteContext = { params: Promise<{ path: string[] }> };

function isPublicDemoRequest(method: string, path: string[]): boolean {
  const key = path.join('/');
  if (method === 'GET') {
    return ['health', 'settings', 'dataset/stats', 'jobs', 'history', 'video-history'].includes(key)
      || (path.length === 2 && path[0] === 'jobs' && /^[a-f0-9]{32}$/.test(path[1]))
      || (path.length === 2 && path[0] === 'generated' && /^[a-f0-9]{32}\.png$/.test(path[1]))
      || (path.length === 2 && path[0] === 'videos' && /^[a-f0-9]{32}\.mp4$/.test(path[1]));
  }
  if (method === 'POST') {
    return ['prompts/generate', 'generate', 'generate-video'].includes(key)
      || (path.length === 3 && path[0] === 'jobs' && /^[a-f0-9]{32}$/.test(path[1]) && path[2] === 'cancel');
  }
  if (method === 'DELETE') {
    // The gallery's delete button.
    return path.length === 2 && ['history', 'video-history'].includes(path[0]) && /^[a-f0-9]{32}$/.test(path[1]);
  }
  return false;
}

async function proxy(request: Request, context: RouteContext): Promise<Response> {
  const { path } = await context.params;
  if (!isPublicDemoRequest(request.method, path)) {
    return Response.json({ detail: 'This endpoint is not available in the public demo.' }, { status: 403 });
  }

  const token = process.env.HF_TOKEN;
  if (!token) {
    return Response.json({ detail: 'The HF_TOKEN server environment variable is not configured on Vercel.' }, { status: 503 });
  }

  const base = (process.env.HF_SPACE_URL || DEFAULT_SPACE_URL).replace(/\/$/, '');
  const target = `${base}/api/${path.map(encodeURIComponent).join('/')}${new URL(request.url).search}`;
  const headers = new Headers();
  for (const name of ['accept', 'content-type', 'range']) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  headers.set('authorization', `Bearer ${token}`);

  try {
    const upstream = await fetch(target, {
      method: request.method,
      headers,
      body: request.method === 'GET' || request.method === 'HEAD' ? undefined : await request.arrayBuffer(),
      cache: 'no-store',
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });

    const responseHeaders = new Headers();
    for (const name of ['content-type', 'content-length', 'content-disposition', 'cache-control', 'accept-ranges', 'content-range']) {
      const value = upstream.headers.get(name);
      if (value) responseHeaders.set(name, value);
    }
    responseHeaders.set('cache-control', 'no-store');
    return new Response(upstream.body, {
      status: upstream.status,
      headers: responseHeaders,
    });
  } catch (error) {
    const detail = error instanceof Error && error.name === 'TimeoutError'
      ? 'The Hugging Face backend did not respond in time. Check whether the Space is awake.'
      : 'Could not reach the Hugging Face backend. Check the Space URL and deployment status.';
    return Response.json({ detail }, { status: 502 });
  }
}

export const GET = proxy;
export const HEAD = proxy;
export const POST = proxy;
export const DELETE = proxy;
