"""Run NatureGen's FastAPI routes through Gradio's Space-managed server."""
import os
from contextlib import asynccontextmanager

os.environ.setdefault('DEVICE', 'cpu')
os.environ.setdefault('TORCH_NUM_THREADS', '2')

# ZeroGPU hardware refuses to start unless one function is decorated with @spaces.GPU.
# Generation stays on CPU; this hook only satisfies that startup check.
try:
    import spaces
except ImportError:  # Not on ZeroGPU hardware (local runs, CPU Spaces).
    spaces = None

if spaces is not None:
    @spaces.GPU(duration=1)
    def _zerogpu_compatibility_hook():
        return None

from gradio import Server

from app.main import app as api_app


@asynccontextmanager
async def lifespan(_server):
    async with api_app.router.lifespan_context(api_app):
        yield


app = Server(
    title=api_app.title,
    version=api_app.version,
    description='NatureGen image and video generation API.',
    lifespan=lifespan,
)

# Reuse the existing API endpoints, security middleware, and error handlers.
app.include_router(api_app.router)
for middleware in reversed(api_app.user_middleware):
    app.add_middleware(middleware.cls, *middleware.args, **middleware.kwargs)
for exception, handler in api_app.exception_handlers.items():
    app.add_exception_handler(exception, handler)


@app.get('/')
def root():
    return {'name': 'NatureGen API', 'health': '/api/health'}


if __name__ == '__main__':
    # No Node front-end: this Space only serves the API, so Python answers on 7860 directly.
    app.launch(server_name='0.0.0.0', server_port=7860, ssr_mode=False)
