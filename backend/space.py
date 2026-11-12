"""Entry point for a Hugging Face Gradio Space: serves the NatureGen API on port 7860.

Local development does not use this file; run uvicorn on app.main:app as before.
"""
import os

os.environ.setdefault('DEVICE', 'cpu')
os.environ.setdefault('TORCH_NUM_THREADS', '2')

import gradio as gr
import uvicorn

from app.main import app

with gr.Blocks(title='NatureGen API') as page:
    gr.Markdown(
        '# NatureGen API\n'
        'This Space is the engine behind the NatureGen studio; the studio itself is hosted separately.\n\n'
        'Health check: `/api/health` · API docs: `/docs`'
    )

# The API routes are registered first, so they win; the info page answers everything else.
app = gr.mount_gradio_app(app, page, path='/')

if __name__ == '__main__':
    # One worker only: the job queue lives in this process.
    uvicorn.run(app, host='0.0.0.0', port=7860)
