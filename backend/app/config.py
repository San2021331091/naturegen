import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
DATA = ROOT / 'data'
CHECKPOINTS = ROOT / 'checkpoints'
RESULTS = ROOT / 'results'
MODEL_CACHE = ROOT / 'models'
SD_LORA_DIR = CHECKPOINTS / 'tiny-sd-lora'
SD_MODEL_ID = os.getenv('NATUREGEN_SD_MODEL', 'segmind/tiny-sd').strip() or 'segmind/tiny-sd'
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '').strip()
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'openrouter/auto').strip() or 'openrouter/auto'
for directory in (DATA, CHECKPOINTS, RESULTS, MODEL_CACHE):
    directory.mkdir(parents=True, exist_ok=True)
# FRONTEND_ORIGIN may list several origins separated by commas (local dev plus a hosted frontend).
ORIGINS = tuple(dict.fromkeys(
    x.strip().rstrip('/') for x in os.getenv('FRONTEND_ORIGIN', 'http://localhost:3000').split(',') if x.strip()
)) or ('http://localhost:3000',)
ORIGIN = ORIGINS[0]
HOSTS = frozenset(x.strip().lower() for x in os.getenv(
    'ALLOWED_IMAGE_HOSTS', 'images.unsplash.com,images.pexels.com,upload.wikimedia.org').split(',') if x.strip())
