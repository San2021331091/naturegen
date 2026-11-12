#!/bin/sh
set -eu

cd /home/user/app

# Hugging Face injects SPACE_HOST for Spaces. Permit only this public app origin.
if [ -n "${SPACE_HOST:-}" ]; then
  export FRONTEND_ORIGIN="https://${SPACE_HOST}"
else
  export FRONTEND_ORIGIN="http://localhost:7860"
fi

python -m uvicorn app.main:app --app-dir /home/user/app/backend --host 127.0.0.1 --port 8000 &
backend_pid=$!
cd /home/user/app/frontend
node server.js &
frontend_pid=$!

shutdown() {
  kill "$backend_pid" "$frontend_pid" 2>/dev/null || true
}
trap shutdown INT TERM EXIT

while kill -0 "$backend_pid" 2>/dev/null && kill -0 "$frontend_pid" 2>/dev/null; do
  sleep 2
done
shutdown
wait "$backend_pid" 2>/dev/null || true
wait "$frontend_pid" 2>/dev/null || true
