FROM node:22-bookworm-slim AS frontend-build
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
ENV NEXT_PUBLIC_API_URL=
RUN npm run build

FROM python:3.12-slim AS backend-build
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="/opt/venv/bin:$PATH" PIP_NO_CACHE_DIR=1
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install 'torch>=2.8,<3' --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r /tmp/requirements.txt

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin" \
    DEVICE=cpu TORCH_NUM_THREADS=2 HF_HOME=/home/user/.cache/huggingface \
    HOSTNAME=0.0.0.0 PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 user
COPY --from=backend-build /opt/venv /opt/venv
COPY --from=frontend-build /usr/local/bin/node /usr/local/bin/node
WORKDIR /home/user/app
COPY --chown=user:user backend/app ./backend/app
COPY --from=frontend-build --chown=1000:1000 /build/frontend/.next/standalone ./frontend
COPY --from=frontend-build --chown=1000:1000 /build/frontend/.next/static ./frontend/.next/static
COPY --from=frontend-build --chown=1000:1000 /build/frontend/public ./frontend/public
COPY --chown=user:user docker/space-entrypoint.sh ./space-entrypoint.sh
RUN chmod +x ./space-entrypoint.sh && mkdir -p backend/data backend/checkpoints backend/results backend/models \
 && chown -R user:user backend
USER user
EXPOSE 7860
CMD ["/home/user/app/space-entrypoint.sh"]
