# Live preview of the viewer + labeling tool: builds the web app and serves it with the REST API.
# Only the API is included (no torch/mediapipe), so the image stays small. The database and the
# frame images are not part of the image, mount them into /data (see docker-compose.yml).

# The web build is platform independent, so it always runs natively (fast multi-arch builds).
FROM --platform=$BUILDPLATFORM node:26-alpine AS web
WORKDIR /web
# corepack is no longer bundled with the node images; install the pnpm version from package.json.
COPY web/package.json web/pnpm-lock.yaml ./
RUN npm install -g "$(node -p "require('./package.json').packageManager")"
RUN pnpm install --frozen-lockfile
COPY web/ ./
RUN pnpm build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GAZE_DATA_DIR=/data \
    GAZE_DATABASE=/data/gaze.db
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.162.7 uvicorn==0.54.0 sqlmodel==0.0.50
COPY gaze_estimator/ ./gaze_estimator/
COPY --from=web /web/dist ./web/dist

VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["python", "-m", "gaze_estimator.cli.serve", "--host", "0.0.0.0", "--port", "8000"]
