# Hybrid RAG Dashboard

React + Vite + Tailwind frontend for the Hybrid RAG API.

## Features

- Ask questions against the indexed corpus and render grounded answers with clickable `[n]` citations.
- **Standard** and **Compare** modes; Compare fires concurrent `hybrid` and `dense` requests and renders them side by side.
- Retrieved-chunks accordion showing the rerank score as a percentage (hybrid) or raw distance (dense).
- Confidence breakdown: Answer Relevance, Faithfulness, and Retrieval Relevance.
- Sidebar lists indexed documents grouped by source file, expandable to chunk index and full text.
- Pure-CSS "gooey" loading animation with a live elapsed timer.
- Slim custom scrollbars and a diffuse-shadow question input.

## Running in Docker (recommended)

From the repository root:

```bash
docker compose up --build -d
```

The dashboard is built by `dashboard/Dockerfile` and served by Nginx on http://localhost:5173. Nginx proxies `/v1/` and `/health` to `api:8000`, so no CORS configuration is needed.

## Local Development

Prerequisites:

- The API running at `http://localhost:8000` (see the repository root).
- Node.js 18+.

```bash
npm install
npm run dev
```

Open http://localhost:5173. Vite proxies `/v1` and `/health` to `http://localhost:8000` (`vite.config.js`), matching the production Nginx routes.

## Build

```bash
npm run build
```

Outputs static assets to `dist/`.

## API Base URL

`src/App.jsx` sets `API_BASE = ''`, so requests are relative and same-origin. In development, Vite proxies `/v1` and `/health` to `localhost:8000`; in production, Nginx proxies the same paths to the `api` service. This keeps one code path and avoids CORS in the deployed stack.

If you bypass both proxies and call the API directly from another origin, the API currently allows only `http://localhost:5173` and `http://localhost:3000` (`api/main.py:177-183`).

## Layout

- `src/App.jsx` — UI, Standard/Compare modes, loader, timer, sidebar, request flow.
- `src/index.css` — Tailwind base layer, gooey loader animation, global scrollbars.
- `vite.config.js` — dev server and `/v1` + `/health` proxy.
- `nginx.conf` — production SPA fallback and API reverse proxy.
