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

## Prerequisites

- The API running at `http://localhost:8000` (see the repository root).
- Node.js 18+.

## Setup

```bash
npm install
npm run dev
```

Open http://localhost:5173.

## Build

```bash
npm run build
```

Outputs static assets to `dist/`.

## API Base URL

Set in `src/App.jsx` as `API_BASE` (default `http://localhost:8000`). The API permits CORS from `http://localhost:5173` and `http://localhost:3000`.

## Layout

- `src/App.jsx` — UI, Standard/Compare modes, loader, timer, sidebar, request flow.
- `src/index.css` — Tailwind base layer, gooey loader animation, global scrollbars.
