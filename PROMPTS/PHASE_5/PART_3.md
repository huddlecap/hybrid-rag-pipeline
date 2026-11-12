Create a `docker-compose.yml` file in the root directory to orchestrate the RAG pipeline. The architecture must include four distinct services:

1. **chromadb:**
   - Use the official `chromadb/chroma:latest` image.
   - Map port `8001:8000` to avoid conflicts with the FastAPI server on the host machine.
   - Mount a persistent volume: `./chroma_data:/chroma/chroma`.
   - Implement a healthcheck (e.g., curling `localhost:8000/api/v1/heartbeat`) so downstream services know when it is ready to accept connections.

2. **api:**
   - Build from the `./api` directory.
   - Map port `8000:8000`.
   - Pass through external environment variables for `GROQ_API_KEY` and `COHERE_API_KEY`.
   - Set internal environment variables `CHROMA_HOST=chromadb` and `CHROMA_PORT=8000`.
   - Add a `depends_on` condition requiring `chromadb` to be healthy before starting.

3. **seeder:**
   - Build from the same `./api` directory (reusing the backend image to access the Python environment and dependencies).
   - Override the default command to execute: `python scripts/seed.py`.
   - Pass through the required `GROQ_API_KEY` and `COHERE_API_KEY` environment variables.
   - Set `CHROMA_HOST=chromadb` and `CHROMA_PORT=8000`.
   - Add a `depends_on` condition requiring `chromadb` to be healthy.
   - Set `restart: "no"` since this is a one-time init script.

4. **dashboard:**
   - Build from the `./dashboard` directory.
   - Map port `5173:80` (assuming Nginx will serve the static files on port 80 internally).
   - Add a `depends_on` condition for the `api` service.

Define the `chroma_data` volume at the bottom of the file. Do not generate the individual Dockerfiles yet; focus entirely on the compose orchestration.

---

First, execute the following architectural fixes to prepare the repository for containerization:

1. **Update `docker-compose.yml`:**
   - Modify the `api` and `seeder` services: change `build: ./api` to `context: .` and add `dockerfile: api/Dockerfile`. This ensures the root `src/`, `scripts/`, and `requirements.txt` are included in the build context.
   - Fix the `chromadb` volume mismatch: change the bind mount (`./chroma_data:/chroma/chroma`) in the service to use a named volume (`chroma_data:/chroma/chroma`), matching the declaration at the bottom of the file.
   - Update the `chromadb` healthcheck to use `wget` (which is included in the Chroma image) instead of `curl`: `test: ["CMD-SHELL", "wget --no-verbose --tries=1 --spider http://localhost:8000/api/v1/heartbeat || exit 1"]`.

2. **Update ChromaDB Initialization (`src/retriever.py`, `src/indexer.py`, etc.):**
   - Locate where `chromadb.PersistentClient` is initialized.
   - Import `os`. Check for the presence of the `CHROMA_HOST` and `CHROMA_PORT` environment variables.
   - If they exist, instantiate and return an HTTP client: `chromadb.HttpClient(host=os.environ["CHROMA_HOST"], port=int(os.environ["CHROMA_PORT"]))`.
   - If they do not exist, fall back to the original `chromadb.PersistentClient(path=...)` logic so local development remains unbroken.

Second, create the backend container instructions in `api/Dockerfile`:

1. **Base Image:** Use `python:3.11-slim`.
2. **System Dependencies:** Set `WORKDIR /app`.
3. **Python Dependencies:** Copy `requirements.txt` to `/app/` and run `pip install --no-cache-dir -r requirements.txt`.
4. **Context Copy:** Copy the `api/`, `src/`, and `scripts/` directories into `/app/`. (Assume `scripts/` will be created in the next step).
5. **Execution:** Expose port `8000` and set the command: `CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]`.

---

Implement the infrastructure configuration fixes for the Docker setup.

1. **Root `.dockerignore`:**
   Create a `.dockerignore` file in the root directory to prevent build context bloat. Add these exact lines:
   .venv
   node_modules
   **pycache**
   chroma_data
   .git
   .env
   dist
   \*.pkl

2. **Update `docker-compose.yml` (Volumes & Healthcheck):**
   - **Healthcheck:** Update the `chromadb` healthcheck command to try v2, then fallback to v1:
     `test: ["CMD-SHELL", "wget --no-verbose --tries=1 --spider http://localhost:8000/api/v2/heartbeat || wget --no-verbose --tries=1 --spider http://localhost:8000/api/v1/heartbeat || exit 1"]`
   - **Shared Data Volume:** Add a new named volume at the bottom of the file: `shared_data:`.
   - Mount this volume in both the `api` and `seeder` services under their `volumes` configuration:
     `- shared_data:/app/data`

---

Create the database seed script to handle automatic data ingestion on the first container boot.

Create `scripts/seed.py` with this exact logic flow:

- **Wait for ChromaDB:** Loop and ping the Chroma HTTP client until it responds successfully (max 30 retries, 2s sleep).
- **Check for Existing Data:** Check if any collections exist. If yes, print `[INFO] Database already seeded. Exiting.` and exit `0`.
- **Generate Synthetic Corpus:** If empty, programmatically generate a ~3,500-word Markdown document (e.g., a dummy "Huddl API Reference" containing repetitive but realistic endpoint descriptions, JSON schemas, and rate-limiting rules). Save this temporarily to `/app/data/seed_corpus.md`.
- **Ingestion Pipeline:**
  - Print `[INFO] Ingesting 3,500-word synthetic corpus...`
  - Import your existing chunking logic from `src`.
  - Import your existing indexing/embedding logic from `src`.
  - Process the generated markdown file, embed the chunks into ChromaDB, and generate the BM25 `.pkl` index inside `/app/data/index/bm25/`.
- Print `[INFO] Seeding complete.` and exit gracefully.

---

Generate the frontend multi-stage container and Nginx proxy setup.

1. **Create `dashboard/nginx.conf`:**
   Write an Nginx configuration file to serve the React SPA and reverse proxy API requests:
   - Listen on port 80.
   - Set the root to `/usr/share/nginx/html` and index to `index.html`.
   - Add a location block for `/` utilizing `try_files $uri $uri/ /index.html;` to support client-side routing.
   - Add a location block for `/v1/` that proxies traffic to the backend container: `proxy_pass http://api:8000/v1/;`. Include standard proxy headers (`Host`, `X-Real-IP`, `X-Forwarded-For`).

2. **Create `dashboard/Dockerfile`:**
   Write a multi-stage Dockerfile to compile the Vite application and serve it via Nginx:
   - **Stage 1 (Builder):** Use `node:18-alpine` as `builder`. Set `WORKDIR /app`. Copy `package.json` (and `package-lock.json` / `yarn.lock` if present), run `npm install`, copy the rest of the dashboard directory, and execute `npm run build`.
   - **Stage 2 (Server):** Use `nginx:alpine`. Copy the compiled `dist` folder from the `builder` stage into `/usr/share/nginx/html`. Copy the `dashboard/nginx.conf` file into `/etc/nginx/conf.d/default.conf` (overwriting the default). Expose port 80, and set the command to `CMD ["nginx", "-g", "daemon off;"]`.

---

Yes, make that change. Update `dashboard/src/App.jsx` (and any other frontend files making API calls) to set `API_BASE = ''`. This ensures all `fetch` requests resolve to relative paths (e.g., `/v1/ask`), routing them seamlessly through the Nginx reverse proxy to the backend container.

Once updated, we are fully ready for launch. Provide me with:

1. The exact terminal command to build and boot the entire stack (e.g., `docker-compose up --build`).
2. A brief checklist of what terminal logs I should look for to confirm ChromaDB is up, the seeder has finished the synthetic corpus, and the backend API is successfully listening.
