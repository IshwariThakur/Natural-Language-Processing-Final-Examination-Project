# AI CEO: Strategic Intelligence Agent — Airbus

An AI-powered strategic intelligence system that collects live information about Airbus, processes it through a RAG pipeline, and generates executive-level recommendations answering: **"If you were the CEO today, what would you do next and why?"**

---

## System Architecture

```mermaid
graph TB
    subgraph SOURCES["📡 Data Sources — Task 1"]
        A[Airbus<br/>98 docs]
        B[Boeing <br/>41 docs]
        C[ESA <br/>32 docs]
    end

    subgraph COLLECT["🔄 Automatic Collection"]
        D[Web Scrapers<br/>requests + BeautifulSoup4]
    end

    subgraph PROCESS["🧹 Information Processing — Task 3"]
        G[Noise Removal<br/>HTML · URLs · boilerplate · duplicates<br/>171 → 154 documents]
        H[Chunking + Embedding<br/>300-word chunks · 50-word overlap<br/>bge-small-en-v1.5 · 384 dimensions<br/>154 docs → 365 chunks]
    end

    subgraph STORE["🗄 Knowledge Repository — Task 2"]
        I[(ChromaDB<br/>Vector Store<br/>365 chunks<br/>cosine similarity)]
    end

    subgraph AICEO["🤖 AI CEO Pipeline — Tasks 4 · 5 · 6"]
        K[Semantic Retrieval<br/>RAG · top-k chunks]
        L[LLM Call 1<br/>Opportunities · Risks · Trends]
        M[LLM Call 2<br/>CEO Recommendations]
        N[LLM Call 3<br/>Executive Briefing]
        O[(Strategic Report<br/>JSON)]
    end

    subgraph DASH["📊 Executive Dashboard — Streamlit"]
        P[7 Sections<br/>Instant load from report]
        Q[Live Q&A<br/>RAG + Qwen3 8B]
    end

    A --> D
    B --> D
    C --> D
    D --> G
    G --> H
    H --> I
    I --> K
    K --> L
    L --> M
    M --> N
    N --> O
    O --> P
    I --> Q
    P --> Q
```

---

## Data Flow Diagram

```mermaid
flowchart LR
    A([🌐 Web Sources]) --> B[Automatic<br/>Collection]
    B --> C[(Raw Data<br/>171 docs)]
    C --> D[Cleaning &<br/>Processing]
    D --> E[(Cleaned Data<br/>154 docs)]
    E --> F[Chunking &<br/>Embedding]
    F --> G[(ChromaDB<br/>365 chunks)]

    G --> H[Semantic<br/>Retrieval]
    H --> I[LLM<br/>Qwen3 8B]
    I --> J[(Strategic<br/>Report JSON)]

    J --> K[Dashboard]
    G --> K
    K --> L([📊 Executive<br/>Dashboard])
```

---

## RAG Pipeline

```mermaid
sequenceDiagram
    participant U as User Query
    participant E as Encoder<br/>bge-small-en-v1.5
    participant C as ChromaDB<br/>Vector Store
    participant R as Retriever
    participant L as Decoder<br/>Qwen3 8B
    participant O as Strategic Output

    U->>E: query text
    E->>E: encode to 384-dim vector
    E->>C: cosine similarity search
    C->>C: compare against 365 chunks
    C->>R: top-k chunks + scores
    R->>R: deduplicate by title
    R->>L: evidence + structured prompt
    L->>L: generate analysis
    L->>O: Opportunities · Risks · Trends<br/>Recommendations · CEO Briefing
```

---

## NLP Preprocessing Pipeline

```mermaid
flowchart TD
    A([Raw Text<br/>171 documents]) --> B

    subgraph B["🧹 Noise Removal"]
        B1[Remove HTML tags]
        B2[Remove URLs]
        B3[Remove page boilerplate<br/>navigation · share bars · footers]
        B4[Remove file asset listings]
        B5[Drop irrelevant documents<br/>media invitations · admin notices]
        B6[Remove exact duplicates<br/>hash comparison]
    end

    B --> C

    subgraph C["⚠️ Deliberately NOT Applied"]
        C1[❌ Lowercasing]
        C2[❌ Stopword Removal]
        C3[❌ Stemming · Lemmatization]
        C4[❌ Punctuation Removal]
        C5[❌ Number Removal]
    end

    C --> D[Reason: Transformer embeddings need<br/>natural text with casing · numbers · word order<br/>Over-preprocessing loses semantic context]

    D --> E

    subgraph E["✂️ Chunking"]
        E1[300-word sliding windows]
        E2[50-word overlap between chunks]
        E3[154 documents → 365 chunks]
        E4[Each chunk stays within<br/>512-token model limit]
    end

    E --> F([Embedding<br/>384-dimensional vectors<br/>stored in ChromaDB])
```

---

## Technology Stack

| Component | Technology | Why chosen |
|---|---|---|
| Data collection | requests + BeautifulSoup4 | Reliable HTML scraping with retry/backoff |
| Knowledge repository | ChromaDB | Persistent vector store with metadata |
| Embedding model | BAAI/bge-small-en-v1.5 | Encoder-only · 384-dim · recommended in brief |
| LLM | Qwen3 8B via Ollama | Decoder-only · open-source · no paid API |
| Dashboard | Streamlit + Plotly | Interactive · on recommended list |
| Sentiment | VADER | Lexicon-based · fast · no model download |

---

## Key Design Decisions

### Why precompute the report offline?
Qwen3 8B takes ~15 minutes to generate the full report on a laptop. Running it live would freeze the dashboard. The report is generated once in the terminal and saved to JSON. The dashboard reads it instantly. The live Q&A box is the only real-time LLM call.

### Why chunking?
The embedding model has a hard 512-token limit. Long documents would be silently truncated without chunking. 300-word chunks with 50-word overlap ensure every part of every document is fully searchable. Result: 154 documents → 365 chunks.

### Why no stopword removal or lemmatization?
Transformer embeddings are trained on natural text and rely on word order, casing, and stopwords to understand meaning. Removing them degrades retrieval quality — the "over-preprocessing" pitfall from the course slides. Numbers like A350-1000 and €73.4bn carry critical domain meaning.

### Why three LLM calls instead of one?
A single prompt with all instructions plus all evidence plus all output templates exceeds the model's output token limit and gets cut off. Three focused calls (intelligence → recommendations → briefing) each produce complete, untruncated output.

### Why cosine similarity?
Cosine similarity measures the angle between vectors — it captures semantic similarity regardless of document length. A short chunk and a long chunk about the same topic score equally similar, unlike Euclidean distance which penalises shorter vectors.

### Why score sentiment on titles not full text?
VADER accumulates positive word counts over long text. A 3,000-word press release scores 0.96 (maximum positive) even when factually neutral. Titles (10–15 words) give a realistic 26% positive / 67% neutral / 7% negative distribution.

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install Ollama and pull the model
# Download from https://ollama.com
ollama pull qwen3:8b

# 3. Build the knowledge base
python main.py

# 4. Generate the strategic report (one-time, ~15 min)
python sources/strategic_report.py

# 5. Launch the dashboard
streamlit run sources/streamlit_app.py
```

---

## Data Sources

| Source | Category | Documents |
|---|---|---|
| Airbus Newsroom | Company Source | ~98 |
| Boeing Newsroom | Market / Competitor Source | 41 |
| ESA Newsroom | Research / Industry Source | 32 |
| **Total** | | **171 raw → 154 cleaned** |

---

## Key Numbers

| Metric | Value |
|---|---|
| Raw documents collected | 171 |
| After cleaning | 154 |
| Chunks in ChromaDB | 365 |
| Chunk size | 300 words |
| Chunk overlap | 50 words |
| Embedding dimensions | 384 |
| Model token limit | 512 |
| LLM parameters | 8 billion |
| LLM context window | 4096 tokens |
| LLM calls per report | 3 |
| Dashboard sections | 7 |
