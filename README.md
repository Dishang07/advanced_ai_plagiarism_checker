# Advanced AI Plagiarism Checker

A FastAPI-based plagiarism checker that analyzes text from pasted input or uploaded documents and reports:

- Potential plagiarism (internal database + web snippet matching)
- AI-likely writing patterns (heuristics + semantic similarity)
- Sentence-level explainability and confidence
- Downloadable analysis reports (PDF)

## Features

- Supports input as:
  - Plain text
  - PDF (.pdf)
  - DOCX (.docx)
- Sentence/chunk processing for long content
- Internal semantic matching using SentenceTransformer embeddings
- Internet source matching using DuckDuckGo HTML search snippets
- AI writing likelihood scoring with transparent metrics
- Visual frontend with highlighted sentence-level results
- Report export endpoints for HTML and PDF

## Tech Stack

- Backend: FastAPI
- NLP embeddings: sentence-transformers (`all-MiniLM-L6-v2`)
- Similarity: scikit-learn cosine similarity
- File parsing:
  - PDF: PyMuPDF
  - DOCX: python-docx
- Async HTTP: aiohttp
- Parsing search result HTML: BeautifulSoup4
- PDF report export: reportlab
- Frontend: static HTML/CSS/JavaScript

## Project Structure

```text
ai-plagarishm-checker/
├─ main.py
├─ search_utils.py
├─ requirements.txt
├─ test_check.py
└─ static/
   ├─ index.html
   ├─ script.js
   └─ styles.css
```

## Requirements

- Python 3.9+ (recommended)
- pip
- Internet connection (needed for web-source matching and first model download)

## Setup

### 1) Clone and enter the project

```bash
git clone <your-repo-url>
cd ai-plagarishm-checker
```

### 2) Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## Run the App

```bash
python main.py
```

Server starts at:

- App UI: http://127.0.0.1:8000/

Alternative run command:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## How It Works

1. Extract text from uploaded file or use pasted text.
2. Split text into sentence-like chunks.
3. Compare chunk embeddings with:
   - Internal reference database
   - Web snippets from DuckDuckGo HTML search
4. Compute plagiarism score from flagged chunks.
5. Compute AI usage score from:
   - Style/structure heuristics
   - Similarity against AI-reference phrase embeddings
6. Return full JSON response and render highlighted output in UI.

## API Endpoints

### GET /

Serves the main web interface.

### POST /api/check

Analyzes text or file input.

Form fields:

- `text` (optional string)
- `file` (optional UploadFile: .pdf, .txt)
- `student_name` (optional string)
- `assignment_name` (optional string)
- `submission_date` (optional string)

Returns JSON including:

- `plagiarism_score`
- `semantic_similarity_score`
- `results` (sentence-level flags, sources, reasons, confidence)
- `ai_usage_score`, `ai_usage_level`
- `ai_analysis` metrics
- `report_token` for report downloads

### POST /api/report/download

Downloads a report from full analysis payload.

Form fields:

- `analysis_json` (JSON string)
- `format` (`html` or `pdf`)

### GET /api/report/download

Downloads a report by token.

Query params:

- `token` (required)
- `format` (`pdf`, default `html`)



## Notes and Limitations

- Internet source checks depend on third-party search result pages and may be rate-limited or unavailable.
- Similarity/AI scores are heuristic indicators, not legal proof of plagiarism or definitive AI authorship.
- Very short text reduces confidence in AI-likelihood estimation.
- The first run can be slower because the transformer model may download and initialize.

## Testing

A helper test script exists:

- `test_check.py`

It runs a local server thread and posts a sample PDF (`dummy.pdf`) to `/api/check`.
Ensure that `dummy.pdf` exists in the project root before running it.


