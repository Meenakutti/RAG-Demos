# SupportDesk-RAG: A Support Ticket Retrieval & Troubleshooting Assistant

## Hands-On RAG Workshop with OpenAI

### Workshop Overview
This comprehensive workshop teaches you to build a production-ready Retrieval-Augmented Generation (RAG) system using OpenAI embeddings and language models. By the end, you'll have a working assistant that answers incident queries using retrieved ticket context, with strong safeguards against hallucinations.

### Learning Objectives
- ✅ Generate and work with OpenAI embeddings
- ✅ Master chunking strategies for optimal retrieval  
- ✅ Compare 5 different indexing strategies (LlamaIndex)
- ✅ Implement a complete RAG pipeline with LangChain
- ✅ Evaluate with two-layer metrics (retrieval + generation)
- ✅ Deploy anti-hallucination safeguards
- ✅ Build agentic RAG systems with multi-step reasoning

---

## 🚀 Quick Start

### 1. Create & activate a Python virtual environment
```bash
# choose a name like ".venv" or "env"; do not use ".env" (that file stores
# your configuration variables and will block venv creation)
# If you have multiple Python versions installed, pick the correct one with
# the Windows launcher (e.g. use 3.13 since 3.14 is incompatible with Chroma):
py -3.13 -m venv .venv
# or explicitly call a python executable:
# C:\Python313\python.exe -m venv .venv
```

#### Activating the venv
- **PowerShell** (preferred on Windows):
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  If you see an error such as
  > running scripts is disabled on this system
  
  then your execution policy is restricting script execution. Fix it by
  running (as Administrator or current user):
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  # or, for a more permissive policy:
  # Set-ExecutionPolicy Unrestricted -Scope CurrentUser
  ```
  then try activation again.

- **cmd.exe** (no policy issues):
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

- **Git Bash / WSL**:
  ```bash
  source .venv/Scripts/activate
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API key
# OPENAI_API_KEY=sk-your-key-here
```

> **Python version notice:**
> The workshop examples rely on the Chroma vector store (via the `chromadb`
> package). Chroma currently suffers from a compatibility bug with Python
> 3.14 and later (pydantic v1 can't infer certain field types). If you are
> running Python 3.14+, you'll see a traceback about `unable to infer type for
> attribute "chroma_server_nofile"` when creating the database. To avoid this,
> please use **Python 3.13 or earlier**. You can install a compatible Python
> version with `pyenv`, `conda`, or the official Windows installer.


### 3. Test Installation
```bash
cd modules/1_embeddings
python demo.py
```

---

## Workshop Modules

### Module 1: Embeddings (`modules/1_embeddings/`)
**Learn:**
- Generate embeddings using OpenAI API
- Compute semantic similarity scores
- Visualize similarity relationships with heatmaps

**Run:**
```bash
cd modules/1_embeddings
python demo.py
```

---

### Module 2: Chunking (`modules/2_chunking/`)
**Learn:**
- Fixed-size vs recursive vs semantic chunking
- Structure-aware splitting (Markdown/HTML)
- Build vector stores with Chroma

**Run:**
```bash
cd modules/2_chunking
python demo.py
```

---

### Module 3: Indexing Strategies (`modules/3_indexing/`)
**Learn:**
- Vector Index - Semantic similarity search (most common)
- Summary Index - High-level document summaries
- Tree Index - Hierarchical retrieval patterns
- Keyword Table Index - Traditional keyword matching
- Hybrid Retrieval - Combining multiple strategies

**Technologies:** LlamaIndex for clean indexing abstractions

**Run:**
```bash
cd modules/3_indexing
python demo.py
```

---

### Module 4: RAG Pipeline (`modules/4_rag_pipeline/`)
**Learn:**
- Complete RAG architecture
- LangChain integration
- Prompt engineering for grounded responses
- Anti-hallucination strategies

**Run:**
```bash
cd modules/4_rag_pipeline
python demo.py
```

---

### Module 5: Evaluation (`modules/5_evaluation/`)
**Learn:**
- Two-layer evaluation approach (Retrieval + Generation)
- Retrieval metrics (Precision@K, Recall@K, F1)
- Generation metrics (Groundedness, Completeness)
- LLM-as-judge for generation evaluation
- Creating comprehensive evaluation reports

**Technologies:** FAISS, LLM-as-Judge evaluation

**Run:**
```bash
cd modules/5_evaluation
python demo.py
```

---

### Module 6: Agentic RAG (`modules/6_agentic_rag/`)
**Learn:**
- Creating custom tools for LangChain agents
- Building agents with OpenAI function calling
- Implementing conversation memory
- Multi-step reasoning with tool selection
- Comparing agentic vs direct RAG approaches

**Technologies:** LangChain Agents, OpenAI Function Calling

**Run:**
```bash
cd modules/6_agentic_rag
python demo.py
```

---

## 📁 Repository Structure

```
SupportDesk-RAG-Workshop/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                # Environment template
├── POST_CLASS_GUIDE.md         # Post-workshop learning guide
├── data/
│   └── synthetic_tickets.json  # Sample support tickets
└── modules/
    ├── 1_embeddings/
    │   ├── demo.py             # Working demo code
    │   ├── notes.md            # Instructor notes
    │   └── exercises.md        # Practice exercises
    ├── 2_chunking/
    │   ├── demo.py
    │   ├── notes.md
    │   └── exercises.md
    ├── 3_indexing/
    │   ├── demo.py
    │   ├── notes.md
    │   └── exercises.md
    ├── 4_rag_pipeline/
    │   ├── demo.py
    │   ├── notes.md
    │   └── exercises.md
    ├── 5_evaluation/
    │   ├── demo.py
    │   ├── notes.md
    │   ├── exercises.md
    │   ├── solutions.py
    │   └── evaluation_queries.json
    └── 6_agentic_rag/
        ├── demo.py
        ├── notes.md
        ├── exercises.md
        ├── solutions.py
        ├── tools.py
        ├── test_setup.py
        └── README.md
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (defaults shown)
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```

### Model Options

**Embeddings:**
- `text-embedding-3-small` (1536 dims, recommended)
- `text-embedding-3-large` (3072 dims, highest quality)

**Chat:**
- `gpt-4o-mini` (recommended for cost/performance)
- `gpt-4o` (most capable)

---

## 💰 Cost Estimate

Running all modules: **< $0.10**
- Embeddings: ~$0.01 (20 tickets + queries)
- Chat completions: ~$0.05 (RAG pipeline demos)

See [OpenAI Pricing](https://openai.com/pricing) for current rates.

---

## 🎯 Prerequisites

- Python 3.8+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Basic understanding of Python
- Familiarity with APIs (helpful but not required)

---

## 🛠️ Troubleshooting

### OpenAI API Errors
- Verify API key in `.env` file
- Check credits: https://platform.openai.com/usage
- Rate limits: Wait 60s if you get 429 errors

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

### Path Issues
- Always run demos from their module directory
- Ensure `data/synthetic_tickets.json` exists

---

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Chroma Documentation](https://docs.trychroma.com/)

---

## 🤝 Contributing

Found a bug or have suggestions? Open an issue or submit a pull request!

---

## 📄 License

MIT License - Feel free to use for learning and teaching!
