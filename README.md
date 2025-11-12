# Aurora Member Q&A System

A natural language question-answering system that answers questions about member data from the Aurora API using RAG (Retrieval-Augmented Generation).

## 📑 Table of Contents

- [Live Demo](#-live-demo)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Architecture](#️-architecture)
- [Alternative Approaches](#-alternative-approaches-considered)
- [Data Insights](#-data-insights--anomalies)
- [Example Queries](#-example-queries)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)

---

## �🚀 Live Demo

**API Endpoint**: [To be deployed]

**Example Usage**:
```bash
curl -X POST "https://your-deployment-url/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "When is Layla planning her trip to London?"}'
```

## 📋 Features

- **Natural Language Q&A**: Ask questions in plain English about member preferences, trips, and activities
- **RAG Architecture**: Uses semantic search + LLM generation for accurate, context-aware answers
- **Fast & Efficient**: Sentence transformers for quick semantic search, Gemini for intelligent responses
- **Real-time Data**: Fetches and indexes latest member messages on startup
- **RESTful API**: Simple JSON API endpoint for easy integration

---

## ⚡ Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Key
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### 3. Run the Server
```bash
uvicorn main:app --reload
```

The service will start on http://localhost:8000

### 4. Test It
Visit http://localhost:8000/docs for interactive API documentation, or:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "When is Layla planning her trip to London?"}'
```

---

## 🏗️ Architecture

### Current Implementation: RAG with Sentence Transformers + Gemini

```
Question → Embedding → Semantic Search → Top-K Messages → LLM → Answer
```

**Components**:
1. **Data Ingestion**: Fetch all 3,349 messages from Aurora API
2. **Vectorization**: Create embeddings using `all-MiniLM-L6-v2` (384-dim vectors)
3. **Retrieval**: Cosine similarity search to find top-10 relevant messages
4. **Generation**: Google Gemini 1.5 Flash generates answer from context

**Why this approach?**
- ✅ Fast inference (< 2 seconds per query)
- ✅ Accurate context retrieval
- ✅ Low memory footprint (~150MB)
- ✅ No training data required
- ✅ Easy to update with new data

## 🔄 Alternative Approaches Considered

### 1. **Fine-tuned LLM**
**Approach**: Fine-tune a small LLM (e.g., GPT-3.5, Llama-2-7B) on member data.

**Pros**:
- Deep understanding of member patterns
- No need for retrieval step
- Can learn implicit relationships

**Cons**:
- Requires substantial training data and compute
- Risk of hallucination/memorization
- Expensive to update with new data
- Harder to debug incorrect answers

**Verdict**: ❌ Overkill for this dataset size; RAG is more flexible

---

### 2. **Traditional NLP + Rules**
**Approach**: Use NER (Named Entity Recognition) + intent classification + rule-based extraction.

**Pros**:
- Fully explainable
- Deterministic results
- No API costs

**Cons**:
- Requires extensive rule engineering
- Brittle to question phrasing variations
- Poor handling of complex queries
- High maintenance overhead

**Verdict**: ❌ Too rigid for natural language queries

---

### 3. **Graph Database + Cypher Queries**
**Approach**: Model members, preferences, and events as a knowledge graph; convert questions to graph queries.

**Pros**:
- Excellent for relationship queries
- Structured, queryable knowledge
- Fast for complex joins

**Cons**:
- Requires data modeling and schema design
- NL → Cypher translation is non-trivial
- Doesn't handle unstructured preferences well
- Setup complexity

**Verdict**: ⚠️ Great for structured data, but overkill here

---

### 4. **Vector Database (Pinecone/Weaviate)**
**Approach**: Same as current, but use dedicated vector DB instead of in-memory.

**Pros**:
- Better for very large datasets (millions of vectors)
- Built-in filtering and hybrid search
- Production-ready infrastructure

**Cons**:
- Additional service dependency
- Unnecessary for 3,349 messages
- Extra latency from network calls

**Verdict**: ⚠️ Would use for larger scale (10K+ members)

---

### 5. **Hybrid: BM25 + Semantic Search**
**Approach**: Combine keyword search (BM25) with semantic search, then re-rank.

**Pros**:
- Best of both worlds (keyword + semantic)
- Better handling of exact matches
- More robust retrieval

**Cons**:
- Added complexity
- Minimal benefit for this conversational dataset
- Slower retrieval

**Verdict**: ⚠️ Could improve accuracy by 5-10%, but not essential

---

## 📊 Data Insights & Anomalies

### Dataset Overview
- **Total Messages**: 3,349
- **Unique Members**: 10
- **Date Range**: November 8, 2024 → November 8, 2025
- **Message Distribution**: Fairly balanced (288-365 messages per user)

### Key Findings

#### ✅ Data Quality (Good News)
1. **No Missing Fields**: All messages have complete `id`, `user_id`, `user_name`, `timestamp`, and `message`
2. **Unique IDs**: All 3,349 message IDs are unique
3. **Consistent User IDs**: Each user name maps to exactly one user ID (no duplicates)

#### ⚠️ Anomalies & Inconsistencies

1. **Future Timestamps**
   - **Issue**: ~50% of messages have timestamps in the future (up to November 8, 2025)
   - **Impact**: May cause confusion for time-based queries
   - **Recommendation**: Clarify whether these are scheduled requests or test data

2. **Message Type Imbalance**
   - Booking requests: 993 (30%)
   - Questions: 1,100 (33%)
   - Preferences: 316 (9%)
   - Complaints: 94 (3%)
   - **Observation**: Very few complaints despite high-value clientele—may indicate under-reporting

3. **Activity Pattern**
   - **Most Active**: Lily O'Sullivan (365 messages)
   - **Least Active**: Lorenzo Cavalli (288 messages)
   - **Variance**: 27% difference between highest/lowest
   - **Question**: Is this natural variation or does it indicate engagement issues?

4. **Ambiguous User Name** 
   - User is "Amina Van Den Berg", but question asks about "Amira"
   - **Impact**: No exact match for "Amira's favorite restaurants"
   - **Recommendation**: Implement fuzzy name matching or clarify user aliases

5. **Limited Temporal Context**
   - Messages mention "tonight", "next week", "this Friday" without absolute dates
   - **Impact**: Difficult to answer "when" questions precisely
   - **Recommendation**: System should acknowledge uncertainty in time-based answers

6. **Incomplete Car Data**
   - Sample question: "How many cars does Vikram Desai have?"
   - **Finding**: Dataset only mentions car *service* requests, not ownership
   - **Impact**: Cannot accurately answer ownership questions from current data
   - **Recommendation**: Distinguish between service requests and asset ownership

### Sample Insights
- **Layla's London Plans**: Found 9 mentions of London—needs suite at Claridge's, car service, art classes for daughter
- **Vikram's Cars**: Found 16 car-related messages, but all are service requests, not ownership data
- **Amina's Restaurants**: 29 restaurant mentions including high-end steakhouses, birthday cakes, Barcelona recommendations

---

## 🛠️ Deployment

### Quick Deploy Options

**Google Cloud Run** (Recommended):
```bash
gcloud run deploy aurora-qa \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your-api-key
```

**Railway**:
1. Sign up at railway.app
2. Deploy from GitHub
3. Set `GEMINI_API_KEY` environment variable

**Render**:
- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Set `GEMINI_API_KEY` environment variable

**Docker** (Any platform):
```dockerfile
# Create Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t aurora-qa .
docker run -p 8000:8000 -e GEMINI_API_KEY=your-key aurora-qa
```

---

## 📁 Project Structure

```
aurora-submission/
├── main.py              # FastAPI application with RAG implementation
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # This file - comprehensive documentation
```

---

## 🔮 Future Improvements

1. **Multi-turn Conversations**: Add conversation history context
2. **Structured Extraction**: Extract preferences into structured database
3. **Confidence Scores**: Return answer confidence/sources
4. **User Disambiguation**: Handle name variations (Amina/Amira)
5. **Temporal Resolution**: Better handling of relative time references
6. **Caching**: Cache frequent queries for faster response
7. **Analytics**: Track question patterns and answer quality

---

## 🤝 Contributing

This is a submission project, but feedback is welcome!

---

## 📄 License

MIT License - See LICENSE file for details
