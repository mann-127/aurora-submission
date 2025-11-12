import os
import httpx
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
app = FastAPI(
    title="Member Q&A System",
    description="Answers natural language questions about member data."
)

# Load API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=GEMINI_API_KEY)

# The source API for member messages
DATA_SOURCE_URL = "https://november7-730026606190.europe-west1.run.app/messages"

# --- Data Models ---
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

# --- In-Memory Data Store ---
# In a production app, this would be a vector database (e.g., Chroma, Pinecone)
class DataStore:
    def __init__(self):
        self.messages = []  # Stores the raw text of messages
        self.embeddings = None # Stores the vector embeddings
        # Use a lightweight, high-performance model for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    async def fetch_and_index(self):
        """Fetches data from the API and creates embeddings."""
        print("Fetching member data...")
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                # Fetch all messages with a high limit
                response = await client.get(DATA_SOURCE_URL, params={"limit": 5000})
                response.raise_for_status() # Raises exception for 4xx/5xx
                
                raw_data = response.json()
                
                # Extract messages in the format: "user_name: message"
                # This helps the model understand who said what
                items = raw_data.get('items', [])
                self.messages = [
                    f"{item['user_name']}: {item['message']}"
                    for item in items
                    if 'message' in item and 'user_name' in item
                ]
                
                if not self.messages:
                    print("Warning: No messages found.")
                    return

                print(f"Fetched {len(self.messages)} messages. Creating embeddings...")
                # Create embeddings for all messages (this can take a moment)
                self.embeddings = self.model.encode(self.messages)
                print("Embeddings created. Service is ready.")

        except httpx.RequestError as e:
            print(f"Error fetching data: {e}")
            # Allow the app to run, but /ask will fail
            self.messages = []
            
    def retrieve_relevant_context(self, question: str, top_k: int = 10) -> str:
        """Finds the most relevant message(s) for a given question."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return ""

        # Create embedding for the question
        question_embedding = self.model.encode([question])

        # Find similarity between question and all messages
        similarities = cosine_similarity(question_embedding, self.embeddings)[0]

        # Get indices of the top_k most similar messages
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]

        # Build the context string
        context = ""
        for idx in top_k_indices:
            # Only include relevant results
            if similarities[idx] > 0.3: # Relevance threshold
                context += f"{self.messages[idx]}\n"
        
        return context

# Create a single instance of the data store
data_store = DataStore()

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """On app startup, fetch and index all member data."""
    await data_store.fetch_and_index()

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Member Q&A System is running.",
        "indexed_messages": len(data_store.messages)
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Accepts a natural-language question and returns an answer.
    
    Example:
        POST /ask
        {
            "question": "When is Layla planning her trip to London?"
        }
        
        Response:
        {
            "answer": "Based on the messages, Layla needs a suite for five nights..."
        }
    """
    if not data_store.messages:
        raise HTTPException(
            status_code=503,
            detail="Service is unavailable. Data store is not initialized."
        )

    # 1. Retrieve: Find relevant messages from our data
    context = data_store.retrieve_relevant_context(request.question)

    if not context:
        return AnswerResponse(answer="I could not find any relevant information in the member messages to answer this question.")

    # 2. Augment & Generate: Pass context + question to the LLM
    prompt = f"""You are a helpful assistant answering questions about member preferences and activities.

Answer the user's question based ONLY on the provided member messages below.
If the answer is not in the messages, clearly state that you cannot find the information.
Be concise but informative.

--- MEMBER MESSAGES ---
{context}
--- END MESSAGES ---

Question: {request.question}

Answer (be specific and cite member names when relevant):"""

    try:
        # Use a fast and capable model for Q&A
        llm = genai.GenerativeModel('gemini-1.5-flash')
        response = await llm.generate_content_async(prompt)
        
        answer = response.text.strip()
        
        # Handle edge cases
        if not answer:
            answer = "I'm having trouble generating an answer. Please try rephrasing your question."
        
        return AnswerResponse(answer=answer)
        
    except Exception as e:
        print(f"Error calling GenerativeModel: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating answer. Please check your GEMINI_API_KEY is valid."
        )
