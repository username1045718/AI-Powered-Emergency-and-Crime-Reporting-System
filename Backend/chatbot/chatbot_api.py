import os
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import google.generativeai as gen_ai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load FAQ data with error handling
try:
    with open("faq.json", "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    logger.info(f"✅ Loaded {len(qa_data)} FAQ entries")
except FileNotFoundError:
    logger.error("❌ faq.json file not found!")
    raise
except json.JSONDecodeError:
    logger.error("❌ Invalid JSON in faq.json!")
    raise

# Load embedding model with error handling
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("✅ Sentence transformer model loaded")
except Exception as e:
    logger.error(f"❌ Failed to load embedding model: {e}")
    raise

# Encode all questions from FAQ
qa_questions = [item["question"] for item in qa_data]
qa_embeddings = embedder.encode(qa_questions, convert_to_tensor=True)
logger.info(f"✅ Encoded {len(qa_questions)} FAQ questions")

# Configure Gemini AI
if GOOGLE_API_KEY:
    gen_ai.configure(api_key=GOOGLE_API_KEY)
    model = gen_ai.GenerativeModel("gemini-2.0-flash-exp")  # Using flash for better performance
    logger.info("✅ Gemini AI configured")
else:
    logger.error("❌ GOOGLE_API_KEY not found in .env file!")
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

# Initialize FastAPI app
app = FastAPI(
    title="Crime Reporting Chatbot API",
    description="AI-powered chatbot for crime reporting assistance",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173", 
        "http://localhost:8080",
        "https://your-frontend-domain.com"  # Add your production domain
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Define the input schema
class Message(BaseModel):
    message: str
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v) > 1000:  # Limit message length
            raise ValueError('Message too long (max 1000 characters)')
        return v.strip()

# Define response schema
class ChatResponse(BaseModel):
    response: str
    source: str  # "faq", "ai", or "fallback"
    confidence: Optional[float] = None

# Enhanced safe topics with more comprehensive coverage
SAFE_TOPICS = [
    # Crime types
    "cybercrime", "online scams", "fraud", "theft", "robbery", "burglary",
    "assault", "harassment", "stalking", "domestic violence", "child abuse",
    
    # Safety and prevention
    "self defense", "crime prevention", "safety tips", "emergency",
    "police", "law enforcement", "legal advice", "victim rights",
    
    # Reporting and procedures
    "file complaint", "report crime", "fir", "investigation", "evidence",
    "witness", "court", "legal process", "bail", "arrest",
    
    # Cyber security
    "password", "phishing", "malware", "identity theft", "data breach",
    "online safety", "social media", "privacy",
    
    # Emergency situations
    "emergency", "help", "urgent", "danger", "threat", "violence"
]

# Enhanced safety check
def is_safe_query(query: str) -> tuple[bool, str]:
    """
    Check if query is safe and return the matched topic
    """
    query_lower = query.lower()
    
    # Check for greeting patterns
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "namaste"]
    if any(greeting in query_lower for greeting in greetings):
        return True, "greeting"
    
    # Check for safe topics
    for topic in SAFE_TOPICS:
        if topic in query_lower:
            logger.info(f"Query matched safe topic: {topic}")
            return True, topic
    
    # Check for question words (likely legitimate questions)
    question_words = ["what", "how", "when", "where", "why", "who", "can", "should", "is", "are"]
    if any(word in query_lower for word in question_words):
        return True, "question"
    
    logger.warning(f"Query '{query}' did not match safe topics")
    return False, "unsafe"

# Enhanced Gemini prompt
def create_gemini_prompt(user_input: str, matched_topic: str) -> str:
    """
    Create a contextual prompt for Gemini based on the matched topic
    """
    base_context = """You are a helpful assistant for a Crime Reporting System. 
    Provide accurate, helpful, and safe information related to crime reporting, legal procedures, and public safety.
    Keep responses concise (2-3 sentences) and actionable."""
    
    if matched_topic == "greeting":
        return f"{base_context}\n\nUser said: {user_input}\nRespond with a friendly greeting and ask how you can help with crime reporting or safety questions."
    elif matched_topic in ["cybercrime", "online scams", "fraud"]:
        return f"{base_context}\n\nUser is asking about cybercrime/online safety: {user_input}\nProvide specific advice about reporting cyber crimes and staying safe online."
    elif matched_topic in ["emergency", "urgent", "danger"]:
        return f"{base_context}\n\nUser has an urgent query: {user_input}\nPrioritize immediate safety advice and emergency contact information."
    else:
        return f"{base_context}\n\nUser question: {user_input}\nAnswer based on crime reporting and public safety context."

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Crime Reporting Chatbot"}

# Main chatbot endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(msg: Message):
    try:
        user_input = msg.message
        logger.info(f"Processing query: {user_input[:100]}...")  # Log first 100 chars
        
        # Step 1: Handle empty or very short queries
        if len(user_input) < 2:
            return ChatResponse(
                response="Please provide a more detailed question so I can help you better.",
                source="fallback"
            )
        
        # Step 2: Embed user input
        try:
            input_embedding = embedder.encode([user_input], convert_to_tensor=True)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise HTTPException(status_code=500, detail="Failed to process your message")
        
        # Step 3: Compute similarity with FAQ
        similarities = cosine_similarity(input_embedding, qa_embeddings)[0]
        max_index = np.argmax(similarities)
        max_score = float(similarities[max_index])
        
        logger.info(f"Best FAQ match score: {max_score:.3f}")
        
        # Step 4: If good match, return FAQ answer
        if max_score >= 0.75:  # Lowered threshold for better coverage
            matched_answer = qa_data[max_index]["answer"]
            matched_question = qa_data[max_index]["question"]
            logger.info(f"FAQ match found: {matched_question}")
            
            return ChatResponse(
                response=matched_answer,
                source="faq",
                confidence=max_score
            )
        
        # Step 5: Check if it's a safe topic for AI generation
        is_safe, matched_topic = is_safe_query(user_input)
        
        if is_safe:
            try:
                logger.info(f"Generating AI response for topic: {matched_topic}")
                
                prompt = create_gemini_prompt(user_input, matched_topic)
                gemini_response = model.generate_content(prompt)
                
                if gemini_response and gemini_response.text:
                    response_text = gemini_response.text.strip()
                    return ChatResponse(
                        response=f"🤖 *AI Assistant*: {response_text}",
                        source="ai",
                        confidence=max_score
                    )
                else:
                    logger.warning("Gemini returned empty response")
                    return ChatResponse(
                        response="I'm having trouble generating a response right now. Please try rephrasing your question or contact support.",
                        source="fallback"
                    )
                    
            except Exception as e:
                logger.error(f"Gemini AI Error: {e}")
                return ChatResponse(
                    response="I'm experiencing technical difficulties. Please try again or contact the police station directly for urgent matters.",
                    source="fallback"
                )
        
        # Step 6: Fallback for unsafe/unrelated queries
        return ChatResponse(
            response="I can only help with questions related to crime reporting, safety, and legal procedures. Please ask about filing complaints, crime prevention, or emergency procedures.",
            source="fallback"
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Additional endpoint to get FAQ categories (useful for frontend)
@app.get("/faq-categories")
async def get_faq_categories():
    """
    Return available FAQ categories for frontend display
    """
    try:
        categories = {}
        for item in qa_data:
            question = item["question"].lower()
            
            # Simple categorization logic
            if any(word in question for word in ["complaint", "file", "report", "submit"]):
                categories.setdefault("Filing Complaints", []).append(item["question"])
            elif any(word in question for word in ["cyber", "online", "hack", "scam"]):
                categories.setdefault("Cybercrime", []).append(item["question"])
            elif any(word in question for word in ["emergency", "urgent", "help"]):
                categories.setdefault("Emergency", []).append(item["question"])
            elif any(word in question for word in ["rights", "legal", "law"]):
                categories.setdefault("Legal Rights", []).append(item["question"])
            else:
                categories.setdefault("General", []).append(item["question"])
        
        return {"categories": categories}
        
    except Exception as e:
        logger.error(f"Error getting FAQ categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to get FAQ categories")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")