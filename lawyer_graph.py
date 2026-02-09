import os
from dotenv import load_dotenv
from typing import Annotated, Dict, TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

# --- CONFIGURATION ---
load_dotenv()  # Load environment variables from .env file

# Error handling for GROQ_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ ERROR: GROQ_API_KEY is still None! Check your .env file.")
else:
    print(f"✅ GROQ Key found: {GROQ_API_KEY[:5]}...")

# Error handling for GOOGLE_API_KEY
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("❌ ERROR: GOOGLE_API_KEY is still None! Check your .env file.")
else:
    print(f"✅ Google Key found: {GOOGLE_API_KEY[:5]}...")

VECTOR_DB_PATH = "./chroma_db"

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    question: str
    context: str
    documents: List[str]
    generation: str
    uploaded_doc_content: str  # For the user upload feature
    hallucination_score: str   # "yes" or "no"
    retry_count: int

# --- MODELS ---
llm_reasoner = ChatGroq(model="gemini-1.5-flash", temperature=0.6, api_key=GOOGLE_API_KEY)
llm_validator = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_API_KEY)

# --- RETRIEVER (With Error Handling) ---
retriever = None
try:
    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        print("Loading embedding model...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        print(f"Loading Vector Store from {VECTOR_DB_PATH}...")
        vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        print("✅ Vector Store loaded successfully.")
    else:
        print("⚠️ VECTOR DB NOT FOUND. Please run 'uv run ingest.py' first.")
except Exception as e:
    print(f"⚠️ CRITICAL ERROR: Could not load Vector Database.\nDetails: {e}")

# --- NODES ---

def retrieve_node(state: AgentState):
    """Retrieves relevant legal sections based on the query."""
    print("--- RETRIEVING DOCUMENTS ---")
    question = state["question"]

    # Check if DB is loaded
    if retriever is None:
        return {
            "context": "🚨 SYSTEM ERROR: Legal Database not found. Please run ingest.py.", 
            "documents": []
        }

    try:
        docs = retriever.invoke(question)
        if not docs:
            return {"context": "No relevant legal sections found.", "documents": []}
            
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"documents": docs, "context": context}
    except Exception as e:
        return {"context": f"Error during retrieval: {str(e)}", "documents": []}
    
def generate_node(state: AgentState):
    """Reasoning Agent synthesizing laws and user situation."""
    print("--- GENERATING RESPONSE ---")
    question = state["question"]
    context = state["context"]
    uploaded_content = state.get("uploaded_doc_content", "No specific document uploaded.")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are 'Nyaya-Agent', an expert Indian Legal Assistant. 
        Your goal is to explain legal procedures to a common person clearly.
        
        INPUT DATA:
        1. General Indian Laws (BNS/BNSS): {context}
        2. User's Specific Document/Scenario: {uploaded_content}
        
        INSTRUCTIONS:
        - Analyze the user's situation using ONLY the retrieved laws.
        - Cite specific Sections (e.g., "Section 101 of BNS").
        - If the laws don't apply, admit it.
        """),
        ("human", "{question}")
    ])

    chain = prompt | llm_reasoner | StrOutputParser()
    try:
        response = chain.invoke({
            "context": context, 
            "uploaded_content": uploaded_content, 
            "question": question
        })
        return {"generation": response, "retry_count": state.get("retry_count", 0)}
    except Exception as e:
        return {"generation": f"Error in generation: {str(e)}"}
    
class GradeHallucination(BaseModel):
    """Binary score for hallucination check."""
    binary_score: str = Field(description="Answer 'yes' if the response is grounded in facts, 'no' if hallucinated.")

def hallucination_check_node(state: AgentState):
    """Recursive Check: Verifies if the answer is supported by facts."""
    print("--- CHECKING FOR HALLUCINATIONS ---")
    generation = state["generation"]
    context = state["context"]
    
    # If there was an error in previous steps, skip check
    if "Error" in generation or "SYSTEM ERROR" in context:
        return {"hallucination_score": "yes"}

    structured_llm = llm_validator.with_structured_output(GradeHallucination)
    
    try:
        score = structured_llm.invoke(f"FACTS: {context}\n\nRESPONSE: {generation}")
        return {"hallucination_score": score.binary_score}
    except Exception as e:
        print(f"Grader Error: {e}") 
        return {"hallucination_score": "yes"}
    
    
def check_hallucination(state: AgentState):
    """
    The 'Traffic Cop' function.
    It reads the state and decides the next step.
    """
    score = state["hallucination_score"]
    retries = state["retry_count"]
    
    # Logic: If grounded (yes) OR we already retried once -> Stop.
    if score == "yes" or retries >= 1: 
        print("--- DECISION: APPROVED (Or Max Retries Reached) ---")
        return "end"
    else:
        print("--- DECISION: HALLUCINATION DETECTED -> RETRYING ---")
        return "retry"
    
# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("hallucination_check", hallucination_check_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "hallucination_check")

workflow.add_conditional_edges(
    "hallucination_check",
    check_hallucination,
    {
        "end": END,
        "retry": "generate"
    }
)

app_graph = workflow.compile()