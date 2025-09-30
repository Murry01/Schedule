# --- Standard Library Imports ---
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from operator import itemgetter
from typing import Optional

# --- LangChain Imports ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable
# -------------------------

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
CHROMA_PATH = "vectorstore"
COLLECTION_NAME = "wbs_schedules_db"

# --- 1. Pydantic Schemas ---

# 1a. Schema for a single WBS item (the lowest level output)
class WBSItem(BaseModel):
    """A single work package item in the Work Breakdown Structure."""
    code: str = Field(description="The WBS numerical code (e.g., 2.1.3)")
    name: str = Field(description="The name of the work package or task.")
    level: int = Field(description="The WBS hierarchy level (1, 2, or 3).")

# 1b. Schema for the complete WBS list (the final API output)
class WBSStructure(BaseModel):
    """The complete WBS structure for the project."""
    project_code: str = Field(description="The project's alphanumeric code, e.g., CIV-001.")
    project_name: str = Field(description="The full project title.")
    wbs_list: List[WBSItem] = Field(description="A list of all WBS items generated for the project.")

# 1c. Schema for the API input (data coming from the frontend)
class ProjectInput(BaseModel):
    """Input parameters provided by the user from the frontend."""
    project_code: str = Field(default="CIV-001")
    project_name: str = Field(default="Infrastructure Project")
    project_type: str = Field(description="e.g., 'bridge' or 'tunnel'")
    detail_level: str = Field(description="e.g., 'low', 'medium', or 'high'")
    # Optional type-specific parameters
    bridge_type: Optional[str] = None
    num_spans: Optional[int] = None
    num_piers: Optional[int] = None
    tunnel_type: Optional[str] = None
    num_tubes: Optional[int] = None
    tunnel_length: Optional[float] = None


# --- 2. Global RAG Components (Initialized on Startup) ---

# We use optional type hints and None as initial values to satisfy Python typing (Pylance)
LLM: Optional[ChatOpenAI] = None
RETRIEVER: Optional[Runnable] = None 


def initialize_rag_components():
    """Initializes the LLM and the persistent ChromaDB retriever."""
    global LLM, RETRIEVER
    
    try:
        # 1. Initialize the LLM
        LLM = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.0
        )

        # 2. Initialize Embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 3. Load the persistent ChromaDB 
        vector_store = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        
        # 4. Create the retriever (search_kwargs={"k": 7} fetches 7 most relevant chunks)
        RETRIEVER = vector_store.as_retriever(search_kwargs={"k": 7})
        
        print("LangChain RAG components initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize RAG components. Check your API key and network: {e}")
        # Setting them to None signals failure
        LLM = None
        RETRIEVER = None


def create_rag_chain() -> Runnable:
    """Creates and returns the core LangChain RAG chain."""
    
    # Safety Check: Ensures LLM and Retriever are initialized (addresses Pylance 'None' error)
    if LLM is None or RETRIEVER is None:
        raise RuntimeError("RAG components (LLM/RETRIEVER) were not initialized. Check API key/Vector DB path.")

    # 1. Define the System Prompt
    system_template = (
        "You are an expert construction project scheduler. Your task is to generate a complete Work Breakdown Structure (WBS) "
        "for the user's project based on the following rules: "
        "1. Strictly adhere to the WBS naming conventions, structure, and detail level from the provided 'CONTEXTUAL WBS SAMPLES'. "
        "2. Only include WBS items relevant to the requested Project Type and configuration parameters. "
        "3. **Your ENTIRE output MUST be a valid JSON object matching the requested schema.** Do not include any explanation or extra text outside the JSON."
        "\n\nCONTEXTUAL WBS SAMPLES: {context}"
    )

    # 2. Define the Human Prompt
    human_template = (
        "Generate the WBS for a project with the following parameters: "
        "Project Type: {project_type}, Name: {project_name}, Code: {project_code}. "
        "Configuration: Bridge Type={bridge_type}, Spans={num_spans}, Piers={num_piers}, Tunnel Type={tunnel_type}, Tubes={num_tubes}, Length={tunnel_length}. "
        "Detail Level Requested: {detail_level}. "
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template),
    ])

    # 3. Configure LLM for Structured Output (Fix for Error 1)
    # This automatically converts the Pydantic schema into function calling instructions for the LLM.
    llm_with_structured_output = LLM.with_structured_output(schema=WBSStructure)

    # 4. Define the RAG Chain using LCEL (Fix for Error 2)
    rag_chain = (
        # 4a. Maps the input fields to the required keys for the prompt and retriever
        {
            "context": itemgetter("query") | RETRIEVER,
            # Pass all other original inputs through for use in the prompt template
            "project_code": itemgetter("project_code"),
            "project_name": itemgetter("project_name"),
            "project_type": itemgetter("project_type"),
            "detail_level": itemgetter("detail_level"),
            "bridge_type": itemgetter("bridge_type"),
            "num_spans": itemgetter("num_spans"),
            "num_piers": itemgetter("num_piers"),
            "tunnel_type": itemgetter("tunnel_type"),
            "num_tubes": itemgetter("num_tubes"),
            "tunnel_length": itemgetter("tunnel_length"),
        }
        # 4b. Format the context and inputs into the final prompt
        | prompt
        # 4c. Generate the structured WBS using the LLM
        | llm_with_structured_output
    )
    
    return rag_chain


# --- 3. FastAPI Application ---

class Settings(BaseSettings):
    """Load settings from environment variables."""
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5500")

settings = Settings()

app = FastAPI(
    title="WBS Auto-Schedule Generator API",
    version="1.0.0",
    # Initialize RAG components when app starts
    on_startup=[initialize_rag_components] 
)

# CORS Middleware: Essential for the frontend (running on one port) to access the backend (running on another).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://127.0.0.1:5501", "http://127.0.0.1:5501"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. API Endpoint ---

@app.post("/generate_wbs", response_model=WBSStructure)
async def generate_wbs_api(input_data: ProjectInput):
    """
    Accepts project parameters and generates a structured WBS using the RAG chain.
    """
    
    if RETRIEVER is None:
        # This occurs if initialize_rag_components failed during startup
        raise Exception("RAG service is unavailable. Check backend setup and API keys.")

    # 1. Prepare the query for the RAG retriever
    retriever_query = (
        f"WBS for a {input_data.detail_level} detail {input_data.project_type} project. "
        f"Configuration: Type={input_data.bridge_type or input_data.tunnel_type}, Spans/Tubes={input_data.num_spans or input_data.num_tubes}."
    )
    
    # 2. Prepare the full input dictionary for the RAG chain
    llm_input = {
        "query": retriever_query, # Key used by the retriever to fetch context
        # All other keys are passed through to the prompt template
        "project_code": input_data.project_code,
        "project_name": input_data.project_name,
        "project_type": input_data.project_type,
        "detail_level": input_data.detail_level,
        "bridge_type": input_data.bridge_type,
        "num_spans": input_data.num_spans,
        "num_piers": input_data.num_piers,
        "tunnel_type": input_data.tunnel_type,
        "num_tubes": input_data.num_tubes,
        "tunnel_length": input_data.tunnel_length
    }
    
    # 3. Invoke the RAG chain
    rag_chain = create_rag_chain()
    
    # The output is a Pydantic object (WBSStructure)
    result = rag_chain.invoke(llm_input)
    
    return result

# --- Initial Test Route ---
@app.get("/")
def read_root():
    return {"status": "WBS RAG API is running"}


# --- How to Run the App (Used when running 'python app.py') ---
if __name__ == "__main__":
    import uvicorn
    # The port needs to be different from the one running your HTML/JS frontend
    print("Server starting at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
