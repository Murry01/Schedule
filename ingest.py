import os
import pandas as pd
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List

# --- Configuration ---
load_dotenv()
# Assumes OPENAI_API_KEY is set in .env
CHROMA_PATH = "vectorstore"
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "wbs_schedules_db"

# --- 1. Data Loading and Document Preparation ---

def load_and_transform_wbs_data(file_paths: List[str]) -> List[Document]:
    """Loads CSV files and transforms WBS rows into LangChain Documents."""
    documents = []
    
    # Define columns to use for embedding (the actual content)
    content_columns = ['WBS_Name', 'WBS_Level']
    
    # Columns to keep as metadata (for retrieval filters and context)
    metadata_columns = ['Project_Type', 'Bridge_Type', 'Tunnel_Type', 'Spans', 'Piers', 'Tubes', 'Length_km', 'Detail_Level', 'WBS_Code']
    
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path).fillna('')
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            continue

        for _, row in df.iterrows():
            # 1. Create content string for embedding (what the LLM will 'read')
            # Combine WBS name with the project's key features for strong semantic context
            if row['Project_Type'] == 'Bridge':
                features = f"Bridge Type: {row['Bridge_Type']}, Spans: {row['Spans']}, Piers: {row['Piers']}, Detail: {row['Detail_Level']}"
            elif row['Project_Type'] == 'Tunnel':
                features = f"Tunnel Type: {row['Tunnel_Type']}, Tubes: {row['Tubes']}, Length (km): {row['Length_km']}, Detail: {row['Detail_Level']}"
            else:
                features = ""

            page_content = f"{row['WBS_Name']} (Level {row['WBS_Level']}) - Features: {features}"

            # 2. Create metadata dictionary
            metadata = {col: row[col] for col in metadata_columns if col in row}
            metadata['source'] = os.path.basename(file_path)

            documents.append(
                Document(page_content=page_content, metadata=metadata)
            )
            
    return documents

# --- 2. Chunking (Splitting into smaller searchable units) ---

def split_documents(documents: List[Document]) -> List[Document]:
    """Splits documents into smaller, contextually relevant chunks."""
    
    # We use a large chunk size with zero overlap because WBS rows are discrete, 
    # but combining them slightly ensures contextual flow. 
    # NOTE: Since each row is a Doc, this primarily serves to process large text 
    # if our WBS_Name columns were paragraphs (not needed here, but kept as a best practice splitter)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150, 
        chunk_overlap=0,
        separators=["\n\n", "\n", ".", " "]
    )
    return text_splitter.split_documents(documents)

# --- 3. Embedding and Storage ---

def main():
    print("--- Starting WBS Vector DB Ingestion ---")
    
    # 1. Load and Transform Data
    file_paths = ["data/bridge_schedules.csv", "data/tunnel_schedules.csv"]
    raw_documents = load_and_transform_wbs_data(file_paths)
    print(f"Loaded {len(raw_documents)} WBS entries.")
    
    # 2. Split Documents
    chunks = split_documents(raw_documents)
    print(f"Split into {len(chunks)} searchable chunks.")

    # 3. Initialize Embedding Model
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    # 4. Create and Persist Vector Store
    # Persistence is handled automatically because persist_directory is set
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME
    )
    
    # 5. Finalize - Remove the explicit .persist() call
    print("WBS Vector Database built and saved successfully! 🎉")

    # You can optionally return the vector_store object for later use if needed
    # return vector_store 


if __name__ == "__main__":
    main()