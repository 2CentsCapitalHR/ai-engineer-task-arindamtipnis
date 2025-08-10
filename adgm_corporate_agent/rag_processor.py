import os
import json
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import docx

# --- Environment and Configuration ---
# Ollama does not require an API key, but the service must be running.
# Make sure you have installed Ollama and pulled a model, e.g., `ollama pull llama3`

DATA_DIR = "../data"  # Point to the parent data directory where documents are stored
PERSIST_DIR = "db"
OLLAMA_MODEL = "llama3" # You can change this to another model like "mistral"

# --- Document Loading ---
def load_documents():
    """Load documents from the data directory."""
    documents = []
    
    try:
        # First try to load text files which are most reliable (limit to first 5 for speed)
        text_loader = DirectoryLoader(
            DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, use_multithreading=True
        )
        try:
            text_docs = text_loader.load()
            # Limit to first 5 text documents for faster processing
            text_docs = text_docs[:5]
            documents.extend(text_docs)
            print(f"Loaded {len(text_docs)} text documents (limited for speed)")
        except Exception as e:
            print(f"Warning: Could not load text documents: {e}")
        
        # Skip PDF and DOCX for now to make it faster
        print("Skipping PDF and DOCX documents for faster startup...")
        
        print(f"Total documents loaded: {len(documents)}")
        return documents
        
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

# --- RAG Pipeline Initialization ---
def get_rag_pipeline():
    """Initializes and returns the RAG pipeline using Ollama."""
    documents = load_documents()
    
    if not documents:
        print("No documents loaded. Please check the data directory.")
        return None

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)

    try:
        # Create embeddings and vector store using Ollama
        print(f"Initializing embeddings with Ollama model '{OLLAMA_MODEL}'...")
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
        vectorstore = FAISS.from_documents(texts, embeddings)
        print("Vector store created successfully.")

        # Create the RetrievalQA chain with ChatOllama
        qa_pipeline = RetrievalQA.from_chain_type(
            llm=ChatOllama(model=OLLAMA_MODEL, format="json", temperature=0),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
        )
        return qa_pipeline
    except Exception as e:
        print(f"Error initializing RAG pipeline: {e}")
        return None

# Initialize the pipeline globally to avoid reloading on every call
# This might take a moment as it loads the model and data.
print("RAG pipeline will be initialized when first needed...")
rag_pipeline = None

def get_rag_pipeline_lazy():
    """Lazy initialization of RAG pipeline - only when needed."""
    global rag_pipeline
    if rag_pipeline is None:
        print("Initializing RAG pipeline... This may take a few moments.")
        rag_pipeline = get_rag_pipeline()
        if rag_pipeline:
            print("RAG pipeline initialized successfully.")
        else:
            print("Failed to initialize RAG pipeline.")
    return rag_pipeline

def analyze_with_rag(doc_path, uploaded_files_info):
    """
    Analyzes a document using the RAG-powered LLM, including checklist verification.
    """
    print(f"Analyzing {doc_path} with RAG...")
    
    # Get the RAG pipeline (lazy initialization)
    pipeline = get_rag_pipeline_lazy()
    if not pipeline:
        return {"error": "RAG pipeline not initialized. Please check Ollama service."}

    # 1. Read the content of the uploaded .docx file
    try:
        doc = docx.Document(doc_path)
        content = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading document {doc_path}: {e}")
        return {"error": f"Failed to read document: {e}"}

    # 2. Formulate a detailed prompt for the LLM
    prompt = f"""
    You are an expert Corporate Agent specializing in ADGM regulations.
    Your task is to review the following legal document and identify any potential red flags,
    non-compliance issues, or inconsistencies based on the ADGM knowledge base.

    Document Content to Analyze:
    ---
    {content}
    ---

    Based on the provided ADGM legal and regulatory context, please perform the following:
    1.  Identify the document type (e.g., Articles of Association, Employment Contract).
    2.  Detect any clauses that are incorrect, missing, or non-compliant with ADGM rules.
        For example, check for incorrect jurisdiction references (e.g., mentioning UAE Federal Courts instead of ADGM Courts).
    3.  For each issue found, provide the specific clause or section, a description of the issue,
        its severity (High, Medium, Low), and a concrete suggestion for correction, citing the relevant ADGM rule if possible.

    Return your findings in a structured JSON format. The JSON object should have a key "issues_found"
    which is a list of objects, where each object contains: 'document', 'section', 'issue', 'severity', and 'suggestion'.
    If no issues are found, return an empty list for "issues_found".

    Example of a finding:
    {{
      "document": "Articles of Association",
      "section": "Clause 3.1",
      "issue": "Jurisdiction clause does not specify ADGM",
      "severity": "High",
      "suggestion": "Update jurisdiction to ADGM Courts. Per ADGM Companies Regulations 2020, Art. 6..."
    }}

    Now, provide your analysis as a JSON object:
    """

    # 3. Execute the RAG chain
    try:
        # The prompt needs to be slightly different for some local models
        # to ensure they stick to JSON output.
        query = f'{prompt}\n\nReturn ONLY the JSON object, with no other text or explanation.'
        result = pipeline.run(query)
        analysis_json = json.loads(result)
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from LLM response.")
        print(f"LLM Raw Response: {result}")
        analysis_json = {"issues_found": [{"issue": "Failed to parse LLM response.", "suggestion": "Check logs."}]}
    except Exception as e:
        print(f"An error occurred during RAG pipeline execution: {e}")
        return {"error": str(e)}

    # 4. Dynamic Document Checklist Verification
    # This is a more advanced checklist that knows which documents are required for which process.
    process_checklists = {
        "Company Incorporation": [
            "Articles of Association",
            "Memorandum of Association",
            "Board Resolution",
            "Incorporation Application",
            "UBO Declaration",
            "Register of Members and Directors",
        ],
        "Employment Agreement": ["Employment Contract"],
    }

    # For now, we'll assume the process is "Company Incorporation" if an AoA is present.
    # A more sophisticated approach might ask the user or infer from the documents.
    uploaded_doc_types = {info["type"] for info in uploaded_files_info}
    process = "Company Incorporation" if "Articles of Association" in uploaded_doc_types else "Unknown Process"

    missing_documents = []
    if process in process_checklists:
        required_docs = set(process_checklists[process])
        missing_documents = list(required_docs - uploaded_doc_types)

    # 5. Combine analysis with checklist results
    final_output = {
        "process": process,
        "documents_uploaded": len(uploaded_doc_types),
        "required_documents": len(process_checklists.get(process, [])),
        "missing_documents": missing_documents,
        "issues_found": analysis_json.get("issues_found", [])
    }

    # Save the JSON report
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    report_path = os.path.join(outputs_dir, f"report_{os.path.basename(doc_path)}.json")
    with open(report_path, 'w') as f:
        json.dump(final_output, f, indent=2)

    return final_output
