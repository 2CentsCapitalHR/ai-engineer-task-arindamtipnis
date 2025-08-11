import os
import json
import time
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    _HAS_HF = True
except Exception:
    _HAS_HF = False
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import docx

# --- Environment and Configuration ---
# Ollama does not require an API key, but the service must be running.
# Make sure you have installed Ollama and pulled a model, e.g., `ollama pull llama3`

# Resolve data directory relative to this file (repo_root/data)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
DATA_DIR = os.path.join(REPO_ROOT, "data")
print(f"[RAG] Computed DATA_DIR: {DATA_DIR}")
PERSIST_DIR = os.path.join(REPO_ROOT, "vectorstore")
OLLAMA_MODEL = "llama3" # You can change this to another model like "mistral"
CHUNK_LIMIT = int(os.environ.get("RAG_CHUNK_LIMIT", "60"))  # hard cap chunks for speed
REBUILD_LIMIT_DOCS = int(os.environ.get("RAG_DOC_LIMIT", "8"))  # limit number of source docs

# --- Document Loading ---
def load_documents():
    """Load documents from the data directory."""
    documents = []
    
    if not os.path.isdir(DATA_DIR):
        print(f"Data directory does not exist: {DATA_DIR}")
        return []

    print(f"Loading documents from: {DATA_DIR}")
    try:
        text_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, use_multithreading=True)
        text_docs = text_loader.load()
        # Limit number of source docs aggressively for speed
        if len(text_docs) > REBUILD_LIMIT_DOCS:
            text_docs = text_docs[:REBUILD_LIMIT_DOCS]
        if not text_docs:
            print("No .txt documents found in data directory.")
        # Limit for speed
        text_docs = text_docs[:20]
        documents.extend(text_docs)
        print(f"Loaded {len(text_docs)} text documents (limited for speed)")
    except Exception as e:
        print(f"Warning: problem loading text files: {e}")

    print(f"Total documents loaded: {len(documents)}")
    return documents

# --- RAG Pipeline Initialization ---
def _vectorstore_path():
    return PERSIST_DIR

def get_rag_pipeline():
    """Initializes and returns the RAG pipeline using Ollama."""
    t0 = time.time()
    # If persisted vectorstore exists, load it directly
    if os.path.isdir(_vectorstore_path()) and any(
        n.startswith("index") for n in os.listdir(_vectorstore_path())
    ):
        try:
            print("Loading existing FAISS vector store from disk...")
            vectorstore = FAISS.load_local(_vectorstore_path(), embeddings=OllamaEmbeddings(model=OLLAMA_MODEL), allow_dangerous_deserialization=True)
            qa_pipeline = RetrievalQA.from_chain_type(
                llm=Ollama(model=OLLAMA_MODEL),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
            )
            print(f"Vector store loaded in {time.time()-t0:.2f}s")
            return qa_pipeline
        except Exception as e:
            print(f"Failed to load existing vector store, rebuilding. Reason: {e}")

    documents = load_documents()
    
    if not documents:
        print("No documents loaded. Please check the data directory.")
        return None

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    if len(texts) > CHUNK_LIMIT:
        texts = texts[:CHUNK_LIMIT]
        print(f"Split into {len(texts)} chunks (capped)")
    else:
        print(f"Split into {len(texts)} chunks")

    try:
        print(f"Initializing embeddings with Ollama model '{OLLAMA_MODEL}'...")
        max_retries = 2  # fewer retries to fail fast
        retry_delay = 3
        ollama_embeddings = None
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries} to connect to Ollama...")
                if _HAS_HF and os.environ.get("USE_HF_EMBED", "1") == "1":
                    try:
                        print("Using HuggingFaceEmbeddings (faster) ...")
                        ollama_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                    except Exception as embed_err:
                        print(f"HF embeddings unavailable ({embed_err}); falling back to OllamaEmbeddings.")
                        os.environ["USE_HF_EMBED"] = "0"
                        ollama_embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
                else:
                    ollama_embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
                test_llm = Ollama(model=OLLAMA_MODEL)
                test_llm.invoke("hello")
                print("Ollama connection successful.")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Connection failed: {e}. Retrying in {retry_delay}s")
                    time.sleep(retry_delay)
                else:
                    raise

        print("Creating FAISS vector store...")
        vectorstore = FAISS.from_documents(texts, ollama_embeddings)
        os.makedirs(_vectorstore_path(), exist_ok=True)
        vectorstore.save_local(_vectorstore_path())
        print(f"Vector store created & saved in {time.time()-t0:.2f}s")

        qa_pipeline = RetrievalQA.from_chain_type(
            llm=Ollama(model=OLLAMA_MODEL),
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
    
    # 1. Read the content of the uploaded .docx file BEFORE heavy init
    try:
        doc = docx.Document(doc_path)
        content = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading document {doc_path}: {e}")
        return {"error": f"Failed to read document: {e}"}

    short_len = len(content.strip())
    if short_len < 80:  # lower threshold early exit
        return {
            "process": "Too small for RAG",
            "documents_uploaded": 1,
            "required_documents": 0,
            "missing_documents": [],
            "issues_found": [],
            "note": f"Document very short ({short_len} chars); skipped RAG."
        }

    # Get the RAG pipeline (lazy initialization) only if needed
    pipeline = get_rag_pipeline_lazy()
    if not pipeline:
        return {"error": "RAG pipeline not initialized. Please check Ollama service."}

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
    def _parse_json(text: str):
        try:
            return json.loads(text)
        except Exception:
            # attempt to extract first JSON object/array
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                snippet = text[start:end+1]
                try:
                    return json.loads(snippet)
                except Exception:
                    pass
            return {"issues_found": [{"issue": "Failed to parse LLM response.", "raw_excerpt": text[:400]}]}

    try:
        query = f'{prompt}\n\nReturn ONLY the JSON object, with no other text.'
        print("Running RAG query...")
        t_q = time.time()
        result = pipeline.run(query)
        print(f"RAG query completed in {time.time()-t_q:.2f}s")
        analysis_json = _parse_json(result)
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
