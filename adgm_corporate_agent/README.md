# ADGM-Compliant Corporate Agent

This project is an intelligent AI-powered legal assistant to help with ADGM business incorporation and compliance. It runs entirely locally using Ollama and a compatible open-source language model.

## Setup

1.  **Install Ollama:**
    -   Go to [https://ollama.com/](https://ollama.com/) and download the application for your operating system (macOS, Windows, or Linux).
    -   Run the installer.

2.  **Download a Language Model:**
    -   Open your terminal and run the following command to download the `llama3` model. This is the recommended model for this project.
    ```bash
    ollama pull llama3
    ```
    -   Ollama will automatically start running in the background.

3.  **Create a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4.  **Install dependencies:**
    -   First, upgrade pip to the latest version:
    ```bash
    pip install --upgrade pip
    ```
    -   Then, install the required packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Ingest Legal Data:**
    -   Before running the application for the first time, you need to download the legal reference documents.
    ```bash
    python3 data_ingestion.py
    ```
    -   This will download all the required documents into the `data/` directory.

## How to Run

1.  **Ensure Ollama is Running:**
    -   The Ollama application should be running in the background. You can check this by seeing its icon in your system's menu bar or taskbar.

2.  **Start the Gradio application:**
    ```bash
    python3 app.py
    ```
    -   The first time you run this, it will take a few moments to initialize the RAG pipeline and create the embeddings. Subsequent runs will be faster.

3.  **Open your browser:**
    -   Navigate to the local URL provided by Gradio (usually `http://127.0.0.1:7860`).

4.  **Upload Documents:**
    -   Upload one or more `.docx` files for analysis.

5.  **Analyze:**
    -   Click the "Analyze Documents" button to start the process. The application will provide a JSON report and a downloadable, reviewed `.docx` file with comments.

## Web App Functionality

The ADGM Corporate Agent provides a comprehensive web interface for analyzing legal documents and ensuring ADGM compliance. Here's what you can do:

### 🚀 **Core Features**

- **Document Upload & Processing**
  - Upload multiple `.docx` files simultaneously
  - Automatic document type detection and parsing
  - Support for complex legal documents with tables, formatting, and structure

- **AI-Powered Analysis**
  - **RAG (Retrieval-Augmented Generation)** pipeline using local Ollama models
  - Intelligent document analysis against ADGM legal framework
  - Context-aware legal compliance checking

- **Compliance Verification**
  - Automatic identification of ADGM regulatory requirements
  - Cross-referencing with current legal frameworks
  - Detection of missing or incomplete compliance elements

### 📊 **Analysis Output**

The application generates comprehensive analysis reports including:

- **JSON Analysis Report**
  - Document type identification
  - Compliance status assessment
  - Missing requirements identification
  - Risk assessment and recommendations
  - Regulatory framework references

- **Enhanced Document**
  - Original document with embedded AI comments
  - Compliance suggestions and improvements
  - Missing information highlights
  - Best practice recommendations

### 🎯 **Use Cases**

- **Business Incorporation**
  - Review incorporation documents for ADGM compliance
  - Identify missing shareholder information
  - Verify corporate structure requirements
  - Check regulatory compliance

- **Employment Contracts**
  - Validate employment contract templates
  - Ensure ADGM employment regulations compliance
  - Identify missing clauses or requirements
  - Suggest improvements for legal compliance

- **Annual Filings & Reports**
  - Review annual accounts and filings
  - Verify regulatory reporting requirements
  - Identify compliance gaps
  - Ensure timely submission requirements

- **Branch Operations**
  - Validate branch establishment documents
  - Check non-financial services compliance
  - Verify licensing requirements
  - Ensure operational compliance

### 🔧 **Technical Capabilities**

- **Local Processing**
  - Runs entirely on your local machine
  - No data sent to external servers
  - Privacy-focused document analysis
  - Offline capability

- **Smart Document Handling**
  - Automatic text extraction from `.docx` files
  - Preservation of document formatting
  - Intelligent comment insertion
  - Batch processing capabilities

- **Advanced AI Integration**
  - Local Ollama language model integration
  - FAISS vector database for efficient document search
  - Semantic understanding of legal requirements
  - Context-aware analysis

### 📁 **File Management**

- **Upload Directory**: `uploads/` - Temporary storage for user files
- **Output Directory**: `outputs/` - Generated reports and enhanced documents
- **Database Directory**: `db/` - FAISS vector database for document search
- **Data Directory**: `data/` - ADGM legal framework documents

### 🚦 **User Interface**

The Gradio web interface provides:

- **Clean, intuitive design** for easy document upload
- **Real-time processing status** updates
- **Downloadable results** in multiple formats
- **Error handling** with helpful messages
- **Responsive design** for various screen sizes

### 🔒 **Security & Privacy**

- **Local Processing**: All document analysis happens on your machine
- **No External Transmission**: Documents never leave your system
- **Secure Storage**: Temporary files are managed securely
- **Privacy Compliance**: Meets enterprise security requirements

## Project Structure

- `app.py`: The main Gradio application file.
- `document_processor.py`: Handles `.docx` file parsing and comment insertion.
- `rag_processor.py`: Contains the core RAG logic using Ollama for document analysis.
- `data_ingestion.py`: Script to download and prepare the legal data.
- `requirements.txt`: Lists all Python dependencies for the project.
- `data/`: Stores the ingested legal knowledge base.
- `uploads/`: Temporarily stores user-uploaded files.
- `outputs/`: Stores the final analysis reports and reviewed documents.
- `db/`: (Will be created on first run) Stores the FAISS vector database.
