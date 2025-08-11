import gradio as gr
import os
import shutil
from document_processor import process_document, identify_document_type
from rag_processor import analyze_with_rag

def process_files(files):
    """
    Gradio function to handle file uploads, processing, and displaying results.
    """
    if not files:
        return "Please upload documents.", "", None

    # Create uploads directory if it doesn't exist
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)

    # Save uploaded files and identify their types
    uploaded_files_info = []
    for temp_file_path in files:
        file_name = os.path.basename(temp_file_path)
        file_path = os.path.join(uploads_dir, file_name)
        shutil.copy(temp_file_path, file_path)

        doc_type = identify_document_type(file_path)
        uploaded_files_info.append({"path": file_path, "type": doc_type})

    # For this version, we'll still focus the deep analysis on the first document,
    # but the context of all uploaded documents is passed to the RAG processor.
    main_document_path = uploaded_files_info[0]["path"]

    # Process the document with RAG, now with document types
    analysis_result = analyze_with_rag(main_document_path, uploaded_files_info)

    # Add comments to the document based on analysis
    output_doc_path = process_document(main_document_path, analysis_result)

    # Clean up uploaded files
    for file_info in uploaded_files_info:
        os.remove(file_info["path"])

    return (
        f"Processed {len(uploaded_files_info)} document(s).",
        analysis_result,
        output_doc_path,
    )

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# ADGM-Compliant Corporate Agent")
    gr.Markdown(
        "Upload your legal documents (.docx) to review them for compliance with ADGM regulations."
    )

    with gr.Row():
        file_upload = gr.File(
            label="Upload .docx Documents", file_count="multiple", type="filepath"
        )

    process_button = gr.Button("Analyze Documents")

    with gr.Row():
        with gr.Column():
            status_output = gr.Textbox(label="Status")
            json_output = gr.JSON(label="Analysis Report")
        with gr.Column():
            download_file = gr.File(label="Download Reviewed Document")

    process_button.click(
        fn=process_files,
        inputs=[file_upload],
        outputs=[status_output, json_output, download_file],
    )

if __name__ == "__main__":
    demo.launch()
