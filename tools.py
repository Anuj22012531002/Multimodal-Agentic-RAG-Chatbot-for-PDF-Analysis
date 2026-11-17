import os
import io
import base64
from PIL import Image
from typing import List, Tuple, Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pymupdf
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma, chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

VECTORSTORE_INSTANCE = None

def get_embeddings(provider: str,api_key:str):
    """ Instantiate embeddings based on the chosen LLm Provider."""
    if provider == "Google_Gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model = "gemini-embedding-001",
            google_api_key = api_key
        )

    else:
        raise ValueError(f"Unsupported provider selected")




def process_and_index_pdfs(uploaded_files, provider:str,api_key:str):
    """
    Extracts text and image content from uploaded PDFs using PymuPDF
    and insdexes them into Chroma DB
    """
    global VECTORSTORE_INSTANCE
    docs = []

    os.makedirs("./temp_uploads", exist_ok=True)

    for uploaded_file in uploaded_files:
        file_path = os.path.join("./temp_uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract Text

        loader = PyPDFLoader(file_path)
        pdf_docs = loader.load()
        for doc in pdf_docs:
            doc.metadata["source_file"] = uploaded_file.name
            docs.append(doc)

        # Extract Embedding Images
        doc_pdf = pymupdf.open(file_path)
        for page_index in range(len(doc_pdf)):
            page = doc_pdf[page_index]
            image_list = page.get_images(full=True)

            for image_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc_pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]


                # convert image to base64 for vision processing
                base64_image = base64.b64encode(image_bytes).decode("utf-8")


                # Add image document entry
                image_doc = Document(
                    page_content=f"[Embedding Diagram/table/Image from {uploaded_file.name} - page{page_index + 1}] Base64 snippet:({base64_image[:80]})",
                    metadata={
                        "source_file": uploaded_file.name,
                        "page_number": page_index + 1,
                        "type": "image",
                        "base64_data": base64_image,
                        "mime_type": f"image/{image_ext}"

                    }
                ) 
                docs.append(image_doc)


    # chunk text documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)

    # index into chromaDB
    embeddings = get_embeddings(provider,api_key)
    VECTORSTORE_INSTANCE = Chroma.from_documents(
        documents=split_docs, 
        embedding=embeddings,
        collection_name="multimodel_RAG_collection"
    )
    return f"Succesfully processed and indexed {len(uploaded_files)} PDF(s)!"


@tool
def pdf_retrieval_tool(query: str):

    """
    Search and retrieve relevant chunks,table data,and image content 
    from uploaded PDFs based on the provided query using the indexed Chroma DB.

    """
    global VECTORSTORE_INSTANCE
    if VECTORSTORE_INSTANCE is None:
        raise ValueError("Vectorstore instance is not initialized. Please process and index PDFs first.")

    # Perform retrieval
    retriever = VECTORSTORE_INSTANCE.as_retriever(search_kwargs={"k": 5})
    results = retriever.invoke(query)

    # Format results
    formatted_results = []
    for  i,doc in enumerate(results):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page_number", "Unknown")
        doc_type = doc.metadata.get("type", "text")

        if doc_type == "image":
            b64 = doc.metadata.get("base64_data", "")
            formatted_results.append(
                f"[Result {i+1} Image/Table] (source: {source}, page: {page})\n"
                 f"Image BAse64 Data:data:{doc.metadata.get('mime_type')};base64,{b64}"
            )

        else:
            formatted_results.append(
                f"[Result {i+1} Text] (source: {source}, page: {page})\n"
                f"{doc.page_content}"
            )

    
    return "\n\n-------\n\n".join(formatted_results)


