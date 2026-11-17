# Multimodal Agentic RAG Chatbot for PDF Analysis

##  Overview
An AI-powered Multimodal Retrieval-Augmented Generation (RAG) chatbot that allows users to upload multiple PDF documents and ask context-aware questions. It retrieves relevant text, tables, and images using vector search and generates responses using Google Gemini.

##  Features
- **Multi-PDF Upload:** Upload and process multiple PDF documents.
- **Multimodal Extraction:** Extract text and images from PDFs using PyMuPDF.
- **Vector Search:** Generate embeddings and store document chunks in ChromaDB.
- **Agentic Workflow:** Use LangGraph for agent orchestration and tool calling.
- **Context-Aware Responses:** Retrieve relevant document content with source file and page references.
- **Interactive UI:** Streamlit-based chatbot interface.

##  Tech Stack
- **Programming Language:** Python
- **LLM:** Google Gemini
- **Frameworks:** LangChain, LangGraph
- **Vector Database:** ChromaDB
- **Embeddings:** Gemini Embeddings
- **PDF Processing:** PyMuPDF, PyPDFLoader
- **Frontend:** Streamlit

##  Project Workflow
1. Upload multiple PDF documents through Streamlit.
2. Extract text and images using PyPDFLoader and PyMuPDF.
3. Split documents into chunks using RecursiveCharacterTextSplitter.
4. Generate vector embeddings using Gemini Embeddings.
5. Store document vectors in ChromaDB.
6. Retrieve relevant chunks using similarity-based search.
7. LangGraph manages agent execution and PDF retrieval tool calling.
8. Google Gemini generates context-aware responses.

##  Project Structure

```text
Multimodal-Agentic-RAG/
│
├── app.py          # Streamlit application
├── agents.py       # LangGraph agent workflow
├── tools.py        # PDF processing and retrieval tools
├── temp_uploads/   # Temporary uploaded PDF files
├── requirements.txt
└── README.md
```

##  Installation & Setup

**1. Clone the repository**
```bash
git clone <your-repository-url>
cd Multimodal-Agentic-RAG
```

**2. Create virtual environment**
```bash
python -m venv venv
```

**3. Activate environment (Windows)**
```bash
venv\Scripts\activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

**5. Run the application**
```bash
streamlit run app.py
```

##  Configuration
- Select Google Gemini as the LLM provider.
- Enter your Google Gemini API key in the application sidebar.
- Upload one or more PDF documents.
- Process and index the PDFs before asking questions.

##  Use Cases
- Research paper analysis
- Document question answering
- Business report analysis
- Technical PDF analysis
- Multi-document information retrieval

##  Author
**Anuj Verma**

Data Analytics | Machine Learning | Generative AI | Agentic AI
