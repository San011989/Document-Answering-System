import os
import shutil
import tempfile
from io import BytesIO

import streamlit as st
from langchain_core.documents import Document
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import pdfplumber
import fitz
import pytesseract
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


if not shutil.which("tesseract"):
    default_tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.isfile(default_tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = default_tesseract_path


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Document Question with RAG",
    page_icon="📖",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }
    .answer-type {
        padding: 8px 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    """
    <div class="main-title">
        📖 Document Question with RAG
    </div>
    <div class="subtitle">
        Ask questions strictly based on the uploaded document.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# CHECK OPENAI API KEY
# =========================================================

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not configured.")
    st.info(
        "Create a .env file in your project folder and add:\n\n"
        "OPENAI_API_KEY=your_api_key_here"
    )
    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_file_processed" not in st.session_state:
    st.session_state.is_file_processed = False

if "selected_file_key" not in st.session_state:
    st.session_state.selected_file_key = None


# =========================================================
# LOAD CHROMA VECTOR DATABASE
# =========================================================

@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    return vectorstore


try:
    vectorstore = load_vectorstore()
except Exception as e:
    st.error("Failed to load Chroma vector database. Ensure persistent path matches './chroma_db'.")
    st.exception(e)
    st.stop()


# =========================================================
# CREATE LLM
# =========================================================

@st.cache_resource
def create_llm(temp):
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temp
    )


try:
    llm = create_llm(0.0)
except Exception as e:
    st.error("Failed to initialize OpenAI model.")
    st.exception(e)
    st.stop()


# =========================================================
# PROMPTS & CHAINS
# =========================================================

# Updated system prompt allowing partial context synthesis
rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful QA assistant that answers questions using the provided document context.

Rules:
1. Use facts from the provided context to answer the user's question.
2. If the document mentions the topic but does not provide a full definition, explain what the document specifically says about it.
3. If the context contains no relevant information at all, state: "I cannot answer this question based on the provided document."

Context:
{context}"""
        ),
        ("human", "{question}")
    ]
)

rag_chain = rag_prompt | llm | StrOutputParser()


# =========================================================
# HELPER FOR PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_documents(file_path, source_name):
    documents = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": source_name, "page": i}
                    )
                )

    if documents:
        return documents

    # Scanned PDFs have no text layer, so render each page and OCR the image.
    try:
        with fitz.open(file_path) as pdf:
            for i, page in enumerate(pdf):
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = pixmap.tobytes("png")
                text = pytesseract.image_to_string(Image.open(BytesIO(image)))
                if text and text.strip():
                    documents.append(
                        Document(
                            page_content=text.strip(),
                            metadata={"source": source_name, "page": i, "ocr": True}
                        )
                    )
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError(
            "This PDF appears to be scanned. Install the Tesseract OCR engine "
            "and ensure tesseract.exe is on PATH, then try again."
        ) from error

    return documents


# =========================================================
# SIDEBAR - PDF UPLOAD & DATABASE MANAGEMENT
# =========================================================

with st.sidebar:
    st.header("📄 PDF Upload & Database")
    
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    current_file_key = None

    if uploaded_file is not None:
        current_file_key = f"{uploaded_file.name}_{uploaded_file.size}"

        if st.session_state.selected_file_key != current_file_key:
            st.session_state.selected_file_key = current_file_key
            st.session_state.is_file_processed = False
            st.session_state.messages = []

        if st.button("Process and Add to Database", use_container_width=True):
            st.session_state.is_file_processed = False

            with st.spinner("Processing PDF and generating overview..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # Use robust text extraction
                    documents = extract_pdf_documents(tmp_path, uploaded_file.name)

                    if not documents:
                        st.error("No readable text found in PDF. If this is an image/scanned PDF, OCR is required.")
                        st.stop()

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = text_splitter.split_documents(documents)

                    if chunks:
                        # Clear old collection on new file upload
                        try:
                            vectorstore.delete_collection()
                            st.cache_resource.clear()
                            vectorstore = load_vectorstore()
                        except Exception:
                            pass

                        vectorstore.add_documents(chunks)

                        sample_content = "\n\n".join(
                            [doc.page_content for doc in documents[:5]]
                        )

                        summary_prompt = (
                            "Provide a comprehensive overview of what this document contains, "
                            "including its main topics, target subject, and key themes based on these initial pages:\n\n"
                            f"{sample_content}"
                        )

                        overview_text = llm.invoke(summary_prompt).content

                        summary_document = Document(
                            page_content=(
                                f"DOCUMENT OVERVIEW AND SUMMARY:\n"
                                f"File Name: {uploaded_file.name}\n\n"
                                f"What this document is about:\n{overview_text}"
                            ),
                            metadata={
                                "source": uploaded_file.name,
                                "type": "overview"
                            }
                        )

                        vectorstore.add_documents([summary_document])

                        st.session_state.is_file_processed = True

                        st.success(
                            f"Successfully processed {len(chunks)} chunks + "
                            "added document overview!"
                        )
                        st.rerun()

                    else:
                        st.session_state.is_file_processed = False
                        st.warning("No text found in the PDF.")

                except Exception as e:
                    st.session_state.is_file_processed = False
                    st.error(f"Error processing PDF: {e}")

                finally:
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.remove(tmp_path)

    if st.button("🗑️ Clean Database", use_container_width=True):
        try:
            vectorstore.delete_collection()
            st.cache_resource.clear()

            st.session_state.is_file_processed = False
            st.session_state.selected_file_key = None
            st.session_state.messages = []

            st.success("Vector database cleared completely!")
            st.rerun()
        except Exception as e:
            st.error(f"Error cleaning database: {e}")

    st.divider()

    st.header("⚙️ Settings")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Lower values produce focused answers. Higher values produce creative answers."
    )

    st.write(f"Current temperature: **{temperature:.1f}**")

    st.divider()

    st.header("📚 RAG Settings")

    top_k = st.slider(
        "Documents to retrieve",
        min_value=1,
        max_value=10,
        value=4,
        step=1
    )

    relevance_threshold = st.slider(
        "Relevance threshold (Max Distance)",
        min_value=0.1,
        max_value=3.0,
        value=2.0,
        step=0.1,
        help="Chroma distance score. Documents with distance GREATER than this threshold are ignored."
    )

    st.caption("Lower distance = higher document similarity.")

    st.divider()

    st.header("💬 Chat History")

    if len(st.session_state.messages) == 0:
        st.info("No conversations yet.")
    else:
        question_number = 1
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"**{question_number}.** {message['content']}")
                question_number += 1

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def format_documents(docs):
    formatted_documents = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source", "Unknown source")
        page = metadata.get("page", None)

        if page is not None:
            source_text = f"Source: {source}, Page: {page + 1}"
        else:
            source_text = f"Source: {source}"

        formatted_documents.append(
            f"Document {index}\n{source_text}\nContent:\n{doc.page_content}"
        )

    return "\n\n".join(formatted_documents)


def retrieve_documents(question, k, threshold):
    results = vectorstore.similarity_search_with_score(question, k=k)
    
    relevant_docs = []
    scores = []

    for doc, score in results:
        scores.append(score)
        if score <= threshold:
            relevant_docs.append(doc)

    return relevant_docs, scores


def is_overview_query(question: str) -> bool:
    overview_keywords = [
        "what is in", "what is inside", "summarize", "summary", 
        "overview", "tell me about", "what does the file", 
        "what is the document about", "what is in doc", "about the document",
        "summary of file", "summary of the file", "file summary", "document summary"
    ]
    q_lower = question.lower()
    return any(keyword in q_lower for keyword in overview_keywords)


def get_stored_overview_doc():
    """Directly queries Chroma for documents tagged with type='overview'."""
    try:
        results = vectorstore.get(where={"type": "overview"})
        if results and results.get("documents"):
            overview_docs = []
            for i, text in enumerate(results["documents"]):
                meta = results["metadatas"][i] if results.get("metadatas") else {}
                overview_docs.append(Document(page_content=text, metadata=meta))
            return overview_docs
    except Exception:
        pass
    return []


def generate_answer(question, k, threshold):
    docs = []
    scores = []

    if is_overview_query(question):
        # Fetch the overview chunk directly via metadata filter
        docs = get_stored_overview_doc()
        
        # Fallback to similarity search if metadata match fails
        if not docs:
            docs, scores = retrieve_documents(question, k, threshold)
        else:
            scores = [0.0] * len(docs)
    else:
        docs, scores = retrieve_documents(question, k, threshold)

    if len(docs) > 0:
        context = format_documents(docs)
        answer = rag_chain.invoke({"context": context, "question": question})

        return {
            "answer": answer,
            "source_type": "📚 RAG — Uploaded Documents",
            "documents": docs,
            "scores": scores
        }

    return {
        "answer": "I cannot answer this question based on the provided document.",
        "source_type": "⚠️ No Relevant Context Found",
        "documents": [],
        "scores": scores
    }

# =========================================================
# DISPLAY PREVIOUS CHAT MESSAGES
# =========================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "source_type" in message:
            st.caption(message["source_type"])


# =========================================================
# CONTROL CHAT INPUT
# =========================================================

is_file_processed = st.session_state.is_file_processed

if not is_file_processed:
    st.info(
        "💡 Please upload a PDF and click "
        "'Process and Add to Database' to enable chat."
    )

query = st.chat_input(
    placeholder=(
        "Ask a question about the uploaded document..."
        if is_file_processed
        else "🔒 Process the PDF first to enable chat..."
    ),
    disabled=not is_file_processed
)

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🔎 Searching documents..."):
            try:
                result = generate_answer(
                    question=query,
                    k=top_k,
                    threshold=relevance_threshold
                )

                answer = result["answer"]
                source_type = result["source_type"]
                documents = result["documents"]
                scores = result["scores"]

                st.caption(source_type)
                st.markdown(answer)

                with st.expander("🛠️ Debug Info & Retrieved Documents"):
                    st.write(f"**Raw Distance Scores retrieved:** {scores}")
                    st.write(f"**Threshold set to:** {relevance_threshold}")
                    st.write(f"**Passed filtering:** {len(documents)} document(s)")

                    for i, doc in enumerate(documents, start=1):
                        metadata = doc.metadata or {}
                        source = metadata.get("source", "Unknown")
                        page = metadata.get("page", None)
                        page_str = f", Page {page + 1}" if page is not None else ""

                        st.markdown(f"**Document {i}** — {source}{page_str}")
                        st.write(doc.page_content)
                        if i < len(documents):
                            st.divider()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "source_type": source_type
                    }
                )

            except Exception as e:
                st.error("An error occurred while generating the answer.")
                st.exception(e)