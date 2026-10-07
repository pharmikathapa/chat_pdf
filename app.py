import streamlit as st
import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please check your .env file.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📚 Chat with your PDF</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a PDF and ask questions about its content.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    st.divider()

    st.header("🎨 Answer Style")

    answer_style = st.selectbox(
        "How should I explain?",
        [
            "Normal",
            "Explain like I'm 5",
            "Bullet points",
            "Detailed",
            "Exam answer"
        ]
    )

    st.divider()

    if uploaded_file is not None:

        st.success("PDF uploaded successfully! ✅")

        st.write("**File:**")

        st.write(uploaded_file.name)

    else:

        st.info("Upload a PDF to start chatting.")


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vectorstore(uploaded_file):

    # Temporary PDF path
    pdf_path = "uploaded.pdf"

    # Save uploaded PDF
    with open(pdf_path, "wb") as f:

        f.write(uploaded_file.getbuffer())

    # Load PDF
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    # Split PDF into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    # Load embedding model
    embeddings = load_embeddings()

    # Create FAISS vector database
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


# ============================================================
# LOAD GEMINI
# ============================================================

@st.cache_resource
def load_llm():

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-3.8-flash",
        temperature=0.2
    )

    return llm


# ============================================================
# ANSWER QUESTION
# ============================================================

def answer_question(question, vectorstore, answer_style):

    # --------------------------------------------------------
    # RETRIEVE RELEVANT DOCUMENTS
    # --------------------------------------------------------

    documents = vectorstore.similarity_search(
        question,
        k=4
    )

    # --------------------------------------------------------
    # CREATE CONTEXT
    # --------------------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # --------------------------------------------------------
    # SELECT ANSWER STYLE
    # --------------------------------------------------------

    if answer_style == "Explain like I'm 5":

        style_instruction = """
Explain the answer in extremely simple language,
as if you are explaining it to a 5-year-old.

Use:
- Very simple words
- Easy examples
- Short sentences

Avoid unnecessary technical jargon.
"""

    elif answer_style == "Bullet points":

        style_instruction = """
Answer using clear bullet points.

Rules:
- Use short bullet points
- Keep the explanation easy to scan
- Highlight important information
- Do not use unnecessarily long paragraphs
"""

    elif answer_style == "Detailed":

        style_instruction = """
Give a detailed explanation.

Explain:
- What the concept means
- How it works
- Important details
- Examples when useful

Organize the answer clearly.
"""

    elif answer_style == "Exam answer":

        style_instruction = """
Give an exam-ready answer.

Structure the answer as:

1. Definition
2. Main explanation
3. Important points
4. Example if relevant

Use simple technical language.
Make the answer easy to write in an exam.
"""

    else:

        style_instruction = """
Give a clear and concise answer.

Use simple language and explain the concept
accurately using the provided context.
"""

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt_template = """
You are an AI assistant that answers questions about a PDF.

IMPORTANT RULE:

Answer the question using ONLY the provided context.

Do NOT use outside knowledge.

If the answer is not present in the context,
say exactly:

"The answer is not available in the provided context."

{style_instruction}

--------------------------------------------------

CONTEXT:

{context}

--------------------------------------------------

QUESTION:

{question}

--------------------------------------------------

ANSWER:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=[
            "context",
            "question",
            "style_instruction"
        ]
    )

    # --------------------------------------------------------
    # CREATE FINAL PROMPT
    # --------------------------------------------------------

    final_prompt = prompt.format(
        context=context,
        question=question,
        style_instruction=style_instruction
    )

    # --------------------------------------------------------
    # CALL GEMINI
    # --------------------------------------------------------

    llm = load_llm()

    response = llm.invoke(final_prompt)

    # --------------------------------------------------------
    # EXTRACT CLEAN TEXT
    # --------------------------------------------------------

    if isinstance(response.content, list):

        answer = "".join(
            block["text"]
            for block in response.content
            if isinstance(block, dict)
            and block.get("type") == "text"
        )

    else:

        answer = response.content

    return answer


# ============================================================
# INITIALIZE CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# INITIALIZE VECTORSTORE
# ============================================================

if "vectorstore" not in st.session_state:

    st.session_state.vectorstore = None


if "uploaded_filename" not in st.session_state:

    st.session_state.uploaded_filename = None


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    # Check if this is a new PDF
    if (
        st.session_state.uploaded_filename
        != uploaded_file.name
    ):

        with st.spinner("📖 Processing your PDF..."):

            try:

                vectorstore = create_vectorstore(
                    uploaded_file
                )

                st.session_state.vectorstore = vectorstore

                st.session_state.uploaded_filename = (
                    uploaded_file.name
                )

                # Clear old chat when a new PDF is uploaded
                st.session_state.messages = []

                st.success(
                    "PDF processed successfully! ✅"
                )

            except Exception as e:

                st.error(
                    f"Error processing PDF: {e}"
                )

                st.stop()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your PDF..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # CHECK PDF
    # --------------------------------------------------------

    if st.session_state.vectorstore is None:

        st.warning(
            "⚠️ Please upload a PDF first."
        )

        st.stop()

    # --------------------------------------------------------
    # DISPLAY USER QUESTION
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("🤔 Thinking..."):

            try:

                answer = answer_question(
                    question,
                    st.session_state.vectorstore,
                    answer_style
                )

                st.markdown(answer)

            except Exception as e:

                answer = (
                    "Sorry, I couldn't generate an answer.\n\n"
                    f"Error: {e}"
                )

                st.error(answer)

    # --------------------------------------------------------
    # SAVE ANSWER
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )