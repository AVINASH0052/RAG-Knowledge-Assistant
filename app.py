# app.py
import os
import re
import operator
import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
DOCUMENT_DIR = "docs"
VECTOR_STORE_NAME = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize NVIDIA Client with environment variable
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]  # Make sure to set this in your environment
)

def load_and_chunk_documents():
    """Load and process documents with error handling"""
    if not os.path.exists(DOCUMENT_DIR):
        raise FileNotFoundError(f"Document directory '{DOCUMENT_DIR}' not found")

    documents = []
    for file in ["doc1.txt", "doc2.txt", "doc3.txt"]:
        file_path = os.path.join(DOCUMENT_DIR, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document {file} not found in {DOCUMENT_DIR}")
        
        loader = TextLoader(file_path)
        documents.extend(loader.load())

    return RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    ).split_documents(documents)

def initialize_vector_store():
    """Create or load vector store with safety override"""
    if os.path.exists(VECTOR_STORE_NAME):
        st.info("Loading existing vector store...")
        return FAISS.load_local(
            VECTOR_STORE_NAME,
            HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL),
            allow_dangerous_deserialization=True  # Required for loading existing index
        )
    
    st.info("Creating new vector store...")
    chunks = load_and_chunk_documents()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_STORE_NAME)
    return vector_store

def generate_answer(query, context):
    """Generate clean answers without XML tags"""
    system_prompt = f"""Answer the question based ONLY on this context. 
    Do NOT use markdown or XML tags. Be concise.
    
    Context: {context}"""
    
    try:
        response = nvidia_client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.6,
            max_tokens=1024
        )
        # Clean up any XML artifacts
        return response.choices[0].message.content.replace("<think>", "").replace("</think>", "").strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def math_calculator(query):
    """Secure calculation without numexpr"""
    try:
        allowed_ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}
        match = re.search(r'(\d+)([\+\-\*\/])(\d+)', query)
        if not match:
            return "Invalid expression format"
        num1, op, num2 = match.groups()
        return str(allowed_ops[op](int(num1), int(num2)))
    except Exception as e:
        return f"Calculation error: {str(e)}"

def term_definition(query):
    """Technical term lookup"""
    tech_terms = {
        "OLED": "Organic Light-Emitting Diode display technology",
        "IP68": "Ingress Protection rating for dust/water resistance",
        "5G": "Fifth-generation cellular network technology"
    }
    match = re.search(r"\b(define|what is|meaning of) (\w+)", query, re.IGNORECASE)
    return tech_terms.get(match.group(2).upper(), "Term not found") if match else "No term specified"

def route_query(query):
    """Improved query routing"""
    query = query.lower()
    if any(kw in query for kw in ["calculate", "compute", "+", "-", "*", "/"]):
        return "calculator", math_calculator(query)
    elif any(kw in query for kw in ["define", "what is", "meaning of"]):
        return "dictionary", term_definition(query)
    return "rag", None

def main():
    """Streamlit UI"""
    st.title("RAG-Powered Multi-Agent Q&A Assistant")
    
    try:
        vs = initialize_vector_store()
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        return

    query = st.text_input("Ask your question:")
    if not query:
        return

    tool, answer = route_query(query)
    context = None

    if tool == "rag":
        context_docs = vs.similarity_search(query, k=3)
        context = "\n---\n".join([doc.page_content for doc in context_docs])
        answer = generate_answer(query, context)

    st.subheader("Analysis")
    st.write(f"**Path Used:** {tool.upper()}")
    
    if context:
        with st.expander("View Context"):
            st.write(context)
    
    st.subheader("Answer")
    st.markdown(f"```\n{answer}\n```")

if __name__ == "__main__":
    main()
