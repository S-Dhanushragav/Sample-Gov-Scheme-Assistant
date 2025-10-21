import os
import pickle
import streamlit as st
import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
openai_base = os.getenv("OPENAI_API_BASE")
FAISS_PATH = os.path.join(os.getcwd(), "faiss_combined.pkl")

# Initialize LLM and embeddings
llm = ChatOpenAI(
    temperature=0.2,
    model="mistralai/mistral-7b-instruct",
    openai_api_key=openai_key,
    openai_api_base=openai_base
)
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

# Streamlit UI
st.set_page_config(page_title="Gov Scheme Assistant")
st.title("Sample Gov-Scheme Assistant")
st.sidebar.header("Upload Scheme URLs")
url_input = st.sidebar.text_area("Paste URLs (one by one):", height=150)
process_button = st.sidebar.button("Process URLs")

# Function to extract HTML or PDF content
def fetch_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        content_type = response.headers.get("Content-Type", "")

        if "pdf" in content_type or url.lower().endswith(".pdf"):
            with open("temp.pdf", "wb") as f:
                f.write(response.content)
            with fitz.open("temp.pdf") as doc:
                text = "\n".join([page.get_text() for page in doc])
            os.remove("temp.pdf")
            return text.strip()
        else:
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = soup.find_all("p")
            return "\n".join([p.get_text() for p in paragraphs]).strip()
    except Exception as e:
        return f"Error fetching content from {url}: {e}"

# Function to generate a structured 4-part summary
def generate_summary(text):
    prompt = f"""
Summarize this government scheme article into the following sections:
1. Scheme Benefits
2. Application Process
3. Eligibility
4. Documents Required

Text:
{text[:3000]}
"""
    try:
        response = llm.invoke(prompt)
        return response.content.strip() if hasattr(response, "content") else response
    except Exception as e:
        return f"Error generating summary: {e}"

# Processing section
if process_button:
    if not url_input.strip():
        st.warning("Please enter at least one URL.")
    else:
        urls = url_input.strip().splitlines()
        all_documents = []

        for url in urls:
            with st.spinner(f"Processing {url}..."):
                text = fetch_text_from_url(url)
                if not text or text.startswith("Error"):
                    st.error(text)
                    continue

                all_documents.append(Document(page_content=text, metadata={"source": url}))

                summary = generate_summary(text)
                st.markdown(f"### Summary for [{url}]({url})")
                st.markdown(f"```\n{summary}\n```")

        if not all_documents:
            st.error("No valid documents found.")
        else:
            chunks = splitter.split_documents(all_documents)
            faiss_index = FAISS.from_documents(chunks, embedding=embedder)
            with open(FAISS_PATH, "wb") as f:
                pickle.dump((faiss_index, all_documents), f)
            st.success(f"{len(all_documents)} documents processed and indexed.")

# Question Answering Section
st.subheader("Ask a Question")
query = st.text_input("Enter your question:")
ask_button = st.button("Get Answer")

# Sample questions
st.markdown("Sample Questions:")
st.markdown("- What are the benefits of the scheme?")
st.markdown("- How can someone apply?")
st.markdown("- Who is eligible?")
st.markdown("- What documents are required?")

if ask_button:
    if not os.path.exists(FAISS_PATH):
        st.warning("Please process URLs first.")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        with open(FAISS_PATH, "rb") as f:
            faiss_index, all_documents = pickle.load(f)

        top_docs = faiss_index.similarity_search(query, k=3)
        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=top_docs, question=query)

        st.markdown("Answer:")
        st.write(answer.strip())

        st.markdown("**Source URLs:**")
        sources = {doc.metadata.get("source", "Unknown") for doc in top_docs}
        for source in sources:
            st.markdown(f"- [{source}]({source})")

        st.success("Answer retrieved successfully.")



