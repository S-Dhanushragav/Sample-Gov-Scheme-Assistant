# Scheme Research Assistant

This is a Streamlit-based web application that automates the extraction, summarization, and question-answering process for government scheme documents and articles. Users can input scheme URLs (HTML or PDF), generate concise summaries, and ask natural language questions to retrieve information using an LLM-powered QA system.

## 🔍 Features

- Accepts both web page URLs and PDF URLs
- Automatically extracts and processes text using `BeautifulSoup` and `PyMuPDF`
- Generates structured summaries (Benefits, Application Process, Eligibility, Required Documents)
- Enables users to ask questions using LLM and FAISS-based document retrieval
- Highlights source URLs for transparency

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- HuggingFace Embeddings
- FAISS (Vector Store)
- PyMuPDF
- OpenAI LLM (Mistral via OpenAI-compatible API)
- BeautifulSoup (for web scraping)

## 🧪 How to Run

```bash
# Clone the repository
git clone https://github.com/S-Dhanushragav/Sample-Gov-Scheme-Assistant.git
cd scheme-research-assistant

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Add your environment variables
touch .env
