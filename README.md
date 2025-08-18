# 📄 DocuChat: Chat with Your Documents & Links 🤖

DocuChat is an interactive Streamlit application that allows you to have a conversation with your documents. Upload a file (`PDF`, `DOCX`, `TXT`, `XLSX`, images) or provide a web link, and the app will use a powerful Large Language Model (LLaMA 3 via Groq) to answer your questions about the content.

## ✨ Features

-   **Multi-Format Support**: Upload various file types, including PDF, TXT, DOCX, Excel spreadsheets, and even images (PNG, JPG) for OCR.
-   **Web Content Ingestion**: Paste a URL to chat with the content of a webpage.
-   **Powered by LLaMA 3 & Groq**: Leverages the speed and power of the LLaMA 3 model running on the Groq LPU™ Inference Engine for real-time responses.
-   **Adjustable Persona**: Change the explanation style of the chatbot to suit a "Child," "Young Person," or "Adult."
-   **Content Dashboard**: After processing a document, view a dashboard with a concise summary, key topics, word count, and estimated reading time.
-   **Secure API Key Handling**: Uses Streamlit's built-in secrets management for safe API key storage.
-   **Modern UI**: A clean, styled, and intuitive user interface built with Streamlit.

## 🛠️ Tech Stack

-   **Frontend**: Streamlit
-   **LLM & Embeddings**:
    -   [Groq](https://groq.com/) for LLaMA 3 inference
    -   [LangChain](https://www.langchain.com/) for RAG pipeline orchestration
    -   [Hugging Face](https://huggingface.co/) Sentence Transformers for embeddings
    -   [FAISS](https://github.com/facebookresearch/faiss) for vector storage
-   **Text Extraction**: `PyPDF2`, `python-docx`, `pandas`, `pytesseract`, `BeautifulSoup`

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

-   Python 3.8+
-   A [Groq Cloud API Key](https://console.groq.com/keys)

### Local Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-folder>
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API key:**
    -   Create a folder in your project's root directory named `.streamlit`.
    -   Inside that folder, create a file named `secrets.toml`.
    -   Add your Groq API key to this file as follows:
        ```toml
        GROQ_API_KEY = "your_groq_api_key_here"
        ```

5.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

## ☁️ Deployment on Streamlit Community Cloud

This app is ready to be deployed on Streamlit Community Cloud.

1.  Push your project to a public or private GitHub repository.
2.  Sign up or log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3.  Click "New app" and connect your GitHub account.
4.  Select the repository and branch.
5.  Under "Advanced settings," go to the "Secrets" section and paste your Groq API key in the same format as your `secrets.toml` file.
6.  Click "Deploy!"```

### Step 2: Create `requirements.txt`

Your Streamlit Cloud app needs to know which Python libraries to install. In your project's root directory, create a file named `requirements.txt` and paste this exact list into it.

```text
streamlit
requests
Pillow
pytesseract
python-docx
pandas
openpyxl
beautifulsoup4
langchain
langchain-community
langchain-groq
faiss-cpu
huggingface-hub
sentence-transformers
pypdf