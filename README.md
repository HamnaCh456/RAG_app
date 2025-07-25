## 🧠 RAG app with Streamlit, Pinecone & Groq

This project is a **Retrieval-Augmented Generation (RAG) Chatbot** built with **Streamlit** that leverages `LangChain`, `Pinecone`, and `Groq` APIs to answer user queries based on a specific blog post — ["Agents"](https://lilianweng.github.io/posts/2023-06-23-agent/) by Lilian Weng.

The app loads and chunks the blog content, embeds and indexes it with Pinecone, and uses **Groq's LLaMA 3.3 model** to generate context-aware responses.

---
### 🎥 Demo
     [Watch the demo](https://youtu.be/qeXetlr7FE0)
     
### ✅ Features

* 🔍 **Web Scraping & Parsing** using `LangChain`'s `WebBaseLoader`
* ✂️ **Smart Chunking** via `RecursiveCharacterTextSplitter`
* 📦 **Vector Indexing** with **Pinecone**
* 🧠 **Semantic Search + LLM** answers with **Groq (LLaMA 3.3)**
* 💬 **Interactive UI** built in **Streamlit**

---

### 🚀 How It Works

1. **Data Loading & Splitting**

   * Loads blog content using bs4.SoupStrainer.
   * Splits it into chunks.

2. **Embeddings + Pinecone Indexing**

   * Uses `sentence-transformers/all-MiniLM-L6-v2` to convert chunks into embeddings.
   * Stores embeddings in a Pinecone vector database.

3. **Query & Response**

   * On user input, retrieves the top `k` similar chunks using vector similarity.
   * Feeds them as context into Groq's LLaMA 3.3 model via a prompt template.
   * Displays the generated response with citation support.

---

### 🛠 Tech Stack

* **Frontend/UI**: `Streamlit`
* **Vector DB**: `Pinecone`
* **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
* **LLM API**: `Groq` (LLaMA 3.3 model)
* **Document Loader**: `LangChain WebBaseLoader`
* **Text Splitting**: `LangChain RecursiveCharacterTextSplitter`
* **Deployment-ready**: Easily run locally 

---

### ⚙️ Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/HamnaCh456/RAG_blog_chatbot.git
cd rag-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API keys
export PINECONE_API_KEY=your_key
export GROQ_API_KEY=your_key

# 4. Run the Streamlit app
streamlit run app.py
```

---

### 💡 Example Query

> **Enter:** “What is the difference between reactive and proactive agents?”
> **RAG Output:** Returns context-rich answer from the blog with cited source.

---

### 📌 Notes

* Ensure Pinecone index is created only once.
* Groq API offers blazing fast LLM access — make sure your key has access to the **LLaMA-3.3-70B-versatile** model.
* The current implementation pulls content from a single blog post, but can be extended to multi-source ingestion.

---

