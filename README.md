# 🧠 Multimodal RAG System – The Batch News Explorer

## 📋 Task: Building a Multimodal RAG System

This project is a **Multimodal Retrieval-Augmented Generation (RAG) System** that enables users to search and retrieve relevant news articles from **The Batch**, leveraging both **textual and visual data** (e.g., article content and associated images).

---

## 🗂️ Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Tools & Models Used](#tools--models-used)
- [Folder Structure](#folder-structure)
- [Future Improvements](#future-improvements)

---

## 🧾 Overview

The goal is to create a complete RAG system that:
- Retrieves and indexes articles with associated media from **The Batch**
- Allows user queries through a simple interface
- Displays relevant content (text + images)
- Enhances results using large language models like **GPT**, **Claude**, or **Gemini**

---

## 🚀 Features

✅ **Data Ingestion & Preprocessing**  
- Scrapes articles and images from The Batch  
- Cleans, structures, and stores data in unified format

✅ **Multimodal Database Creation**  
- Vectorizes textual data using embeddings (e.g., OpenAI, HuggingFace)  
- Stores image and text embeddings for retrieval

✅ **Query Interface**  
- Users input natural language questions  
- System retrieves relevant articles and visuals

✅ **Multimodal Fusion**  
- Combines semantic relevance of text and visuals  
- Generates concise, context-aware answers

✅ **Simple UI for Testing**  
- Built with Streamlit for real-time interaction

✅ **Evaluation & Reporting**  
- System is evaluated on relevance, response quality, and UI usability

---

## 🏗️ System Architecture

```plaintext
User Query → UI → RAG Engine
               ↓
       Text + Image Retriever
               ↓
      Textual & Visual Context
               ↓
       LLM Answer Generator
               ↓
      UI Displays Results + Media
```
## ⚙️ Installation
1. Clone the Repository
```bash
git clone https://github.com/yourusername/multimodal-rag-system.git
cd multimodal-rag-syste
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
## ▶️ Running the Application

To run the Multimodal RAG system, execute the scripts in the following order:

1.  **Clean Previous Data (Crucial for fresh start):**
    Before running, it's highly recommended to remove any old scraped data, images, and index files to ensure a clean re-ingestion and re-indexing.
    ```bash
    rm -rf data index images
    ```

2.  **Run Data Ingestion:**
    This script will scrape articles and images from `deeplearning.ai/the-batch` and save them into the `data/articles.json` and `images/` directories.
    ```bash
    python scrape_articles.py
    ```

3.  **Build the Multimodal Database:**
    This script will process the scraped data, generate multimodal embeddings, and create the FAISS index and metadata files in the `index/` directory.
    ```bash
    python create_index.py
    ```

4.  **Run the UI:**
    Launch the Streamlit application.
    ```bash
    streamlit run app.py
    ```
    After running this command, open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).
Then open http://localhost:8501 in your browser.
## 🧰 Tools & Models Used
- LLMs: OpenAI GPT-4 / Gemini (configurable)
- Embeddings: OpenAI Embeddings or HuggingFace models (e.g., all-MiniLM-L6-v2)
- Image Processing: CLIP, PIL, OpenCV
- UI: Streamlit
- Storage: FAISS for vector similarity search

## 📁 Folder Structure
```bash
.
├── .devcontainer/                # Docker Dev Container configuration
│   └── devcontainer.json
├── data/                         # Stores raw and processed data
│   ├── articles.json             # Scraped article metadata
├── images/                       # Downloaded images from articles
├── index/                        # FAISS index and associated metadata
│   ├── index.faiss               # The FAISS vector index
│   └── metadata.json             # Metadata for indexed articles
├── module/                       # Modules for AI models (Gemini, GPT)
│   ├── gemini_module.py
│   └── gpt_module.py
├── .gitignore                    # Files/folders to ignore in Git
├── app.py                        # Main Streamlit application
├── create_index.py               # Script for creating FAISS index and metadata
├── docker-compose.yml            # (Optional) Docker Compose configuration for multi-service setup
├── Dockerfile                    # (Optional) Dockerfile for building the application image
├── embedding.py                  # Script/module for embedding generation
├── README.md                     # Project documentation (this file)
├── requirements.txt              # Python dependencies
├── retriever.py                  # Script/module for retrieval logic
├── scrape_articles.py            # Script for scraping articles and images
└── start.sh                      # (Optional) Shell script for starting the application
```
## 📄 License
- This project is for educational/demo purposes and not intended for commercial use.
