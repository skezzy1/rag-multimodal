#!/bin/bash
echo "Starting Multimodal RAG System setup..."

echo "Running data scraping..."
python scrape_articles.py

if [ $? -ne 0 ]; then
    echo "scrape_articles.py failed. Exiting."
    exit 1
fi

echo "Running index creation..."
python create_index.py

if [ $? -ne 0 ]; then
    echo "create_index.py failed. Exiting."
    exit 1
fi

echo "Starting Streamlit application in development mode..."
streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.runOnSave=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false