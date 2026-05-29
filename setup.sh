#!/bin/bash

set -e

echo "Setting up RAG Naive vs Advanced project..."
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
echo "Virtual environment created."
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "Virtual environment activated."
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo ""

# Install requirements
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "Dependencies installed."
else
    echo "Error: requirements.txt not found."
    exit 1
fi
echo ""

# Run ingest script
echo "Ingesting dataset..."
if [ -f "ingest.py" ]; then
    python ingest.py
    echo "Dataset ingestion complete."
else
    echo "Error: ingest.py not found."
    exit 1
fi
echo ""

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. In a separate terminal, start Ollama:"
echo "   ollama serve"
echo ""
echo "2. After Ollama is running, in this terminal run:"
echo "   streamlit run main.py"
echo ""
echo "The Streamlit interface will open in your browser at http://localhost:8501"
