# TechnoMile's Governer AI

## Overview
This system is a sophisticated contract analysis tool that combines vector similarity search with BM25 text ranking to efficiently search and analyze government contract data stored in MongoDB. It uses OpenAI's embeddings and a hybrid retrieval approach for optimal search results.

## Table of Contents
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [API Reference](#api-reference)

## Features
- Hybrid search combining vector similarity and BM25 text ranking
- Real-time contract data retrieval from MongoDB
- Advanced document indexing and search optimization
- Detailed contract information extraction and formatting
- Configurable search parameters and scoring
- Robust error handling and connection management

## Technology Stack
- **Database**: MongoDB
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: OpenAI (text-embedding-3-large)
- **Text Ranking**: BM25Okapi
- **Backend Framework**: Flask with CORS support
- **Environment Management**: python-dotenv
- **Additional Libraries**: numpy, langchain, langgraph

## Project Structure
project/
├── app.py # Main application file
├── .env # Environment variables
├── faiss_index/ # Stored vector indices
└── requirements.txt # Project dependencies

## Core Components

### 1. Vector Store Initialization
The system uses FAISS for efficient similarity search:

Key features:
- Configurable index parameters
- Cosine similarity metric
- Automatic dimension detection
- Local index storage and loading

### 2. Hybrid Retriever
Combines vector similarity with BM25 text ranking:

Features:
- Configurable alpha parameter for score balancing
- Optimized BM25 parameters
- Score normalization
- Combined ranking algorithm

### 3. Data Processing
MongoDB data fetching and processing:

## Setup and Installation

1. Clone the repository
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
```

5. Ensure MongoDB is running locally on default port (27017)

## Usage

1. Initialize the system:
```python
from app import initialize_vector_store, HybridRetriever

# Initialize vector store
vector_store = initialize_vector_store()

# Create hybrid retriever
hybrid_retriever = HybridRetriever(vector_store, documents, alpha=0.5)
```

2. Perform searches:
```python
results = hybrid_retriever.hybrid_search("your search query", k=20)
```

## API Reference

### HybridRetriever Class
- `__init__(vector_store, documents, alpha=0.5)`: Initialize the hybrid retriever
- `hybrid_search(query: str, k: int = 20)`: Perform hybrid search

### Vector Store Functions
- `initialize_vector_store()`: Create new vector store
- `load_or_initialize_vector_store()`: Load existing or create new vector store
- `fetch_and_process_data()`: Retrieve and process MongoDB data

### Data Processing
The system processes the following contract fields:
- Contract ID
- Agency information
- Financial details
- Vendor information
- Competition details
- Important dates
- Contract descriptions

## Error Handling
The system includes comprehensive error handling for:
- MongoDB connection issues
- OpenAI API failures
- Vector store initialization problems
- Search and retrieval errors

## Performance Optimization
- Efficient indexing strategies
- Optimized search parameters
- Batch processing capabilities
- Configurable search depth and precision

