# Salamander Re-ID Web Application

This is a modern web-based version of the Salamander Re-Identification tool.

## Prerequisites
- Python 3.10+
- Node.js & npm

## How to Run

### 1. Start the Backend
```bash
# In the project root
python server.py
```
The API will be available at `http://localhost:8000`.

### 2. Start the Frontend
```bash
# In a new terminal
cd frontend
npm install
npm run dev
```
The web application will be available at `http://localhost:3000`.

## Features
- **Modern Dashboard**: Clean interface for managing salamander identities.
- **Identity Browser**: Search and browse unique identities with representative images.
- **Validation Mode**: Compare query images with their closest matches to verify or update IDs.
- **Error Detection**: Automated identification of potential identity mismatches.
- **Data Management**: Load metadata, generate embeddings, split data, and export corrected results.
