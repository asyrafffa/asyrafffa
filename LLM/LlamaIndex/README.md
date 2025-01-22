# LlamaIndex with LanceDB and LM Studio

This project demonstrates how to use **LlamaIndex** to create a vector store with **LanceDB** and query it using a locally hosted **LM Studio** LLM (Large Language Model). It allows users to interact with the LLM by asking questions and receiving responses in real-time.

## Features

- **Document Indexing**: Load and index documents from a specified folder.
- **Vector Store**: Use **LanceDB** as a local vector store for efficient storage and retrieval.
- **LLM Integration**: Query the indexed documents using a locally hosted LLM via **LM Studio** by using DeepSeek-R1 (deepseek-r1-distill-qwen-7b).
- **Interactive Querying**: Ask multiple questions interactively until you decide to quit.

## Prerequisites

Before running the project, ensure you have the following installed:

1. **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
2. **LM Studio**: [Download LM Studio](https://lmstudio.ai/) and set up a local LLM server.
3. **Required Python Libraries**: Install the required libraries using `pip`.

## Installation

1. Clone this repository:

2. Install the required Python libraries:
    ```bash
    pip install llama-index-core llama-index-llms-lmstudio llama-index-embeddings-huggingface lancedb
    ```
3. Place your documents in the documents folder. These will be indexed by LlamaIndex.
4. Start LM Studio and host your LLM locally. Ensure the API is accessible at http://localhost:1234/v1.

## Usage

1. Run the script
  ```bash
  python main.py
  ```
2. The program will load and index the documents from the documents folder. Once ready, it will prompt you to ask a question.
3. Type your question and press Enter. The program will query the LLM and display the response.
4. To exit, type q, quit, exit, or leave the input blank.

## Code Overview

The main components of the code are:
1. **Document Loading**:
- Documents are loaded from the **documents** folder using **SimpleDirectoryReader**.
2. **Vector Store**:
- **LanceDB** is used as the vector store for efficient storage and retrieval of embeddings.
3. **LLM Integration**:
- The **LM Studio** LLM is used for querying the indexed documents. The LLM is hosted locally and accessed via the LM Studio API.
4. **Interactive Querying**:
- The program runs in a loop, allowing users to ask multiple questions until they decide to quit.

## Customization

- **LLM Model**: You can change the **model_name** in the LMStudio configuration to use a different model.
- **Embedding Model**: Modify the **HuggingFaceEmbedding** configuration to use a different embedding model.
- **Document Folder**: Place your documents in the **documents** folder or update the path in the **SimpleDirectoryReader** call.
