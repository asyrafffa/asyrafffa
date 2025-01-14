# LangFlow RAG Agent Portfolio

![Project Banner](https://github.com/your-username/your-repository-name/blob/main/banner.png)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Welcome to the **LangFlow RAG Agent** project! This repository showcases a Retrieval-Augmented Generation (RAG) agent built using LangFlow, AstraDB, and a local Large Language Model (LLM). The project demonstrates how to efficiently ingest, manage, and retrieve information from PDF documents to power intelligent agent responses.

## Features

- **PDF Upload & Processing**: Easily upload PDF documents which are then split into manageable text chunks.
- **Data Ingestion**: Processed data is ingested into the AstraDB vector database using NVIDIA embeddings.
- **Local LLM Integration**: Utilizes the Hermes-3-llama-3.2-3b model from LM Studio as the underlying agent.
- **Intelligent Retrieval**: The agent retrieves relevant data from AstraDB based on user queries related to the uploaded PDFs.
- **Seamless Interaction**: Provides accurate and context-aware answers based on the ingested information.

## Architecture

The project is structured into two main workflows, orchestrated using LangFlow:

### 1. Data Ingestion Flow

1. **Upload PDF**: Users can upload PDF documents through the interface.
2. **Text Splitting**: The uploaded PDF is split into smaller text chunks for efficient processing.
3. **Embedding Generation**: NVIDIA embeddings provided by AstraDB are used to convert text chunks into vector representations.
4. **Data Ingestion**: The vectorized data is ingested into the AstraDB vector database for storage and retrieval.

### 2. Query & Response Flow

1. **User Query**: Users input queries related to the content of the uploaded PDFs.
2. **Data Retrieval**: The local LLM agent (Hermes-3-llama-3.2-3b) retrieves relevant data from the AstraDB based on the query.
3. **Answer Generation**: The agent generates responses using the retrieved information, ensuring accurate and contextually relevant answers.

## Setup & Installation

Follow these steps to set up and run the LangFlow RAG Agent on your local machine.

### Prerequisites

- **Python 3.8+**
- **Node.js & npm** (for frontend dependencies, if any)
- **AstraDB Account**: Sign up at [AstraDB](https://www.datastax.com/products/datastax-astra) and obtain your database credentials.
- **LM Studio**: Ensure LM Studio is installed and the Hermes-3-llama-3.2-3b model is available.

### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   cd your-repository-name
