from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext, StorageContext
from llama_index.llms.lmstudio import LMStudio
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.vector_stores.lancedb import LanceDBVectorStore
import lancedb
import os

# Configure LlamaIndex to use the LM Studio API
llm = LMStudio(
    model_name="deepseek-r1-distill-qwen-7b",
    base_url="http://localhost:1234/v1",
    temperature=0.7,
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Load documents from "documents" folder
documents = SimpleDirectoryReader("documents").load_data(show_progress=True)

# Ensure the "lancedb_data" folder exists
lancedb_data_dir = "./lancedb_data"
if not os.path.exists(lancedb_data_dir):
    os.makedirs(lancedb_data_dir)
    print(f"Created directory: {lancedb_data_dir}")

# Set up LanceDB
db = lancedb.connect(lancedb_data_dir)  # Connect to the local directory

# Set up the vector store
vector_store = LanceDBVectorStore(uri=lancedb_data_dir, table_name="my_table")

# Create a storage context with LanceDB
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create an index with LanceDB as the vector store
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

# Create a query engine
query_engine = index.as_query_engine(llm=llm)

# Loop to allow multiple queries
while True:
    # Get user input
    user_input = input("Ask a question (or type 'q', 'quit', 'exit' to quit): ").strip()

    # Exit conditions
    if user_input.lower() in ["q", "quit", "exit", ""]:
        print("Goodbye!")
        break

    # Query the index with user input
    response = query_engine.query(user_input)
    print(f"Response: {response}\n")