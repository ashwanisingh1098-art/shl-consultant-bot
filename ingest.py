import json
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def run_ingestion():
    # 1. Load the JSON
    loader = JSONLoader(
        file_path="./shl_product_catalog.json",
        jq_schema=".[]",
        text_content=False
    )
    raw_documents = loader.load()

    clean_docs = []
    for doc in raw_documents:
        item = json.loads(doc.page_content)
        # Creating a rich text representation for better search
        text = f"Name: {item.get('name')}\nDescription: {item.get('description')}\nCategory: {item.get('keys')}"
        
        clean_docs.append(
            Document(
                page_content=text,
                metadata={
                    "name": item.get("name"),
                    "link": item.get("link")
                }
            )
        )

    # 2. Setup Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Create FAISS Database
    print("Creating vector database...")
    db = FAISS.from_documents(clean_docs, embeddings)

    # 4. Save it locally
    db.save_local("faiss_index")
    print("Successfully saved 'faiss_index' folder!")

if __name__ == "__main__":
    run_ingestion()