import json
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.ollama import OllamaEmbedding

#Elasticsearch Configuration
es_vector_store = ElasticsearchStore(
    index_name="twitter_posts",
    vector_field="text_vector",
    text_field="text",
    es_url="http://localhost:9200"
)

# Preprocessing
def preprocess_text(text):
    return text.replace("\n", " ").strip()

# Load data from json file
def get_documents_from_file(file_path):
    with open(file_path, mode="rt", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        metadata = {k: v for k, v in item.items() if k != "text"}
        documents.append(Document(
            text=preprocess_text(item["text"]),
            metadata=metadata
        ))
    return documents

# Main function
def main():
    # Embedding model
    embedding = OllamaEmbedding("llama3.1:8b")

    pipeline = IngestionPipeline(
        transformations=[
            embedding,
        ],
        vector_store=es_vector_store
    )

    # Load data from a json file into a list of LlamaIndex Documents
    documents = get_documents_from_file("./dataset.json")

    pipeline.run(documents=documents, show_progress=True)
    print(".....Start pipeline.....\n")


if __name__ == "__main__":
    main()
