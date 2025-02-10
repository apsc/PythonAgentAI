import os
from pathlib import Path
import weaviate
from weaviate import EmbeddedOptions
from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core.storage import StorageContext
from llama_index.core.readers import SimpleDirectoryReader

client = weaviate.connect_to_local(host="localhost", port=8080, grpc_port=50051, skip_init_checks=True)
# Check that Weaviate is up and live
if client.is_live():
    print("Weaviate is live!")
else:
    print("Weaviate is not reachable.")

from weaviate.classes.config import Configure, Property, DataType, VectorDistances

# Define the collection name and properties

# Define properties with correct field names
properties = [
    Property(name="question", data_type=DataType.TEXT),
    Property(name="answer", data_type=DataType.TEXT),
    Property(name="round", data_type=DataType.TEXT)
]

collection_name = "Canada"
# Create the collection with properly configured vectorizer and vector index
client.collections.create(
    name=collection_name,
    properties=properties,
    vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
    vector_index_config=Configure.VectorIndex.hnsw(
        distance_metric=VectorDistances.COSINE
    )
)



def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index


pdf_path = os.path.join("data", "Canada.pdf")
canada_pdf = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()
canada_index = get_index(canada_pdf, "canada")
canada_engine = canada_index.as_query_engine()
