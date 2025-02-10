from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from dotenv import load_dotenv
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent

load_dotenv()

llm = Ollama(model="deepseek-r1:8b",request_timeout=30)
# result = llm.complete("Hello World")
# print(result)

parser = LlamaParse(result_type="markdown")
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
canada_engine = vector_index.as_query_engine(llm=llm)

response = canada_engine.query("What is the capital of Canada?")
print(response)