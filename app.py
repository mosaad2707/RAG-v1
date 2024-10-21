from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

print(pinecone_api_key)
pc = Pinecone(api_key=pinecone_api_key)


# print the data
data = list(pc.Index(pinecone_index_name).list())
print(data)
