# create_vectorstore.py

import pandas as pd
import openai
from pinecone import Pinecone, ServerlessSpec
import os
from os import path
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from a .env file 
load_dotenv()

# Set API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')

pc = Pinecone(
        api_key=pinecone_api_key
    )

# Define the name of your index
index_name = 'research-papers-v1'

def get_embedding(text, model="text-embedding-ada-002"):
    """Generate an embedding for a given text using OpenAI's API."""
    response = openai.Embedding.create(
        input=text,
        engine=model
    )
    return response['data'][0]['embedding']

def main():
    # Load the Excel file
    # df = pd.read_excel('/Users/saad/RAG-v1/arxiv_papers_abstract_nlp.xlsx')
    df = pd.read_excel(path.join(path.dirname(__file__), 'dataset', 'arxiv_papers_abstract_nlp.xlsx'))

    # Ensure the DataFrame has the correct columns
    required_columns = {'Title', 'Abstract', 'Authors'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The Excel file must contain the columns: {required_columns}")

    # Combine relevant text fields for embedding
    df['Combined_Text'] = df['Title'].astype(str) + ' ' + df['Abstract'].astype(str) + ' ' + df['Authors'].astype(str)
    

    # Generate embeddings and store them in a list
    embeddings = []
    print("Generating embeddings...")
    for text in tqdm(df['Combined_Text'], desc="Embedding texts"):
        embedding = get_embedding(text)
        embeddings.append(embedding)

    # Add embeddings to the DataFrame
    df['Embedding'] = embeddings

    # Create a Pinecone index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name='research-papers-v1', 
            dimension=1536, 
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    # Connect to the index
    index = pc.Index(index_name)

    # Prepare data for upsert
    print("Preparing data for upsert...")
    vectors = []
    for i, row in df.iterrows():
        metadata = {
            'Title': row['Title'],
            'Abstract': row['Abstract'],
            'Authors': row['Authors']
        }
        vectors.append((str(i), row['Embedding'], metadata))

    # Upsert vectors to Pinecone in batches
    batch_size = 100  # Adjust the batch size as needed
    print("Upserting vectors to Pinecone...")
    for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting batches"):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)

    print("Finished upserting vectors to Pinecone.")

if __name__ == '__main__':
    main()
