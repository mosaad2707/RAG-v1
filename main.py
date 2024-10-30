import main as st
import openai
from pinecone import Pinecone, ServerlessSpec
import os
from uuid import uuid4
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from a .env file (optional)
load_dotenv()

# Set API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_index_name = os.getenv('PINECONE_INDEX_NAME', 'research-papers-v1')

# Initialize Pinecone
pc = Pinecone(
    api_key=pinecone_api_key
)
index = pc.Index(pinecone_index_name)


# Set up Streamlit app
st.set_page_config(page_title="Arxiv Assistant", page_icon="📚")
st.title("📚 Research Paper Assistant")

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

def get_embedding(text, model="text-embedding-ada-002"):
    """Generate an embedding for a given text using OpenAI's API."""
    response = openai.Embedding.create(
        input=text,
        engine=model
    )
    return response['data'][0]['embedding']

def query_papers(query, top_k=3):
    """Query Pinecone index to retrieve relevant papers."""
    # Generate embedding for the query
    query_embedding = get_embedding(query)
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Extract and return results
    papers = []
    for match in results['matches']:
        metadata = match['metadata']
        papers.append({
            'Title': metadata.get('Title', 'No Title'),
            'Abstract': metadata.get('Abstract', 'No Abstract'),
            'Authors': metadata.get('Authors', 'Unknown Authors'),
            'Score': match['score']
        })
    return papers

def generate_response(query, chat_history):
    """Generate response using OpenAI's GPT model with streaming."""
    papers = query_papers(query, top_k=3)
    
    # Create a summary of the top papers
    paper_summaries = '\n\n'.join([
        f"**Title:** {p['Title']}\n**Authors:** {p['Authors']}\n**Abstract:** {p['Abstract']}"
        for p in papers
    ])
    
    
    # Construct the system prompt
    system_prompt = f"""
You are a helpful assistant specialized in providing information based on research papers.

Based on the following research papers, provide an answer to the user's query.

Research Papers:
{paper_summaries}
"""

    # Prepare the messages for the chat
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": query})
    
    # Call OpenAI's ChatCompletion with streaming
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',  # Change to 'gpt-4' if you have access
        messages=messages,
        stream=True
    )
    
    # Stream the response
    full_response = ""
    for chunk in response:
        chunk_message = chunk['choices'][0]['delta'].get('content', '')
        full_response += chunk_message
        yield chunk_message  # Use yield to stream the response

# Display chat messages from history on app rerun
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        

# Accept user input using st.chat_input
if user_query := st.chat_input("Ask a question about research papers"):
    # Display user's message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Update chat history
    st.session_state['messages'].append({"role": "user", "content": user_query})
    
    # Generate and display assistant's response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in generate_response(user_query, st.session_state['messages']):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)
    
    # Update chat history with assistant's response
    st.session_state['messages'].append({"role": "assistant", "content": full_response})

# Optional: clear chat history button
if st.button("Clear Chat"):
    st.session_state['messages'] = []
    st.rerun()
