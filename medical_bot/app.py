

import os
import re
import json

import pandas as pd
import numpy as np

import faiss
import nltk

from openai import OpenAI
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from textwrap import dedent

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textwrap import dedent
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 512
AVG_TOK_LEN = 4
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_NAME = "gpt-3.5-turbo"
DATA_DIR = "medical_bot/data/"
FILE_PATH = f"{DATA_DIR}guideline-for-the-diagnosis-and-management-of-aortic-disease-a-report-of-the.pdf"
EMBED_FILE = f"{DATA_DIR}overlap_embeddings_1k.csv"


def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    return "\n".join(text)

def get_stop_words():
    stop_words = list(stopwords.words('english'))
    stop_words.remove("no")
    return stop_words

def process_text(text, stop_words):
    text = re.sub('[^A-Za-z0-9]+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in stop_words] # word_tokenize(text.lower())
    return ' '.join(tokens)

def generate_embeddings(text, model):
    embedding = client.embeddings.create(input = [text], model=model).data[0].embedding
    return embedding

def save_embedding_chunks(pdf_text_processed, model, embed_file, avg_token_len=4, max_tokens=1024, overlap=512):
    chunk_len = avg_token_len * max_tokens
    text_list = []
    embed_list = []
    for i in tqdm(range(0, len(pdf_text_processed)-chunk_len+overlap, chunk_len-overlap)):
        text = pdf_text_processed[i:i+chunk_len]
        embed = generate_embeddings(text, model)
        text_list.append(text)
        embed_list.append(embed)
    
    df = pd.DataFrame({'text': text_list, 'embedding': embed_list})
    df.to_csv(EMBED_FILE, index=False)
    return df

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_similarity(embeddings, query_embed, top_k=5):
    embed_scores = embeddings.apply(
        lambda x: (x['text'], x['embedding'], cosine(x['embedding'], query_embed)), axis=1
    )
    embed_scores_df = pd.DataFrame(embed_scores.tolist(), columns=['text', 'embedding', 'score'])
    sim_df = embed_scores_df.sort_values(by='score', ascending=False)[:top_k]
    return sim_df

def duplicate_response(previous_responses, resp_embed, threshold=0.9):
    #return any(np.allclose(resp_embed, prev_resp, atol=1e-2) for prev_resp in previous_responses) 
    for pr_embed in previous_responses:
        cos = round(cosine(pr_embed, resp_embed), 2)
        if cos >= threshold:
            return True
    return False

def is_satisfied(query_response, user_input):
    if "satisfied with response" in query_response.lower() and "yes" in user_input.lower():
        return True
    return False

##### Vectorized implementation of MMR (maximal marginal relevance) #####
    
def diversity_ranking(query_embed, selected_docs, unselected_docs, lambda_=0.7, top_k=5):
    # convert embeddings to np arrays
    docs = [d[0] for d in unselected_docs]
    usd = np.array([d[1] for d in unselected_docs])
    sd = np.array([d[1] for d in selected_docs])
    q = np.array(query_embed).reshape(1, -1)
    
    # similarity of query with all unselected docs
    cos_one = np.dot(q, usd.T) / (np.linalg.norm(q, axis=1).reshape(-1, 1) * np.linalg.norm(usd, axis=1).reshape(1, -1))
    
    # similarity of selected docs with all unselected docs
    cos_two = np.dot(sd, usd.T) / (np.linalg.norm(sd, axis=1).reshape(-1, 1) * np.linalg.norm(usd, axis=1).reshape(1, -1))
    cos_two = np.max(cos_two, axis=0).reshape(1, -1)
    
    # compute mmr scores and create df
    score = cos_one * lambda_ - (1 - lambda_) * cos_two
    score = score.flatten().tolist()

    # return top_k docs with highest MMR (maximal marginal relevance)
    score_df = pd.DataFrame({'doc': docs, 'score': score})
    
    top_k_idx = score_df['score'].nlargest(top_k).index
    return score_df.iloc[top_k_idx]


st.set_page_config(page_title="MedBot")
st.title("Medical Chatbot")

# initialize openai client
client = OpenAI()
    
# build and load vector store at app startup
with st.spinner("Starting up..."):
    pdf_text = read_pdf(FILE_PATH)
    stop_words = get_stop_words()
    pdf_text_processed = process_text(pdf_text, stop_words)
    if os.path.basename(EMBED_FILE) not in os.listdir(DATA_DIR):
        print(os.listdir(DATA_DIR))
        embed_df = save_embedding_chunks(
            pdf_text_processed, 
            model=EMBEDDING_MODEL, 
            embed_file=EMBED_FILE, 
            avg_token_len=AVG_TOK_LEN, 
            max_tokens=CHUNK_SIZE, 
            overlap=CHUNK_OVERLAP
        )
    else:
        embed_df = pd.read_csv(EMBED_FILE)
        embed_df['embedding'] = embed_df['embedding'].apply(lambda x: json.loads(x))
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, how can I help you?")]

# User input
user_input = st.chat_input("Type your message here...")
query_response  = ""
if user_input is not None and user_input != "":
    with st.spinner("Processing..."):
        # Initialize user interaction and setup
        previous_responses = []
        messages = [{}]
        st.session_state['stop'] = False
        query = dedent(f"{query_response}, {user_input}")
        st.session_state.chat_history.append(user_input)
        
        # Generate query embeddings and perform retrieval and ranking
        query_embed = generate_embeddings(query, EMBEDDING_MODEL) #
        selected_docs = compute_similarity(embed_df, query_embed, top_k=5)
        unselected_docs = embed_df.loc[~embed_df.index.isin(selected_docs.index)]

        # Document Ranking using Maximum Marginal Relevance to select top_k docs (using higher lambda for relevance)
        # TODO: figure out a mechanism to tweak values of lambda in cases of invalid or duplicate responses.
        ranked_docs = diversity_ranking(query_embed, selected_docs.values.tolist(), unselected_docs.values.tolist(), lambda_=0.7, top_k=5)
        context = '\n'.join([d[0] for d in ranked_docs.values.tolist()])

        # System message setup
        system_msg = dedent(f"""
            You are a medical expert. You can only use the information provided in the context that comes from 
            our own data source to response. When asked a question, you will only ask relevant follow-up questions
            till you are able to find relevant sections of the data/information required by the user. If you cannot 
            answer, then respond with I don't know the answer. Once relevant information is provided to the user,
            ask them if they are satisfied with response (yes/no), if they answer is yes, then terminate the conversation. 
            Also, make sure NOT to ask duplicate questions based on previous chat history.
            Context: {context}
            """)

        messages[0] = {"role": "system", "content": system_msg}
        messages.append({"role": "user", "content": query})
        
        
        # Get model response
        response = client.chat.completions.create(model=LLM_NAME, messages=messages, temperature=0)
        query_response = response.choices[0].message.content

        # Process and generate embedding for the response for duplication check
        resp_processed = process_text(query_response, stop_words)
        resp_embed = generate_embeddings(resp_processed, EMBEDDING_MODEL)

        # Check for duplicate response
        if duplicate_response(previous_responses, resp_embed):
            msg = "It is a duplicate response, generate a new response."
            messages.append({"role": "user", "content": msg})

        # Display and verify the response
        previous_responses.append(resp_embed) 
        
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=query_response))
        
        st.session_state['stop'] = is_satisfied(query_response, user_input)

            
        
# Conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

