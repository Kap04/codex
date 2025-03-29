# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import random
from crawl4ai import CrawlerRunConfig , CacheMode , AsyncWebCrawler
from supabase import create_client
from mistralai import Mistral
import uuid
from dotenv import load_dotenv
from urllib.parse import urlparse
import numpy as np
from typing import List, Dict, Any
import asyncio

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import UserMessage

import  traceback
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy

# Instantiate the tokenizer (using v3 as per docs)
tokenizer = MistralTokenizer.from_model("mistral-embed"  , strict=True)

# Update the special token policy to IGNORE special tokens during decoding
tokenizer.instruct_tokenizer.tokenizer.special_token_policy = SpecialTokenPolicy.IGNORE


# tokenizer = MistralTokenizer.v3(is_tekken=True)



# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Supabase setup
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

# Mistral AI setup
mistral_api_key = os.environ.get("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=mistral_api_key)
print(mistral_api_key)
MISTRAL_MODEL = "codestral-2501"  # Choose appropriate model version

@app.route('/crawl', methods=['POST'])
def crawl():
    try:
        # Get URL from request
        data = request.json
        url = data.get('url')
        
        if not url:
            return jsonify({"error": "URL is required"}), 400
        
        # Debug log for the received URL
        print(f"Received URL for crawling: {url}")
        
        # Async function to run the crawler with BFSDeepCrawlStrategy
        async def run_crawler():
            try:
                # Configure deep crawl using BFS strategy
                deep_crawl_strategy = BFSDeepCrawlStrategy(
                    max_depth=1,             # Adjust max_depth as needed for your documentation
                    include_external=False   # Do not follow external links
                )
                
                run_conf = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    word_count_threshold=1,
                    page_timeout=60000,  # 60 seconds timeout
                    deep_crawl_strategy=deep_crawl_strategy
                )
                
                async with AsyncWebCrawler() as crawler:
                    results = await crawler.arun(url, config=run_conf)
                    
                    # Validate that we have results
                    if not results or not isinstance(results, list):
                        raise ValueError("Failed to extract content from the URL")
                    
                    return results
            
            except Exception as crawler_error:
                print(f"Crawler Error: {str(crawler_error)}")
                raise
        
        # Run the async crawler and process the list of results
        try:
            results = asyncio.run(run_crawler())
            
            # Log each result's scraped markdown content (fit version) and internal links
            for idx, result in enumerate(results, start=1):
                print(f"--- Page {idx} ---")
                if result.markdown and hasattr(result.markdown, "fit_markdown"):
                    print("Fit Markdown Content:")
                    print(result.markdown.fit_markdown)
                else:
                    print("No markdown content found for this page.")
                
                # Log internal links for each page
                print("Internal Links:")
                internal_links = result.links.get("internal", [])
                for link in internal_links:
                    href = link.get("href", "No URL")
                    text = link.get("text", "").strip()
                    print(f"URL: {href} | Text: {text}")
                print("\n")
            
            # Example: Aggregate raw markdown from all pages if needed
            aggregated_raw_markdown = "\n\n".join(
                [result.markdown.raw_markdown for result in results if result.markdown and hasattr(result.markdown, "raw_markdown")]
            )
            
        except Exception as async_error:
            print(f"Async Crawler Execution Error: {str(async_error)}")
            return jsonify({
                "error": "Failed to crawl the documentation",
                "details": str(async_error)
            }), 500
        
        # Store the aggregated document in Supabase
        doc_id = str(uuid.uuid4())
        
        # Create embeddings for the aggregated markdown content
        markdown_embedding = create_embeddings(aggregated_raw_markdown)
        
        # Insert document into Supabase
        supabase.table("documents").insert({
            "id": str(uuid.uuid4()),  # Unique UUID for the primary key
            "doc_id": doc_id,
            "url": url,
            "content": aggregated_raw_markdown,
            "embedding": markdown_embedding
        }).execute()
        
        return jsonify({
            "message": "Document crawled and stored successfully",
            "doc_id": doc_id,
            "markdown_length": len(aggregated_raw_markdown)
        })
        
    except Exception as e:
        # Print full traceback for detailed error information
        import traceback
        print("Full error traceback:")
        traceback.print_exc()
        
        return jsonify({
            "error": "Internal Server Error", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500



@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
            
        # Get the most recent doc_id from the documents table
        response = supabase.table("documents").select("doc_id").limit(1).order("created_at", desc=True).execute()
        
        if not response.data:
            return jsonify({"error": "No documents found. Please process a documentation first."}), 400
            
        doc_id = response.data[0]['doc_id']
        
        # Create embedding for the question
        question_embedding = create_embeddings(question)
        
        # Query Supabase for relevant document chunks using vector similarity
        query_response = supabase.rpc(
            "match_documents",
            {
                "query_embedding": question_embedding,  
                "match_document_id": doc_id,
                "match_threshold": 0.7,
                "match_count": 3
            }
        ).execute()
        
        relevant_docs = query_response.data
        
        if not relevant_docs:
            return jsonify({
                "answer": "I couldn't find relevant information to answer your question in the provided documentation."
            })
            
        # Construct context from relevant documents
        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        
        # Generate response using Mistral AI
        response = query_mistral(question, context)
        
        return jsonify({"answer": response})
        
    except Exception as e:
        print(f"Error in ask: {str(e)}")
        return jsonify({"error": str(e)}), 500

def tokenize_text(text: str) -> (List[int], str):
    """
    Tokenizes a plain text string using the Mistral tokenizer by wrapping it in a ChatCompletionRequest.
    Returns a tuple of (tokens, debug_text).
    """
    chat_req = ChatCompletionRequest(
        messages=[UserMessage(content=text)],
        model="test"  # Dummy model name for tokenization purposes
    )
    tokenized = tokenizer.encode_chat_completion(chat_req)
    return tokenized.tokens, tokenized.text

def get_token_count(text: str) -> int:
    tokens, _ = tokenize_text(text)
    return len(tokens)



def get_embeddings_by_chunks(text: str, max_tokens: int = 8000, delay: float = 0.5, max_retries: int = 5) -> List[float]:
    """
    Splits the input text into token-based chunks (each under max_tokens) 
    and returns the concatenated embeddings.
    Implements exponential backoff to handle rate limit errors.
    Uses the v1 tokenizer for mistral-embed without filtering any tokens.
    """
    # Tokenize the full text first using our helper function
    tokens, _ = tokenize_text(text)
    print(f"Total tokens for document: {len(tokens)}")
    
    # Create chunks based on token count
    token_chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    
    embeddings = []
    for idx, chunk in enumerate(token_chunks, start=1):
        # Directly decode the chunk without filtering any tokens
        chunk_text = tokenizer.decode(chunk)
        token_count = len(chunk)
        print(f"Processing chunk {idx} with {token_count} tokens")
        
        # Exponential backoff for rate limiting
        retries = 0
        while retries < max_retries:
            try:
                response = mistral_client.embeddings.create(
                    model="mistral-embed",
                    inputs=[chunk_text]
                )
                embeddings.extend(response.data[0].embedding)
                time.sleep(delay)
                break  # Exit retry loop on success
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    retries += 1
                    sleep_time = delay * (2 ** retries) + random.uniform(0, 0.5)
                    print(f"Rate limit hit on chunk {idx}. Retrying in {sleep_time:.2f} seconds (Attempt {retries}/{max_retries})...")
                    time.sleep(sleep_time)
                else:
                    raise e
        else:
            raise Exception("Exceeded maximum retries due to rate limit errors.")
    return embeddings





def create_embeddings(text: str) -> List[float]:
    """
    Create embeddings using batch processing with Mistral AI API.
    Instead of concatenating chunk embeddings, we aggregate them (average)
    so that the final vector dimension remains consistent (e.g., 1024).
    """
    # Get the list of embeddings for each chunk.
    # Assume get_embeddings_by_chunks returns a concatenated list of embeddings
    # from each chunk.
    chunk_embeddings = get_embeddings_by_chunks(text, max_tokens=3000)
    
    # For aggregation, we need to know the dimension of each individual embedding.
    # This might be provided by the API or be documented (e.g., 1024 for mistral-embed).
    embedding_dim = 1024  # update based on your model's specification
    
    # Calculate how many chunks we have by dividing the total length by the embedding dimension.
    num_chunks = len(chunk_embeddings) // embedding_dim
    if num_chunks == 0:
        raise ValueError("No embeddings were returned.")
    
    # Reshape the embeddings list into a 2D array: (num_chunks, embedding_dim)
    embeddings_array = np.array(chunk_embeddings).reshape((num_chunks, embedding_dim))
    
    # Aggregate (e.g., average) the embeddings along the chunks axis to get a single vector.
    aggregated_embedding = embeddings_array.mean(axis=0)
    
    return aggregated_embedding.tolist()



def query_mistral(question: str, context: str) -> str:
    """Query Mistral AI with the question and context and log token usage."""
    prompt = f"""Answer the following question based on the provided documentation context:

    Context:
    {context}

    Question: {question}

    Provide a concise and accurate answer based only on the information in the context. If the answer cannot be determined from the context, please state that.
    """
    # Log token count for the prompt
    #token_count = get_token_count(prompt)
    #print(f"Prompt token count: {token_count}")
    
    chat_response = mistral_client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000
    )
    
    return chat_response.choices[0].message.content


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)