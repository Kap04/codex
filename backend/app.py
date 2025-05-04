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
import jwt
from datetime import datetime, timedelta
from functools import wraps

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import UserMessage

import  traceback
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy

import re
from bs4 import BeautifulSoup


# Instantiate the tokenizer (using v3 as per docs)
tokenizer = MistralTokenizer.from_model("mistral-embed"  , strict=True)

# Update the special token policy to IGNORE special tokens during decoding
tokenizer.instruct_tokenizer.tokenizer.special_token_policy = SpecialTokenPolicy.IGNORE




# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Supabase setup
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase = create_client(supabase_url, supabase_key)

# Mistral AI setup
mistral_api_key = os.environ.get("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=mistral_api_key)
print(mistral_api_key)
MISTRAL_MODEL = "codestral-2501"  # Choose appropriate model version

# JWT Configuration
JWT_SECRET = os.environ.get("JWT_SECRET", "your-secret-key")  # Make sure to set this in .env
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_DELTA = timedelta(days=1)



def setup_database():
    """Create necessary tables if they don't exist"""
    try:
        print("Setting up database tables...")
        
        # Check if users table exists
        try:
            # Try to query the users table
            supabase.table("users").select("id").limit(1).execute()
            print("Users table already exists")
        except Exception as e:
            print(f"Users table doesn't exist or is not accessible: {str(e)}")
            print("Please run the SQL script in the Supabase SQL editor to create the table")
            print("You can find the script at: backend/create_users_table.sql")
            
        print("Database setup completed")
    except Exception as e:
        print(f"Error setting up database: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

# Run database setup on startup
setup_database()

def create_token(user_id: str) -> str:
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + JWT_EXPIRATION_DELTA
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(" ")[1]
            else:
                token = auth_header
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            current_user = supabase.table("users").select("*").eq("id", data['user_id']).execute()
            if not current_user.data:
                return jsonify({'message': 'Invalid token'}), 401
        except Exception as e:
            print(f"Token validation error: {str(e)}")
            return jsonify({'message': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    return decorated

@app.route('/auth/register', methods=['POST'])
def register():
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        print(f"Registration attempt for email: {email}")
        
        if not email or not password:
            print("Registration failed: Missing email or password")
            return jsonify({"error": "Email and password are required"}), 400
            
        # Check if user already exists
        print("Checking if user already exists...")
        existing_user = supabase.table("users").select("*").eq("email", email).execute()
        print(f"Existing user check result: {existing_user}")
        
        if existing_user.data:
            print(f"Registration failed: User already exists with email {email}")
            return jsonify({"error": "User already exists"}), 400
            
        # Create new user
        user_id = str(uuid.uuid4())
        print(f"Creating new user with ID: {user_id}")
        
        try:
            new_user = supabase.table("users").insert({
                "id": user_id,
                "email": email,
                "password": password  # In production, hash the password!
            }).execute()
            print(f"User creation result: {new_user}")
        except Exception as db_error:
            print(f"Database error during user creation: {str(db_error)}")
            print(f"Error type: {type(db_error)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return jsonify({"error": f"Database error: {str(db_error)}"}), 500
        
        # Generate token
        token = create_token(user_id)
        print(f"Token generated successfully for user: {user_id}")
        
        return jsonify({
            "message": "User created successfully",
            "token": token
        })
        
    except Exception as e:
        print(f"Unexpected error during registration: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/auth/login', methods=['POST'])
def login():
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        print(f"Login attempt for email: {email}")
        
        if not email or not password:
            print("Login failed: Missing email or password")
            return jsonify({"error": "Email and password are required"}), 400
            
        # Find user
        print("Attempting to find user in database...")
        try:
            user = supabase.table("users").select("*").eq("email", email).eq("password", password).execute()
            print(f"User query result: {user}")
        except Exception as db_error:
            print(f"Database error during login: {str(db_error)}")
            print(f"Error type: {type(db_error)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return jsonify({"error": f"Database error: {str(db_error)}"}), 500
        
        if not user.data:
            print(f"Login failed: Invalid credentials for email {email}")
            return jsonify({"error": "Invalid credentials"}), 401
            
        # Generate token
        user_id = user.data[0]['id']
        token = create_token(user_id)
        print(f"Login successful for user: {user_id}")
        
        return jsonify({
            "message": "Login successful",
            "token": token
        })
        
    except Exception as e:
        print(f"Unexpected error during login: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500










@app.route('/sessions', methods=['GET'])
@token_required
def get_sessions():
    try:
        # Get user_id from JWT token
        token = request.headers['Authorization'].split(" ")[1]
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = data['user_id']
        
        # Query all sessions for this user
        response = supabase.table("chat_sessions") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("updated_at", desc=True) \
            .execute()
        
        return jsonify({"sessions": response.data})
        
    except Exception as e:
        print(f"Error fetching sessions: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/sessions', methods=['POST'])
@token_required
def create_session():
    try:
        # Get user_id from JWT token
        token = request.headers['Authorization'].split(" ")[1]
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = data['user_id']
        
        print(f"Creating session for user_id: {user_id}")
        
        # Get request data
        req_data = request.json
        doc_id = req_data.get('doc_id')
        title = req_data.get('title', f"New Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        if not doc_id:
            # Get the most recent doc_id if not provided
            doc_response = supabase.table("documents") \
                .select("doc_id") \
                .limit(1) \
                .order("created_at", desc=True) \
                .execute()
            
            if not doc_response.data:
                print("No documents found for session creation.")
                # Create a default document or prompt user to upload one
                return jsonify({
                    "error": "No documents found. Please process a documentation first.",
                    "code": "NO_DOCUMENTS"
                }), 400
            
            doc_id = doc_response.data[0]['doc_id']
        
        print(f"Using doc_id: {doc_id} for session creation.")
        
        # Create new session
        session_id = str(uuid.uuid4())
        supabase.table("chat_sessions").insert({
            "id": session_id,
            "user_id": user_id,
            "doc_id": doc_id,
            "title": title
        }).execute()
        
        print(f"Session created successfully with session_id: {session_id}")
        
        return jsonify({
            "message": "Session created successfully",
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"Error creating session: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/sessions/<session_id>', methods=['DELETE'])
@token_required
def delete_session(session_id):
    try:
        # Get user_id from JWT token
        token = request.headers['Authorization'].split(" ")[1]
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = data['user_id']
        
        # Check if session belongs to user
        session = supabase.table("chat_sessions") \
            .select("*") \
            .eq("id", session_id) \
            .eq("user_id", user_id) \
            .execute()
            
        if not session.data:
            return jsonify({"error": "Session not found or you don't have permission"}), 403
            
        # Delete session (will cascade delete messages)
        result = supabase.table("chat_sessions") \
            .delete() \
            .eq("id", session_id) \
            .execute()
        print("Delete result:", result)

        return jsonify({"message": "Session deleted successfully"})
        
    except Exception as e:
        print(f"Error deleting session: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/sessions/<session_id>/messages', methods=['GET'])
@token_required
def get_messages(session_id):
    try:
        # Get user_id from JWT token
        token = request.headers['Authorization'].split(" ")[1]
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = data['user_id']
        
        # Check if session belongs to user
        session = supabase.table("chat_sessions") \
            .select("*") \
            .eq("id", session_id) \
            .eq("user_id", user_id) \
            .execute()
            
        if not session.data:
            return jsonify({"error": "Session not found or you don't have permission"}), 403
            
        # Get messages for this session
        messages = supabase.table("chat_messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("created_at") \
            .execute()
            
        return jsonify({"messages": messages.data})
        
    except Exception as e:
        print(f"Error fetching messages: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/sessions/<session_id>/messages', methods=['POST'])
@token_required
def create_message(session_id):
    try:
        # Get user_id from JWT token
        token = request.headers['Authorization'].split(" ")[1]
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = data['user_id']
        
        print(f"Creating message for session_id: {session_id}, user_id: {user_id}")
        
        # Check if session belongs to user
        session = supabase.table("chat_sessions") \
            .select("*") \
            .eq("id", session_id) \
            .eq("user_id", user_id) \
            .execute()
            
        if not session.data:
            print("Session not found or user does not have permission.")
            return jsonify({"error": "Session not found or you don't have permission"}), 403
            
        # Get request data
        req_data = request.json
        question = req_data.get('question')
        
        if not question:
            print("Question is required but not provided.")
            return jsonify({"error": "Question is required"}), 400
            
        # Create user message
        message_id = str(uuid.uuid4())
        supabase.table("chat_messages").insert({
            "id": message_id,
            "session_id": session_id,
            "is_user": True,
            "content": question
        }).execute()
        
        #print(f"User message created with message_id: {message_id}")
        
        # Get doc_id from the session
        doc_id = session.data[0]['doc_id']
        
        # Create embedding for the question
        question_embedding = create_embeddings(question)
        
        # Query Supabase for relevant document chunks using vector similarity
        query_response = supabase.rpc(
            "match_documents",
            {
                "query_embedding": question_embedding,
                "match_document_id": doc_id,
                "match_threshold": 0.2,
                "match_count": 3
            }
        ).execute()
        
        print(f"Query response: {query_response.data}")
        
        relevant_docs = query_response.data
        
        # Generate AI response
        if not relevant_docs:
            ai_response = "I couldn't find relevant information to answer your question in the provided documentation."
        else:
            # Construct context from relevant documents
            context = "\n\n".join([doc["content"] for doc in relevant_docs])
            
            # Generate response using Mistral AI
            ai_response = query_mistral_with_prefix(question, context)
        
        # Store AI response
        ai_message_id = str(uuid.uuid4())
        supabase.table("chat_messages").insert({
            "id": ai_message_id,
            "session_id": session_id,
            "is_user": False,
            "content": ai_response
        }).execute()
        
        # print(f"AI response created with message_id: {ai_message_id}")
        
        # Update session's updated_at timestamp
        supabase.table("chat_sessions") \
            .update({"updated_at": datetime.now().isoformat()}) \
            .eq("id", session_id) \
            .execute()
        
        return jsonify({
            "message": "Message created successfully",
            "response": ai_response
        })
        
    except Exception as e:
        print(f"Error creating message: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500







        

@app.route('/crawl', methods=['POST'])
@token_required
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
                    page_timeout=120000,  # 120 seconds timeout
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
            
            # Aggregate raw markdown from all pages
            aggregated_raw_markdown = "\n\n".join(
                [result.markdown.raw_markdown for result in results if result.markdown and hasattr(result.markdown, "raw_markdown")]
            )
            
            # Process the document and store each chunk separately
            chunker = SectionHeaderChunking()
            chunks = chunker.process_document(aggregated_raw_markdown)
            
            # Store the aggregated document in Supabase
            doc_id = str(uuid.uuid4())
            
            # Insert the document into the documents table
            supabase.table("documents").insert({
                "id": str(uuid.uuid4()),  # Unique UUID for the primary key
                "doc_id": doc_id,
                "url": url,
                "content": aggregated_raw_markdown
            }).execute()
            
            # Process all chunks at once more efficiently
            try:
                # Create embeddings for all chunks in batches
                all_embeddings = create_embeddings_batch(chunks)
                
                # Then store each embedding
                for idx, embedding in enumerate(all_embeddings):
                    supabase.table("document_chunks").insert({
                        "id": str(uuid.uuid4()),
                        "doc_id": doc_id,
                        "embedding": embedding,
                        "chunk_content": chunks[idx]  # Store the chunk content for reference
                    }).execute()
                    print(f"Stored embedding {idx+1}/{len(all_embeddings)}: {embedding[:5]}...")
                
                return jsonify({
                    "message": "Document crawled and stored successfully",
                    "doc_id": doc_id,
                    "chunk_count": len(chunks),
                    "markdown_length": len(aggregated_raw_markdown)
                })
                
            except Exception as embedding_error:
                print(f"Embedding Generation Error: {str(embedding_error)}")
                return jsonify({
                    "error": "Failed to generate embeddings",
                    "details": str(embedding_error)
                }), 500
        
        except Exception as async_error:
            print(f"Async Crawler Execution Error: {str(async_error)}")
            return jsonify({
                "error": "Failed to crawl the documentation",
                "details": str(async_error)
            }), 500
        
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
@token_required
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
                "match_threshold": 0.8,
                "match_count": 3
            }
        ).execute()
        
        print(f"Query response: {query_response.data}")
        
        relevant_docs = query_response.data
        
        if not relevant_docs:
            return jsonify({
                "answer": "I couldn't find relevant information to answer your question in the provided documentation."
            })
            
        # Construct context from relevant documents
        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        
        # Generate response using Mistral AI with prefix
        response = query_mistral_with_prefix(question, context)
        
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


def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Create embeddings for a batch of text chunks efficiently."""
    if not texts:
        print("No texts provided for embedding.")
        return []
    
    print(f"Creating embeddings for {len(texts)} chunks")
    
    try:
        # Process texts directly with optimized batching
        embeddings = get_embeddings_by_chunks_v2(texts, chunk_size=50, batch_size=8)
        
        if not embeddings:
            raise ValueError("No embeddings were returned")
        
        print(f"Successfully created {len(embeddings)} embeddings")
        return embeddings
        
    except Exception as e:
        print(f"Error creating batch embeddings: {e}")
        raise

def create_embeddings(text: str) -> List[float]:
    try:
        # Pass a single text string, not wrapped in a list
        chunk_embeddings = get_embeddings_by_chunks_v2([text], chunk_size=50)
        if not chunk_embeddings:
            raise ValueError("No embeddings were returned.")
        
        # Convert directly into a (num_chunks × embedding_dim) array
        arr = np.array(chunk_embeddings)             # shape: (num_chunks, embedding_dim)
        
        aggregated_embedding = arr.mean(axis=0)      # average across chunks
        print(f"Created embedding of dimension {aggregated_embedding.shape[0]}")
        return aggregated_embedding.tolist()
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        raise


def query_mistral_with_prefix(question: str, context: str) -> str:
    """Query Mistral AI with the question and context using a prefix and log token usage."""
    prefix = "Coding Assistant:"
    
    #Answer using the provided context:
    prompt = f"""

    Context:
    {context}

    Question: {question}

    """
    
    # Log token count for the prompt
    token_count = get_token_count(prompt)
    print(f"Prompt token count: {token_count}")
    
    chat_response = mistral_client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": prefix, "prefix": True}
        ],
        temperature=0.7,
        top_p=0.5,
        max_tokens=1000
    )
    
    return chat_response.choices[0].message.content

@app.route('/documents', methods=['GET'])
@token_required
def get_documents():
    try:
        # Get user_id from JWT token
        token = request.headers['Authorization'].split(" ")[1]
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = data['user_id']
        
        # Fetch all documents
        response = supabase.table("documents") \
            .select("*") \
            .order("created_at", desc=True) \
            .execute()
            
        return jsonify(response.data)
        
    except Exception as e:
        print(f"Error fetching documents: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500



def get_embeddings_by_chunks_v2(data: List[str], chunk_size: int = 50, delay: float = 1.0, 
                                max_retries: int = 5, batch_size: int = 8) -> List[List[float]]:
    """Process text data and retrieve embeddings in batches."""
    if not data:
        print("No data provided for embedding.")
        return []

    # Process each text entry in batches
    total_texts = len(data)
    embeddings = []
    
    # Process texts in batches to reduce API calls
    for batch_start in range(0, total_texts, batch_size):
        batch_end = min(batch_start + batch_size, total_texts)
        current_batch = data[batch_start:batch_end]
        print(f"Processing batch {batch_start//batch_size + 1} ({batch_start}-{batch_end-1} of {total_texts} texts)")
        
        retries = 0
        while retries < max_retries:
            try:
                # Send batch directly to API
                response = mistral_client.embeddings.create(model="mistral-embed", inputs=current_batch)
                
                if not response.data:
                    print(f"No embeddings returned for batch {batch_start//batch_size + 1}.")
                    break
                
                # Extract embeddings from response
                batch_embeddings = [d.embedding for d in response.data]
                embeddings.extend(batch_embeddings)
                print(f"Successfully processed {len(batch_embeddings)} texts in this batch")
                
                # Add delay between batches to avoid rate limits
                time.sleep(delay)
                break  # Exit retry loop on success
                
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    retries += 1
                    sleep_time = delay * (2 ** retries) + random.uniform(0, 1.0)
                    print(f"Rate limit hit on batch {batch_start//batch_size + 1}. Retrying in {sleep_time:.2f} seconds (Attempt {retries}/{max_retries})...")
                    time.sleep(sleep_time)
                else:
                    print(f"Error processing batch {batch_start//batch_size + 1}: {str(e)}")
                    raise e
        else:
            print(f"Exceeded maximum retries for batch {batch_start//batch_size + 1}.")
            raise Exception("Exceeded maximum retries due to rate limit errors.")
    
    if not embeddings:
        print("No embeddings were returned for any texts.")
    
    return embeddings

# def get_embeddings_by_chunks_v2(
#     data: List[str],
#     chunk_size: int,
#     batch_size: int = 8,
#     delay: float = 0.5,
#     max_retries: int = 5
# ) -> List[List[float]]:
#     """
#     Splits `data` (a list of texts) into chunks of `chunk_size` items,
#     then sends those chunks in batches of `batch_size` to Mistral's embeddings API.
#     Returns a flat list of embeddings corresponding 1:1 to the original `data` items.
#     """
#     if not data:
#         print("No data provided for embedding.")
#         return []

#     # 1) Split into chunks of up to chunk_size texts each
#     chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
#     all_embeddings: List[List[float]] = []

#     # 2) Group those chunks into batches of `batch_size` API calls
#     for batch_start in range(0, len(chunks), batch_size):
#         batch = chunks[batch_start : batch_start + batch_size]
#         # flatten batch into one list of texts
#         inputs = [text for chunk in batch for text in chunk]

#         # 3) Retry on rate limit up to max_retries
#         for attempt in range(1, max_retries + 1):
#             try:
#                 resp = mistral_client.embeddings.create(
#                     model="mistral-embed",
#                     inputs=inputs
#                 )
#                 # extend with each returned embedding
#                 all_embeddings.extend([d.embedding for d in resp.data])
#                 # brief pause (optional, you can reduce or remove)
#                 time.sleep(delay)
#                 break
#             except Exception as e:
#                 err = str(e).lower()
#                 if "rate limit" in err or "429" in err:
#                     backoff = delay * (2 ** attempt) + random.uniform(0, 0.5)
#                     print(f"Rate limit on batch {batch_start//batch_size + 1}, "
#                           f"retry {attempt}/{max_retries} sleeping {backoff:.1f}s…")
#                     time.sleep(backoff)
#                 else:
#                     # non–rate-limit error, give up immediately
#                     print(f"Error on batch {batch_start//batch_size + 1}: {e}")
#                     raise
#         else:
#             # exhausted retries
#             print(f"Skipping batch {batch_start//batch_size + 1} after {max_retries} rate‐limit retries.")
#             continue

#     if not all_embeddings:
#         print("No embeddings were returned for any batches.")
#     return all_embeddings



# Define a simple sentence tokenizer using regex
def simple_sentence_tokenize(text):
    sentence_endings = re.compile(r'(?<=[.!?]) +')
    sentences = sentence_endings.split(text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Update the SectionHeaderChunking class to use the new tokenizer
class SectionHeaderChunking:
    def __init__(self, header_pattern=r'<h[1-6]>.*?</h[1-6]>', window_size=500, step=200):
        self.header_pattern = header_pattern
        self.window_size = window_size
        self.step = step

    def chunk_by_headers(self, html_content):
        # Use BeautifulSoup to parse HTML and extract text
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text()
        
        # Split text by section headers
        sections = re.split(self.header_pattern, text)
        return sections

    def sliding_window_chunking(self, text):
        words = text.split()
        chunks = []
        for i in range(0, len(words) - self.window_size + 1, self.step):
            chunks.append(' '.join(words[i:i + self.window_size]))
        return chunks

    def process_document(self, html_content):
        sections = self.chunk_by_headers(html_content)
        all_chunks = []
        for section in sections:
            # Fallback to simple sentence splits if section is too small
            if len(section.split()) < self.window_size:
                sentences = simple_sentence_tokenize(section)
                all_chunks.extend(sentences)
            else:
                all_chunks.extend(self.sliding_window_chunking(section))
        return all_chunks




@app.route('/documents/<doc_id>/chunks', methods=['GET'])
@token_required
def get_document_chunks(doc_id):
    try:
        # Get user_id from JWT token
        token = request.headers['Authorization'].split(" ")[1]
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = data['user_id']
        
        # Fetch all chunks for the specified document
        response = supabase.table("document_chunks") \
            .select("*") \
            .eq("doc_id", doc_id) \
            .order("created_at", desc=True) \
            .execute()
            
        return jsonify(response.data)
        
    except Exception as e:
        print(f"Error fetching document chunks: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)