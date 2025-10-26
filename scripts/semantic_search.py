import os
import pandas as pd
import openai
import numpy as np
import pickle
from tqdm import tqdm  # Import tqdm for progress bars
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from csv_writer import write_results_to_csv
except ImportError:
    from scripts.csv_writer import write_results_to_csv

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure the API key is set as an environment variable

def clean_text(text):
    # Remove excess line breaks, rejoin broken lines, etc.
    lines = text.splitlines()
    cleaned = ' '.join(line.strip() for line in lines if line.strip())
    return cleaned

def embed_text(text):
    text = clean_text(text)
    """Generate embeddings for a single text using OpenAI's API."""
    first_line = text.strip().splitlines()[0] if text.strip() else "[EMPTY TEXT]"
    print(f"📄 Embedding first line: {first_line}")
    
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        print(f"✅ Got embedding, length: {len(embedding)}")
        return np.array(embedding)
    except Exception as e:
        print(f"❌ Error getting embedding: {e}")
        return np.zeros(1536)  # fallback to something to avoid breaking


def save_embedding(embedding, file_path):
    """Save the embedding for a single file."""
    with open(file_path, 'wb') as f:
        pickle.dump(embedding, f)

def load_embedding(file_path):
    """Load the embedding for a single file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def embed_and_save_file(row, embeddings_dir):
    """Embed and save a single file."""
    embedding_file = os.path.join(embeddings_dir, f"{row['filename']}.pkl")
    if os.path.exists(embedding_file):
        print(f"✅ Embedding already exists for {row['filename']}. Skipping.")
        return

    try:
        embedding = embed_text(row['content'])
        save_embedding(embedding, embedding_file)
        print(f"✅ Successfully embedded and saved {row['filename']}")
    except Exception as e:
        print(f"❌ Error embedding {row['filename']}: {e}")

def embed_and_save_corpus(df, embeddings_dir, max_workers=4):
    """Embed and save embeddings for each file in the corpus using parallel processing with progress bar."""
    os.makedirs(embeddings_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a tqdm progress bar
        with tqdm(total=len(df), desc="Embedding and saving files") as progress_bar:
            futures = [
                executor.submit(embed_and_save_file, row, embeddings_dir)
                for _, row in df.iterrows()
            ]
            for future in as_completed(futures):
                try:
                    future.result()  # Ensure exceptions are raised
                except Exception as e:
                    print(f"❌ Error in parallel task: {e}")
                finally:
                    progress_bar.update(1)  # Update the progress bar

def load_corpus_embeddings(df, embeddings_dir):
    """Load all embeddings from the saved files, generating them if they don't exist."""
    embeddings = []
    missing_embeddings = []
    
    # First pass: check which embeddings exist
    for _, row in df.iterrows():
        embedding_file = os.path.join(embeddings_dir, f"{row['filename']}.pkl")
        if os.path.exists(embedding_file):
            embeddings.append(load_embedding(embedding_file))
        else:
            missing_embeddings.append((row['filename'], row['content']))
            embeddings.append(None)  # Placeholder
    
    # Generate missing embeddings
    if missing_embeddings:
        print(f"Generating {len(missing_embeddings)} missing embeddings...")
        for i, (filename, content) in enumerate(tqdm(missing_embeddings, desc="Generating embeddings")):
            # Generate embedding
            embedding = embed_text(content)
            
            # Save embedding for future use
            embedding_file = os.path.join(embeddings_dir, f"{filename}.pkl")
            os.makedirs(os.path.dirname(embedding_file), exist_ok=True)
            save_embedding(embedding, embedding_file)
            
            # Find the index where this embedding should go
            for j, emb in enumerate(embeddings):
                if emb is None and df.iloc[j]['filename'] == filename:
                    embeddings[j] = embedding
                    break
    
    return np.array(embeddings)

def semantic_search(query, corpus_embeddings, top_n=10):
    """Perform semantic search using cosine similarity."""
    print("Embedding query...")
    query_embedding = embed_text(query)
    
    # Debug: Print query embedding
    print("Query embedding:", query_embedding[:5])  # Print first 5 values for brevity
    
    # Normalize the query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Debug: Print normalized query embedding
    print("Normalized query embedding:", query_embedding[:5])
    
    # Normalize the corpus embeddings
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    normalized_corpus_embeddings = corpus_embeddings / (np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + epsilon)
    
    # Debug: Print a few normalized corpus embeddings
    print("Sample normalized corpus embeddings:", normalized_corpus_embeddings[:3, :5])  # Print first 3 embeddings, first 5 values
    
    print("Calculating cosine similarity...")
    cos_scores = np.dot(normalized_corpus_embeddings, query_embedding)
    
    # Debug: Print cosine similarity scores
    print("Cosine similarity scores:", cos_scores[:10])  # Print first 10 scores
    
    top_indices = np.argsort(cos_scores)[-top_n:][::-1]
    return top_indices, cos_scores[top_indices]

if __name__ == "__main__":
    print("Loading articles...")
    # Load articles using article provider system
    try:
        from provider_config import get_article_provider
        provider = get_article_provider()
        df = provider.get_articles()
        print(f"Loaded {len(df)} articles from {provider.__class__.__name__}")
    except ImportError:
        print("❌ Article provider system not available. Please install and configure the provider system.")
        exit(1)
    except Exception as e:
        print(f"❌ Error loading articles: {e}")
        exit(1)
    
    print(df.head())

    # Directory to save individual embeddings
    embeddings_dir = 'models/embeddings'

    # Embed and save embeddings file by file
    print("Embedding and saving corpus...")
    embed_and_save_corpus(df, embeddings_dir)

    # Load all embeddings
    print("Loading embeddings...")
    corpus_embeddings = load_corpus_embeddings(df, embeddings_dir)

    # Perform semantic search
    query = "advanced cardiac life support training in UK medical schools"
    print("Performing semantic search...")
    indices, scores = semantic_search(query, corpus_embeddings)
    results = df.iloc[indices]

    # Display the results
    print("Displaying results...")
    for i, row in results.iterrows():
        print(f"\n--- {row['filename']} ---\n{row['content'][:500]}...\nScore: {scores[results.index.get_loc(i)]:.4f}")

    # Write results to CSV
    output_file = "semantic_search_results.csv"
    write_results_to_csv(results, scores, output_file)
