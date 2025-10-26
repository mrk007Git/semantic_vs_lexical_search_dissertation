import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Use the article provider system directly instead of preprocess
import pickle
import os
try:
    from csv_writer import write_results_to_csv
except ImportError:
    from scripts.csv_writer import write_results_to_csv

# Filepath to save the precomputed TF-IDF matrix and vectorizer
TFIDF_MATRIX_FILE = "tfidf_matrix.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"

def precompute_tfidf(corpus, save_path_matrix, save_path_vectorizer):
    """Precompute the TF-IDF matrix for the corpus and save it."""
    print("Precomputing TF-IDF matrix...")
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Save the TF-IDF matrix and vectorizer to disk
    with open(save_path_matrix, 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    with open(save_path_vectorizer, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"TF-IDF matrix saved to {save_path_matrix}")
    print(f"Vectorizer saved to {save_path_vectorizer}")
    return vectorizer, tfidf_matrix

def load_precomputed_tfidf(matrix_path, vectorizer_path):
    """Load the precomputed TF-IDF matrix and vectorizer from disk."""
    print("Loading precomputed TF-IDF matrix and vectorizer...")
    with open(matrix_path, 'rb') as f:
        tfidf_matrix = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer, tfidf_matrix

def lexical_search(query, vectorizer, tfidf_matrix, top_n=10):
    """Perform lexical search using precomputed TF-IDF matrix."""
    query_vector = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-top_n:][::-1]
    return top_indices, cosine_sim[top_indices]

if __name__ == "__main__":
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
    
    corpus = df['content'].tolist()

    # Check if precomputed TF-IDF matrix exists
    if not os.path.exists(TFIDF_MATRIX_FILE) or not os.path.exists(VECTORIZER_FILE):
        # Precompute and save TF-IDF matrix and vectorizer
        vectorizer, tfidf_matrix = precompute_tfidf(corpus, TFIDF_MATRIX_FILE, VECTORIZER_FILE)
    else:
        # Load precomputed TF-IDF matrix and vectorizer
        vectorizer, tfidf_matrix = load_precomputed_tfidf(TFIDF_MATRIX_FILE, VECTORIZER_FILE)

    # Perform lexical search
    query = "advanced cardiac life support training in UK medical schools"
    indices, scores = lexical_search(query, vectorizer, tfidf_matrix)
    results = df.iloc[indices]

    # Display results
    for i, row in results.iterrows():
        print(f"\n--- {row['filename']} ---\n{row['content'][:50]}...\nScore: {scores[results.index.get_loc(i)]:.4f}")

    # Write results to CSV
    output_file = "tf_idf_search_results.csv"
    write_results_to_csv(results, scores, output_file)