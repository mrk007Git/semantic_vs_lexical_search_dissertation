import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Use the article provider system directly instead of preprocess
try:
    from csv_writer import write_results_to_csv
except ImportError:
    from scripts.csv_writer import write_results_to_csv

def lexical_search(query, corpus, top_n=10):
    """Perform lexical search using Bag of Words (BoW)."""
    docs = corpus.tolist()
    vectorizer = CountVectorizer(stop_words='english')  # Use CountVectorizer for raw word counts
    
    # Fit vectorizer on corpus first, then transform query separately
    corpus_matrix = vectorizer.fit_transform(docs)
    query_vector = vectorizer.transform([query])
    
    cosine_sim = cosine_similarity(query_vector, corpus_matrix).flatten()  # Compute cosine similarity
    top_indices = cosine_sim.argsort()[-top_n:][::-1]  # Get top N results
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
    
    top_n = 100
    query = "advanced cardiac life support training in UK medical schools"
    indices, scores = lexical_search(query, df['content'], top_n)
    results = df.iloc[indices]

    # Display results
    for i, row in results.iterrows():
        print(f"\n--- {row['filename']} ---\n{row['content'][:50]}...\nScore: {scores[results.index.get_loc(i)]:.4f}")

    # Write results to CSV
    output_file = "lexical_search_results.csv"
    write_results_to_csv(results, scores, output_file)

    # Filter for a specific PMC number
    pmc_number = "PMC1342422"
    filtered_result = results[results['filename'].str.startswith(pmc_number)]

    # Print the filtered result with its lexical score
    if not filtered_result.empty:
        print(f"\nFile containing PMC number {pmc_number}:")
        for i, row in filtered_result.iterrows():
            score = scores[results.index.get_loc(i)]
            print(f"Filename: {row['filename']}")
            print(f"Content: {row['content'][:100]}...")  # Print the first 100 characters of the content
            print(f"Lexical Score: {score:.4f}")
    else:
        print(f"\nNo file found containing PMC number {pmc_number}.")
