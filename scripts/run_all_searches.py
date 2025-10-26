import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import provider system
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    from provider_config import get_article_provider
    PROVIDER_SYSTEM_AVAILABLE = True
    print("✅ Provider system loaded successfully")
except ImportError:
    PROVIDER_SYSTEM_AVAILABLE = False
    get_article_provider = None  # Ensure it's defined
    print("⚠️  Provider system not available. Users must install and configure article_provider system.")

# Import preprocess functions - no longer needed since we use article_provider directly
# Legacy imports removed - users should configure article_provider system instead

try:
    from scripts.lexical_search import lexical_search
except ImportError:
    from lexical_search import lexical_search

try:
    from scripts.semantic_search import semantic_search, load_corpus_embeddings
except ImportError:
    from semantic_search import semantic_search, load_corpus_embeddings

try:
    from scripts.tf_idf_search import precompute_tfidf, load_precomputed_tfidf, lexical_search as tf_idf_search
except ImportError:
    from tf_idf_search import precompute_tfidf, load_precomputed_tfidf, lexical_search as tf_idf_search
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Filepaths for precomputed TF-IDF matrix and vectorizer
TFIDF_MATRIX_FILE = "tfidf_matrix.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"

# Directory for embeddings
EMBEDDINGS_DIR = "models/embeddings"

def load_articles_with_provider(limit=None, prefer_database=True):
    """
    Load articles using the provider system with enhanced error handling.
    
    Args:
        limit: Maximum number of articles to load
        prefer_database: Whether to prefer database over files
        
    Returns:
        DataFrame with articles and metadata about the source
    """
    if PROVIDER_SYSTEM_AVAILABLE and get_article_provider is not None:
        try:
            provider = get_article_provider()
            if provider is None:
                print("⚠️  Provider system returned None. Falling back to simple loading...")
                raise ValueError("Provider is None")
                
            provider_info = provider.get_provider_info()
            print(f"📊 Using provider: {provider_info.get('provider_type', 'Unknown')}")
            
            if provider_info.get('active_provider'):
                print(f"   Active provider: {provider_info['active_provider']}")
            
            print(f"   Available: {provider_info.get('is_available', False)}")
            print(f"   Total articles: {provider_info.get('total_articles', 0)}")
            
            df = provider.get_articles(limit)
            print(f"✅ Loaded {len(df)} articles via provider system")
            return df, provider_info
            
        except Exception as e:
            print(f"❌ Provider system failed: {e}")
            print("   Falling back to simple loading...")
    
    # Fallback: Try to create a minimal provider from article_provider system
    print("📁 Using fallback article loading...")
    try:
        # Import and use the provider system directly
        from article_provider import FileBasedArticleProvider
        # Let the provider handle its own configuration/defaults
        fallback_provider = FileBasedArticleProvider(".")  # Current directory as minimal fallback
        df = fallback_provider.get_articles(limit)
        fallback_info = {
            'provider_type': 'Fallback',
            'active_provider': 'Direct FileBasedArticleProvider',
            'is_available': len(df) > 0,
            'total_articles': len(df)
        }
    except Exception as fallback_error:
        print(f"❌ All providers failed: {fallback_error}")
        print("   Users must configure their own data sources via article_provider system")
        # Return empty DataFrame - users must set up their providers
        df = pd.DataFrame(columns=['filename', 'content'])
        fallback_info = {
            'provider_type': 'Empty',
            'active_provider': 'No data source configured',
            'is_available': False,
            'total_articles': 0
        }
    return df, fallback_info

def run_lexical_search(df, query, top_n=10):
    """Run lexical search and return results."""
    indices, scores = lexical_search(query, df['content'], top_n=top_n)
    
    # Ensure indices are within bounds
    valid_indices = [i for i in indices if i < len(df)]
    if len(valid_indices) < len(indices):
        print(f"Warning: {len(indices) - len(valid_indices)} indices were out of bounds in lexical search")
    
    results = df.iloc[valid_indices].copy()  # Use .copy() to avoid SettingWithCopyWarning
    results.loc[:, 'Score'] = scores[:len(valid_indices)]  # Match scores to valid indices
    results.loc[:, 'Search Type'] = 'BoW'
    return results

def run_tf_idf_search(df, query, top_n=10):
    """Run TF-IDF search and return results."""
    corpus = df['content'].tolist()

    # Always recompute TF-IDF when using database articles to ensure indices match
    # Check if we have database articles (they have different structure)
    has_database_columns = 'Id' in df.columns or 'Title' in df.columns
    
    # If using database articles, always recompute TF-IDF to ensure indices match
    if has_database_columns:
        print("Database articles detected, recomputing TF-IDF matrix for proper indexing...")
        vectorizer, tfidf_matrix = precompute_tfidf(corpus, "temp_tfidf_matrix.pkl", "temp_tfidf_vectorizer.pkl")
    else:
        # Check if precomputed TF-IDF matrix exists for file-based articles
        if not os.path.exists(TFIDF_MATRIX_FILE) or not os.path.exists(VECTORIZER_FILE):
            vectorizer, tfidf_matrix = precompute_tfidf(corpus, TFIDF_MATRIX_FILE, VECTORIZER_FILE)
        else:
            vectorizer, tfidf_matrix = load_precomputed_tfidf(TFIDF_MATRIX_FILE, VECTORIZER_FILE)
            # Verify that the matrix size matches the current corpus
            if tfidf_matrix.shape[0] != len(corpus):
                print(f"TF-IDF matrix size ({tfidf_matrix.shape[0]}) doesn't match corpus size ({len(corpus)}). Recomputing...")
                vectorizer, tfidf_matrix = precompute_tfidf(corpus, TFIDF_MATRIX_FILE, VECTORIZER_FILE)

    indices, scores = tf_idf_search(query, vectorizer, tfidf_matrix, top_n=top_n)
    
    # Ensure indices are within bounds
    valid_indices = [i for i in indices if i < len(df)]
    if len(valid_indices) < len(indices):
        print(f"Warning: {len(indices) - len(valid_indices)} indices were out of bounds")
    
    results = df.iloc[valid_indices].copy()  # Use .copy() to avoid SettingWithCopyWarning
    results.loc[:, 'Score'] = scores[:len(valid_indices)]  # Match scores to valid indices
    results.loc[:, 'Search Type'] = 'TF-IDF'
    return results

def run_semantic_search(df, query, top_n=10):
    """Run semantic search and return results."""
    # Load embeddings
    corpus_embeddings = load_corpus_embeddings(df, EMBEDDINGS_DIR)

    # Perform semantic search
    indices, scores = semantic_search(query, corpus_embeddings, top_n=top_n)
    
    # Ensure indices are within bounds
    valid_indices = [i for i in indices if i < len(df)]
    if len(valid_indices) < len(indices):
        print(f"Warning: {len(indices) - len(valid_indices)} indices were out of bounds in semantic search")
    
    results = df.iloc[valid_indices].copy()  # Use .copy() to avoid SettingWithCopyWarning
    results.loc[:, 'Score'] = scores[:len(valid_indices)]  # Match scores to valid indices
    results.loc[:, 'Search Type'] = 'Semantic'
    return results

def get_all_search_results(df, query):
    """Get all search results (not limited to top N) for all three methods."""
    print("Getting all search results (unlimited)...")
    
    # Run all searches with no limit (use len(df) as top_n)
    all_bow_results = run_lexical_search(df, query, top_n=len(df))
    all_tfidf_results = run_tf_idf_search(df, query, top_n=len(df))
    all_semantic_results = run_semantic_search(df, query, top_n=len(df))
    
    # Add rank for each method
    all_bow_results = all_bow_results.copy()
    all_bow_results['Rank'] = all_bow_results['Score'].rank(ascending=False, method='first').astype(int)
    
    all_tfidf_results = all_tfidf_results.copy()
    all_tfidf_results['Rank'] = all_tfidf_results['Score'].rank(ascending=False, method='first').astype(int)
    
    all_semantic_results = all_semantic_results.copy()
    all_semantic_results['Rank'] = all_semantic_results['Score'].rank(ascending=False, method='first').astype(int)
    
    # Normalize scores within each search method (0-1 range)
    scaler = MinMaxScaler()
    all_bow_results['Normalized_Score'] = scaler.fit_transform(all_bow_results[['Score']])
    all_tfidf_results['Normalized_Score'] = scaler.fit_transform(all_tfidf_results[['Score']])
    all_semantic_results['Normalized_Score'] = scaler.fit_transform(all_semantic_results[['Score']])
    
    # Combine all results
    all_combined = pd.concat([all_bow_results, all_tfidf_results, all_semantic_results], ignore_index=True)
    
    # Prepare for CSV output
    if 'Title' in all_combined.columns:
        # Use Title from database
        all_results_csv = all_combined[['filename', 'Score', 'Normalized_Score', 'Rank', 'Search Type', 'Title']].copy()
        all_results_csv.rename(columns={'Title': 'First Line'}, inplace=True)
    else:
        # Fallback to first line of content for file-based articles
        all_results_csv = all_combined[['filename', 'Score', 'Normalized_Score', 'Rank', 'Search Type', 'content']].copy()
        all_results_csv.rename(columns={'content': 'First Line'}, inplace=True)
        all_results_csv['First Line'] = all_results_csv['First Line'].apply(lambda x: x.strip().splitlines()[0] if x.strip() else "[EMPTY TEXT]")
    
    # Sort by Search Type and then by Rank
    all_results_csv = all_results_csv.sort_values(['Search Type', 'Rank'])
    
    return all_results_csv

def visualize_results(combined_results):
    """Visualize the ranking of files according to each search type."""
    # Debug: Print the combined results to verify data
    print("Combined Results:")
    print(combined_results.head())
    print("Search Type Counts:")
    print(combined_results['Search Type'].value_counts())

    # Sort results by score for better visualization
    combined_results = combined_results.sort_values(by="Score", ascending=False)

    # Reorder the Search Type categories
    combined_results['Search Type'] = pd.Categorical(
        combined_results['Search Type'],
        categories=['Semantic', 'BoW', 'TF-IDF'],  # Ensure "BOW" is included
        ordered=True
    )

    # Create a bar plot with custom colors
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=combined_results,
        x="filename",
        y="Score",
        hue="Search Type",
        dodge=True,
        palette={"Semantic": "blue", "BoW": "orange", "TF-IDF": "green"}  # Custom colors
    )

    # Customize the plot
    plt.title("Search Results Comparison by Search Type", fontsize=16)
    plt.xlabel("Filename", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.legend(title="Search Type", fontsize=10)
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig("search_results_comparison.png")
    print("Visualization saved as 'search_results_comparison.png'.")

    # Show the plot
    # plt.show()

def visualize_results_heatmap(combined_results):
    """Visualize the normalized cosine similarity scores across search methods as a heatmap."""
    # Normalize scores within each search method
    normalized_results = combined_results.copy()
    scaler = MinMaxScaler()

    for search_type in ['Semantic', 'BoW', 'TF-IDF']:
        mask = normalized_results['Search Type'] == search_type
        normalized_results.loc[mask, 'Score'] = scaler.fit_transform(
            normalized_results.loc[mask, ['Score']]
        )

    # Pivot the data for the heatmap
    heatmap_data = normalized_results.pivot_table(
        index='filename', columns='Search Type', values='Score', aggfunc='first'
    )

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        cbar_kws={'label': 'Normalized Score'}
    )

    # Customize the plot
    plt.title("Heatmap of Normalized Cosine Similarity Scores", fontsize=16)
    plt.xlabel("Search Method", fontsize=12)
    plt.ylabel("Document (Filename)", fontsize=12)
    plt.tight_layout()

    # Save the heatmap as an image
    plt.savefig("search_results_heatmap.png")
    print("Heatmap saved as 'search_results_heatmap.png'.")

    # Show the heatmap
    # plt.show()

def get_unique_filename(base_name, extension="csv"):
    """Generate a unique filename by appending a number if the file already exists."""
    counter = 1
    unique_name = f"{base_name}.{extension}"
    while os.path.exists(unique_name):
        unique_name = f"{base_name}_{counter}.{extension}"
        counter += 1
    return unique_name

def save_pmc_urls(combined_results, output_txt_file="pmc_urls.txt"):
    combined_results = combined_results.sort_values(by="Score", ascending=False)
    """Extract PMC numbers from filenames, format them into URLs, and save to a text file."""
    # Extract PMC numbers from filenames
    pmc_urls = combined_results['filename'].apply(
        lambda x: f"https://pmc.ncbi.nlm.nih.gov/articles/{x.split('_')[0]}"
    )

    # Save the URLs to a text file
    with open(output_txt_file, "w") as f:
        for url in pmc_urls:
            f.write(url + "\n")

    print(f"PMC URLs successfully saved to {output_txt_file}.")

def get_top_n_with_ranks(results, method, n=10):
    results = results.copy()
    results['rank_in_' + method] = results['Score'].rank(ascending=False, method='first').astype(int)
    results = results.sort_values(by='Score', ascending=False).head(n)
    return results

def get_rank(filename, results):
    row = results[results['filename'] == filename]
    if not row.empty:
        return int(row.iloc[0]['rank'])
    return None

def multi_term_search(search_terms, top_n=10, use_database=True, db_limit=1000):
    # Load articles using enhanced provider system
    print(f"🔍 Starting multi-term search for {len(search_terms)} terms...")
    df, provider_info = load_articles_with_provider(limit=db_limit if use_database else None, 
                                                   prefer_database=use_database)
    
    if df is None or len(df) == 0:
        print("❌ No articles loaded. Cannot perform search.")
        return pd.DataFrame()
    
    print(f"📚 Loaded {len(df)} articles from {provider_info.get('active_provider', 'unknown source')}")
        
    all_rows = []
    for term in search_terms:
        print(f"🔎 Processing term: {term}")
        # Run all searches
        bow = run_lexical_search(df, term, top_n=top_n*5)  # Get more results to ensure we can rank
        tfidf = run_tf_idf_search(df, term, top_n=top_n*5)
        semantic = run_semantic_search(df, term, top_n=top_n*5)

        # Add rank columns for each method
        bow = bow.copy()
        bow['rank_in_BoW'] = bow['Score'].rank(ascending=False, method='first').astype(int)
        tfidf = tfidf.copy()
        tfidf['rank_in_TFIDF'] = tfidf['Score'].rank(ascending=False, method='first').astype(int)
        semantic = semantic.copy()
        semantic['rank_in_Semantic'] = semantic['Score'].rank(ascending=False, method='first').astype(int)

        # Get top N for each method
        top_bow = bow.sort_values(by='Score', ascending=False).head(top_n)
        top_tfidf = tfidf.sort_values(by='Score', ascending=False).head(top_n)
        top_semantic = semantic.sort_values(by='Score', ascending=False).head(top_n)

        # Union of all filenames in any top 10
        filenames = set(top_bow['filename']) | set(top_tfidf['filename']) | set(top_semantic['filename'])

        for filename in filenames:
            # Get row info from each method if present
            bow_row = bow[bow['filename'] == filename]
            tfidf_row = tfidf[tfidf['filename'] == filename]
            semantic_row = semantic[semantic['filename'] == filename]
            
            # Get title from the original dataframe if available
            original_row = df[df['filename'] == filename]
            title = original_row['Title'].iloc[0] if not original_row.empty and 'Title' in df.columns else filename
            
            all_rows.append({
                'search_term': term,
                'title': title,
                'filename': filename,
                'score_BoW': float(bow_row['Score']) if not bow_row.empty else None,
                'score_TFIDF': float(tfidf_row['Score']) if not tfidf_row.empty else None,
                'score_Semantic': float(semantic_row['Score']) if not semantic_row.empty else None,
                'rank_in_BoW': int(bow_row['rank_in_BoW']) if not bow_row.empty else None,
                'rank_in_TFIDF': int(tfidf_row['rank_in_TFIDF']) if not tfidf_row.empty else None,
                'rank_in_Semantic': int(semantic_row['rank_in_Semantic']) if not semantic_row.empty else None,
            })
    return pd.DataFrame(all_rows)

def multi_term_top10_full_ranking(search_terms, top_n=10, use_database=True, db_limit=1000):
    # Load articles using enhanced provider system
    print(f"🏆 Starting multi-term full ranking search for {len(search_terms)} terms...")
    df, provider_info = load_articles_with_provider(limit=db_limit if use_database else None, 
                                                   prefer_database=use_database)
    
    if df is None or len(df) == 0:
        print("❌ No articles loaded. Cannot perform search.")
        return pd.DataFrame()
    
    print(f"📚 Loaded {len(df)} articles from {provider_info.get('active_provider', 'unknown source')}")
        
    all_rows = []
    for term in search_terms:
        print(f"🔎 Processing term: {term}")
        # Run all searches on the full corpus
        bow = run_lexical_search(df, term, top_n=len(df))
        tfidf = run_tf_idf_search(df, term, top_n=len(df))
        semantic = run_semantic_search(df, term, top_n=len(df))

        # Add rank columns for each method (1 = best)
        bow = bow.copy()
        bow['rank_in_BoW'] = bow['Score'].rank(ascending=False, method='first').astype(int)
        tfidf = tfidf.copy()
        tfidf['rank_in_TFIDF'] = tfidf['Score'].rank(ascending=False, method='first').astype(int)
        semantic = semantic.copy()
        semantic['rank_in_Semantic'] = semantic['Score'].rank(ascending=False, method='first').astype(int)

        # Get top N for each method
        top_bow = bow.sort_values(by='Score', ascending=False).head(top_n)
        top_tfidf = tfidf.sort_values(by='Score', ascending=False).head(top_n)
        top_semantic = semantic.sort_values(by='Score', ascending=False).head(top_n)

        # Union of all filenames in any top 10 (should be up to 30 unique articles)
        filenames = set(top_bow['filename']) | set(top_tfidf['filename']) | set(top_semantic['filename'])

        for method, top_df in [('BoW', top_bow), ('TF-IDF', top_tfidf), ('Semantic', top_semantic)]:
            for _, row in top_df.iterrows():
                filename = row['filename']
                # Get the rank and score for this filename in all methods
                bow_row = bow[bow['filename'] == filename]
                tfidf_row = tfidf[tfidf['filename'] == filename]
                semantic_row = semantic[semantic['filename'] == filename]
                
                # Get title from the original dataframe if available
                original_row = df[df['filename'] == filename]
                title = original_row['Title'].iloc[0] if not original_row.empty and 'Title' in df.columns else filename
                
                all_rows.append({
                    'search_term': term,
                    'search_method': method,
                    'title': title,
                    'filename': filename,
                    'score_BoW': float(bow_row['Score']) if not bow_row.empty else None,
                    'rank_in_BoW': int(bow_row['rank_in_BoW']) if not bow_row.empty else None,
                    'score_TFIDF': float(tfidf_row['Score']) if not tfidf_row.empty else None,
                    'rank_in_TFIDF': int(tfidf_row['rank_in_TFIDF']) if not tfidf_row.empty else None,
                    'score_Semantic': float(semantic_row['Score']) if not semantic_row.empty else None,
                    'rank_in_Semantic': int(semantic_row['rank_in_Semantic']) if not semantic_row.empty else None,
                })
    # Remove duplicates (if an article is in top 10 for more than one method, keep all appearances)
    df_out = pd.DataFrame(all_rows)
    # Order by search_term, search_method, then rank in that method
    df_out = df_out.sort_values(by=['search_term', 'search_method', 
                                    'rank_in_BoW', 'rank_in_TFIDF', 'rank_in_Semantic'])
    return df_out

def save_to_csv(df, filename, **kwargs):
    """Save a DataFrame to CSV, overwriting if the file already exists."""
    try:
        print(f"Saving DataFrame to {filename}")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {list(df.columns)}")
        df.to_csv(filename, **kwargs)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving {filename}: {e}")

# Example usage in your __main__ block:
if __name__ == "__main__":
    # Load articles using enhanced provider system
    print("🚀 Starting search system...")
    df, provider_info = load_articles_with_provider(limit=500, prefer_database=True)
    
    if df is None or len(df) == 0:
        print("❌ No articles available. Exiting.")
        exit(1)
    
    print(f"📊 Using articles from: {provider_info.get('active_provider', 'unknown source')}")
    print(f"📚 Total articles loaded: {len(df)}")
    
    # Determine if we're using database for backward compatibility with existing functions
    use_database = provider_info.get('active_provider') not in ['FileBasedArticleProvider', 'Fallback']

    # Define the query
    query1 = "Impact of Advanced Cardiac Life Support courses on knowledge retention and competency in undergraduate medical education in the UK."
    query2 = "Impact of first-pass defibrillation on survival and neurological outcome in out-of-hospital ventricular fibrillation."
    query3 = "Effect of glucose administration or hyperglycemia management on outcomes after cardiac arrest."

    query1a = "Effect of ACLS training programs on medical students’ skills, knowledge retention, and performance in the UK."
    query2a = "Does successful defibrillation on the first shock improve survival rates and neurological recovery in OHCA patients with ventricular fibrillation?"
    query3a = "Influence of blood glucose control and hyperglycemia treatment on survival and neurological recovery in post–cardiac arrest patients."

    query = query3a

    # Run all searches
    print("Running lexical search...")
    lexical_results = run_lexical_search(df, query)

    print("Running TF-IDF search...")
    tf_idf_results = run_tf_idf_search(df, query)

    print("Running semantic search...")
    semantic_results = run_semantic_search(df, query)

    # Combine results
    print("Combining results...")
    combined_results = pd.concat([lexical_results, tf_idf_results, semantic_results], ignore_index=True)

    # Save combined results to a central CSV file
    output_file = "combined_search_results.csv"
    print(f"Saving combined results to {output_file}...")

    # Make a copy for CSV output, modify only the copy
    # Check if we have database articles (with Title column) or file-based articles
    if 'Title' in combined_results.columns:
        # Use Title from database
        combined_results_csv = combined_results[['filename', 'Score', 'Search Type', 'Title']].copy()
        combined_results_csv.rename(columns={'Title': 'First Line'}, inplace=True)
    else:
        # Fallback to first line of content for file-based articles
        combined_results_csv = combined_results[['filename', 'Score', 'Search Type', 'content']].copy()
        combined_results_csv.rename(columns={'content': 'First Line'}, inplace=True)
        combined_results_csv['First Line'] = combined_results_csv['First Line'].apply(lambda x: x.strip().splitlines()[0] if x.strip() else "[EMPTY TEXT]")
    print("DataFrame to be saved to CSV:")
    print(combined_results_csv.head(10))
    combined_results_csv.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Results successfully saved to {output_file}.")

    # Save ALL search results (not just top 10)
    print("\\nSaving all search results...")
    all_results = get_all_search_results(df, query)
    all_results_file = "all_search_results.csv"
    all_results.to_csv(all_results_file, index=False, encoding='utf-8')
    print(f"All search results saved to {all_results_file} ({len(all_results)} total rows)")

    # Save PMC URLs to a text file (use original combined_results)
    save_pmc_urls(combined_results, output_txt_file="pmc_urls.txt")

    # Visualize the results (use original combined_results)
    visualize_results(combined_results)
    visualize_results_heatmap(combined_results)

    search_terms = [
        query,
        # Add more search terms as needed
    ]

    print("Starting multi_term_search...")
    results_df = multi_term_search(search_terms, top_n=2, use_database=use_database)
    print("multi_term_search finished.")
    print("[main] DataFrame to be saved (multi_term_search_results):")
    print(results_df.head(20))  # Print after all modifications
    print(f"[main] DataFrame shape: {results_df.shape}")
    print(f"[main] DataFrame columns: {list(results_df.columns)}")
    save_to_csv(results_df, "multi_term_search_results.csv", index=False)

    print("Starting multi_term_top10_full_ranking...")
    results_df_full_ranking = multi_term_top10_full_ranking(search_terms, top_n=10, use_database=use_database)
    print("[main] DataFrame to be saved (multi_term_top10_full_ranking):")
    print(results_df_full_ranking.head(30))  # Print after all modifications
    print(f"[main] DataFrame shape: {results_df_full_ranking.shape}")
    print(f"[main] DataFrame columns: {list(results_df_full_ranking.columns)}")
    save_to_csv(results_df_full_ranking, "multi_term_top10_full_ranking.csv", index=False)

