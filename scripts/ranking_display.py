import pandas as pd
try:
    from run_all_searches import run_lexical_search, run_tf_idf_search, run_semantic_search
except ImportError:
    from scripts.run_all_searches import run_lexical_search, run_tf_idf_search, run_semantic_search
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def display_search_results(pmc_number, query, top_n=10):
    """Display search results for BoW, TF-IDF, and Semantic Search for a given PMC number."""
    # Load articles using article provider system
    print("Loading articles...")
    try:
        try:
            from run_all_searches import load_articles_with_provider
        except ImportError:
            from scripts.run_all_searches import load_articles_with_provider
        df, provider_info = load_articles_with_provider()
        print(f"Loaded {len(df)} articles using {provider_info.get('active_provider', 'unknown provider')}")
    except Exception as e:
        print(f"Error loading articles: {e}")
        print("Please configure your article provider system.")
        return

    # Run BoW search
    print("Running BoW search...")
    bow_results = run_lexical_search(df, query, top_n=top_n)

    # Run TF-IDF search
    print("Running TF-IDF search...")
    tfidf_results = run_tf_idf_search(df, query, top_n=top_n)

    # Run Semantic search
    print("Running Semantic search...")
    semantic_results = run_semantic_search(df, query, top_n=top_n)

    # Debug: Print individual search results
    print("\nBoW Results:")
    print(bow_results[['filename', 'Score']])
    print("\nTF-IDF Results:")
    print(tfidf_results[['filename', 'Score']])
    print("\nSemantic Search Results:")
    print(semantic_results[['filename', 'Score']])

    # Combine results
    combined_results = pd.concat([bow_results, tfidf_results, semantic_results], ignore_index=True)

    # Debug: Print combined results before filtering
    print("\nCombined Results:")
    print(combined_results[['filename', 'Score', 'Search Type']])

    # Filter results for the given PMC number (case-insensitive match)
    filtered_results = combined_results[combined_results['filename'].str.contains(pmc_number, case=False, na=False)]

    if filtered_results.empty:
        print(f"No results found for PMC number: {pmc_number}")
    else:
        print(f"\nSearch Results for PMC number: {pmc_number}")
        print(filtered_results[['filename', 'Score', 'Search Type']])

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

    # Sort rows by average score across all methods
    heatmap_data['Average Score'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values(by='Average Score', ascending=False)
    heatmap_data = heatmap_data.drop(columns=['Average Score'])  # Remove the temporary column

    # Sort columns by a logical order (Semantic, TF-IDF, BoW)
    heatmap_data = heatmap_data[['Semantic', 'TF-IDF', 'BoW']]

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
    plt.savefig("search_results_heatmap_sorted.png")
    print("Heatmap saved as 'search_results_heatmap_sorted.png'.")

    # Show the heatmap
    plt.show()

if __name__ == "__main__":
    # Input PMC number
    # Define the query
    query = "advanced cardiac life support training in UK medical schools"
    print(query)
    pmc_number = input("Enter the PMC number (e.g., PMC1342422): ").strip()
    top_n = int(input("Enter the number of top results to display: ").strip())
    display_search_results(pmc_number, query, top_n=top_n)