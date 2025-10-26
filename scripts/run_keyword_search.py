import pandas as pd
import re
import argparse
# Use the article provider system directly instead of preprocess
try:
    from csv_writer import write_results_to_csv
except ImportError:
    from scripts.csv_writer import write_results_to_csv

def keyword_search(articles_df, keyword_groups, top_n=100):
    """
    Perform keyword search using nested array logic.
    
    Args:
        articles_df: DataFrame containing articles with 'content' column
        keyword_groups: List of lists where main array = AND, inner arrays = OR
                       e.g. [["training", "education"], ["medical student*", "resident*"]]
        top_n: Number of top results to return
    
    Returns:
        DataFrame with results and scores
    """
    results = []
    
    for idx, row in articles_df.iterrows():
        content = str(row['content']).lower()
        score = 0
        group_matches = 0
        
        # Each group must have at least one match (AND logic)
        for group in keyword_groups:
            group_matched = False
            group_score = 0
            
            # At least one term in the group must match (OR logic)
            for term in group:
                term_lower = term.lower()
                
                # Handle wildcard terms (ending with *)
                if term_lower.endswith('*'):
                    base_term = term_lower[:-1]
                    # Use regex for partial matching
                    pattern = r'\b' + re.escape(base_term) + r'\w*'
                    matches = len(re.findall(pattern, content))
                else:
                    # Exact word matching
                    pattern = r'\b' + re.escape(term_lower) + r'\b'
                    matches = len(re.findall(pattern, content))
                
                if matches > 0:
                    group_matched = True
                    # Score based on frequency and term importance
                    term_score = matches * (1 + len(term_lower) / 20)  # Longer terms get slight bonus
                    group_score = max(group_score, term_score)  # Best match in group
            
            if group_matched:
                group_matches += 1
                score += group_score
            else:
                # If any group fails to match, this article doesn't qualify
                score = 0
                break
        
        # Only include articles that match ALL groups
        if group_matches == len(keyword_groups) and score > 0:
            results.append({
                'index': idx,
                'score': score,
                'filename': row['filename'],
                'content': row['content'],
                'Id': row.get('Id', ''),
                'Title': row.get('Title', ''),
                'PmcId': row.get('PmcId', ''),
                'Keywords': row.get('Keywords', '')
            })
    
    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top N results
    top_results = results[:top_n]
    
    if not top_results:
        return pd.DataFrame(), []
    
    # Create DataFrame from results
    result_df = pd.DataFrame(top_results)
    scores = [r['score'] for r in top_results]
    
    return result_df, scores

def display_results(results_df, scores, keyword_groups):
    """Display search results in a readable format."""
    print(f"\nKeyword Search Results")
    print("=" * 50)
    print(f"Search Pattern: {format_search_pattern(keyword_groups)}")
    print(f"Found {len(results_df)} matching articles")
    print("=" * 50)
    
    for i, (idx, row) in enumerate(results_df.iterrows()):
        print(f"\n{i+1}. {row['filename']}")
        if row.get('Title'):
            print(f"   Title: {row['Title'][:100]}...")
        print(f"   PMC ID: {row.get('PmcId', 'N/A')}")
        print(f"   Score: {scores[i]:.4f}")
        print(f"   Content: {row['content'][:200]}...")
        print("-" * 50)

def format_search_pattern(keyword_groups):
    """Format the keyword groups for display."""
    formatted_groups = []
    for group in keyword_groups:
        if len(group) == 1:
            formatted_groups.append(f'"{group[0]}"')
        else:
            or_terms = ' OR '.join([f'"{term}"' for term in group])
            formatted_groups.append(f"({or_terms})")
    
    return " AND ".join(formatted_groups)

def main():
    parser = argparse.ArgumentParser(description='Run keyword search on articles database')
    parser.add_argument('--limit', type=int, default=500, help='Number of articles to load from database')
    parser.add_argument('--top', type=int, default=100, help='Number of top results to return')
    parser.add_argument('--output', type=str, default='keyword_search_results.csv', help='Output CSV file')
    parser.add_argument('--sample', action='store_true', help='Use sample search terms')
    
    args = parser.parse_args()
    
    # Load articles using the article provider system
    print(f"Loading {args.limit} articles...")
    try:
        from provider_config import get_article_provider
        provider = get_article_provider()
        articles_df = provider.get_articles(limit=args.limit)
        print(f"Loaded {len(articles_df)} articles from {provider.__class__.__name__}")
    except ImportError:
        print("❌ Article provider system not available. Please install and configure the provider system.")
        return
    except Exception as e:
        print(f"❌ Error loading articles: {e}")
        return
    
    if articles_df.empty:
        print("No articles loaded. Exiting.")
        return
    
    # Define search terms - you can modify this or make it configurable
    if args.sample:
        # Sample search as described in the request
        keyword_groups = [
            ["training", "education", "course", "certification"],
            ["medical student*", "junior doctor*", "resident*", "trainee*"],
            ["skill retention", "knowledge retention", "competence", "performance"]
        ]
    else:
        # Default cardiac-related search
        keyword_groups = [
            ["cardiac", "heart", "cardiovascular"],
            ["emergency", "resuscitation", "cpr", "acls"],
            ["training", "education", "course"]
        ]
    
    print(f"\nPerforming keyword search...")
    print(f"Search pattern: {format_search_pattern(keyword_groups)}")
    
    # Perform search
    results_df, scores = keyword_search(articles_df, keyword_groups, top_n=args.top)
    
    if results_df.empty:
        print("No articles matched the search criteria.")
        return
    
    # Display results
    display_results(results_df, scores, keyword_groups)
    
    # Write results to CSV
    print(f"\nWriting results to {args.output}...")
    write_results_to_csv(results_df, scores, args.output)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
