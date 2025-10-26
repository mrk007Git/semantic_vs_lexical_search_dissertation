import csv

def write_results_to_csv(results, scores, output_file):
    """Write search results to a CSV file."""
    print(f"Writing results to {output_file}...")
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Check if Title column is available
        has_title = 'Title' in results.columns
        
        # Write the header
        if has_title:
            writer.writerow(["Filename", "Score", "Title", "First Line"])
        else:
            writer.writerow(["Filename", "Score", "First Line"])
        
        # Write each result
        for i, row in results.iterrows():
            first_line = row['content'].strip().splitlines()[0] if row['content'].strip() else "[EMPTY TEXT]"
            score_value = f"{scores[results.index.get_loc(i)]:.4f}"
            
            if has_title:
                title = row.get('Title', '')
                writer.writerow([row['filename'], score_value, title, first_line])
            else:
                writer.writerow([row['filename'], score_value, first_line])
    print(f"Results successfully written to {output_file}.")