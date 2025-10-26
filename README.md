# 🔍 Biomedical Literature Search & Ranking Framework

A modular Python toolkit for **biomedical literature retrieval and ranking** using **Lexical (BoW)**, **TF-IDF**, and **Semantic** search methods.  
It allows side-by-side comparison of traditional and embedding-based search results on biomedical text datasets (e.g., PubMed Central).

---

## 🚀 Quick Start

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Set your OpenAI API key
```bash
export OPENAI_API_KEY="your_api_key_here"
```

### 3️⃣ Run all search methods
```bash
python run_all_searches.py
```

### 4️⃣ Visualize the results
- `search_results_comparison.png` — bar plot of search method scores  
- `search_results_heatmap.png` — normalized similarity heatmap  

---

## ⚙️ Features

| Module | Purpose |
|--------|----------|
| `lexical_search.py` | Bag-of-Words cosine similarity |
| `tf_idf_search.py` | TF-IDF vectorization and ranking |
| `semantic_search.py` | OpenAI embedding-based similarity |
| `run_keyword_search.py` | Boolean keyword-based filtering |
| `run_all_searches.py` | Orchestrates and compares all search types |
| `ranking_display.py` | Visualizes combined results |
| `csv_writer.py` | Exports ranked results to CSV |

---

## 📦 Output Files

| File | Description |
|------|--------------|
| `combined_search_results.csv` | Combined top results across methods |
| `all_search_results.csv` | Full ranked list of all documents |
| `search_results_comparison.png` | Comparison bar chart |
| `search_results_heatmap.png` | Heatmap of normalized scores |
| `pmc_urls.txt` | Linked PMC article URLs |

---

## 🧩 Requirements
- Python ≥ 3.9  
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `openai`, `tqdm`

Install them via:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openai tqdm
```

---

## 🏷️ License
MIT License © 2025
