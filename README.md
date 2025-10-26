# Semantic vs Lexical Search - Database Agnostic

A flexible, database-agnostic search system that compares semantic and lexical search approaches across different article datasets. The system now supports multiple data sources through a provider-based architecture.

## 🆕 New Features

- **Database Agnostic**: Support any data source by implementing the `ArticleProvider` interface
- **Provider System**: Clean separation between data access and search logic
- **Automatic Fallbacks**: Graceful degradation between data sources
- **Backward Compatibility**: Existing code continues to work unchanged
- **Custom Providers**: Easy integration of new data sources

## Architecture

### Provider-Based System

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Search Scripts  │────│ ArticleProvider  │────│ Data Sources    │
│ (Unchanged)     │    │ (Interface)      │    │ (Configurable)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ├── DatabaseProvider (SQL)
                              ├── FileProvider (Text files)
                              ├── CSVProvider (CSV files)
                              ├── APIProvider (REST APIs)
                              └── CustomProvider (Your impl.)
```

### Core Components

- **`ArticleProvider`**: Abstract interface for article retrieval
- **`DatabaseArticleProvider`**: SQL database implementation
- **`FileBasedArticleProvider`**: Text file implementation
- **`CompositeArticleProvider`**: Multi-provider with fallbacks
- **`provider_config.py`**: Configuration and factory functions

## Quick Start

### 1. Using the System (No Changes Required)

Your existing code continues to work:

```python
# Existing code - still works
from scripts.preprocess import load_articles_from_db

articles = load_articles_from_db(limit=100)
```

### 2. Using the New Provider System (Recommended)

```python
# New approach - more flexible
from provider_config import get_article_provider

provider = get_article_provider()
articles = provider.get_articles(limit=100)

# Even simpler
from scripts.preprocess import load_articles

articles = load_articles(limit=100)  # Automatic provider selection
```

### 3. Creating Custom Providers

```python
from article_provider import ArticleProvider
import pandas as pd

class MyCustomProvider(ArticleProvider):
    def __init__(self, connection_string):
        self.connection_string = connection_string
    
    def get_articles(self, limit=None):
        # Your implementation here
        return pd.DataFrame([...])
    
    def get_article_by_id(self, article_id):
        # Your implementation here
        return Article(...)
    
    def get_total_count(self):
        return 1000  # Your count logic
    
    def is_available(self):
        return True  # Your availability check
```

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/mrk007Git/semantic-vs-lexical-search.git
cd semantic-vs-lexical-search

# Install dependencies
pip install -r requirements.txt
```

### Optional Dependencies (for specific providers)

```bash
# For database providers
pip install pyodbc sqlalchemy python-dotenv

# For additional search features
pip install openai scikit-learn tqdm
```

## Configuration

### Environment Variables

Create a `.env` file:

```env
# Database Configuration (optional)
DB_SERVER=your-server.database.windows.net
DB_DATABASE=your-database-name
DB_USERNAME=your-username
DB_PASSWORD=your-password
DB_DRIVER=ODBC Driver 17 for SQL Server

# Provider Selection (optional)
ARTICLE_PROVIDER=default  # or 'database', 'files', 'custom'

# OpenAI API Key (for semantic search)
OPENAI_API_KEY=your-openai-api-key
```

### Provider Configuration

Customize providers in `provider_config.py`:

```python
def get_article_provider():
    # Option 1: Use default (database + file fallback)
    return create_default_provider()
    
    # Option 2: Use specific provider
    return MyCustomProvider("connection-string")
    
    # Option 3: Custom fallback chain
    return CompositeArticleProvider([
        MyCustomProvider("primary"),
        DatabaseArticleProvider(get_db_connection),
        FileBasedArticleProvider("backup/folder")
    ])
```

## Usage Examples

### Search Scripts

All search scripts work with any provider:

```bash
# Semantic search
python scripts/semantic_search.py

# Lexical search  
python scripts/lexical_search.py

# TF-IDF search
python scripts/tf_idf_search.py

# Keyword search
python scripts/run_keyword_search.py --limit 100

# Run all searches
python scripts/run_all_searches.py
```

### Provider Testing

```bash
# Test the provider system
python test_provider_system.py

# Test database connection (legacy)
python scripts/test_db_connection.py

# Test custom providers
python example_providers.py
```

### Data Loading

```python
# Load with automatic provider selection
from scripts.preprocess import load_articles

# Prefer database, fallback to files
articles = load_articles(limit=1000, prefer_database=True)

# Prefer files over database
articles = load_articles(limit=1000, prefer_database=False)

# Use specific provider
from provider_config import get_article_provider

provider = get_article_provider()
articles = provider.get_articles(limit=1000)
```

## Supported Data Sources

### Built-in Providers

1. **Database Provider**: SQL databases (SQL Server, PostgreSQL, MySQL)
2. **File Provider**: Text files in directories
3. **CSV Provider**: CSV files with article data (example)
4. **API Provider**: REST APIs (example/template)

### Creating Custom Providers

See the documentation:

- **[Implementation Guide](PROVIDER_IMPLEMENTATION_GUIDE.md)**: How to create providers
- **[Migration Guide](MIGRATION_GUIDE.md)**: Migrating from legacy system
- **[Example Providers](example_providers.py)**: Working examples

### Required Data Schema

Your data source must provide articles with:

- **Id**: Unique identifier
- **Title**: Article title
- **content**: Full text content
- **filename**: File identifier (for compatibility)

Additional fields are optional and preserved.

## Search Methods

### 1. Semantic Search
Uses OpenAI embeddings for meaning-based search:
- Understands context and synonyms
- Better for conceptual queries
- Requires OpenAI API key

### 2. Lexical Search
Traditional keyword-based search:
- Exact word matching
- Fast and lightweight
- No external dependencies

### 3. TF-IDF Search
Statistical text analysis:
- Term frequency analysis
- Good balance of speed and accuracy
- Works offline

## File Structure

```
├── article_provider.py              # Core provider interfaces
├── provider_config.py               # Provider configuration
├── example_providers.py             # Example implementations
├── test_provider_system.py          # System tests
├── PROVIDER_IMPLEMENTATION_GUIDE.md # Implementation docs
├── MIGRATION_GUIDE.md               # Migration help
├── DATABASE_INTEGRATION.md          # Legacy database docs
├── scripts/
│   ├── preprocess.py                # Data loading (updated)
│   ├── semantic_search.py           # Semantic search
│   ├── lexical_search.py            # Lexical search
│   ├── tf_idf_search.py             # TF-IDF search
│   ├── run_keyword_search.py        # Keyword search
│   ├── run_all_searches.py          # Combined searches
│   └── test_db_connection.py        # Database testing
└── models/                          # Search models and data
    └── embeddings/                  # Cached embeddings
```

## Migration from Legacy System

The new system is **100% backward compatible**:

### Phase 1: No Changes Required
- Keep using existing functions
- System automatically detects and uses providers
- No code changes needed

### Phase 2: Gradual Adoption
- Start using `load_articles()` function
- Switch to provider-based loading
- Test new functionality alongside existing

### Phase 3: Full Migration
- Implement custom providers for your data sources
- Remove legacy database-specific code
- Take advantage of full provider system

See **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** for detailed steps.

## Testing

### System Tests

```bash
# Test entire provider system
python test_provider_system.py

# Test specific components
python -c "
from provider_config import get_article_provider
provider = get_article_provider()
print('Available:', provider.is_available())
print('Count:', provider.get_total_count())
"
```

### Database Tests

```bash
# Test database connectivity
python scripts/test_db_connection.py
```

### Custom Provider Tests

```bash
# Test example providers
python example_providers.py
```

## Troubleshooting

### Common Issues

1. **Provider Not Available**
   ```python
   provider = get_article_provider()
   print(provider.get_provider_info())
   ```

2. **Database Connection Issues**
   ```bash
   python scripts/test_db_connection.py
   ```

3. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **File Path Issues**
   - Check `CHUNKS_FOLDER` path in configuration
   - Ensure article files exist and are readable

### Error Handling

The system provides detailed error messages and automatic fallbacks:

- Database unavailable → Falls back to files
- Files unavailable → Shows clear error message
- Provider errors → Detailed logging and troubleshooting tips

## Contributing

### Adding New Providers

1. Inherit from `ArticleProvider`
2. Implement required methods
3. Add to `provider_config.py`
4. Include tests and documentation

### Code Structure

- Keep providers in separate files
- Follow existing naming conventions
- Include comprehensive error handling
- Add tests for new functionality

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Documentation

- **[Provider Implementation Guide](PROVIDER_IMPLEMENTATION_GUIDE.md)**: Create custom providers
- **[Migration Guide](MIGRATION_GUIDE.md)**: Migrate from legacy system
- **[Database Integration](DATABASE_INTEGRATION.md)**: Legacy database documentation
- **[Example Providers](example_providers.py)**: Working code examples