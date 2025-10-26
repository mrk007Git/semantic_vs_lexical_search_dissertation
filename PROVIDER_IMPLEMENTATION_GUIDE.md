# Article Provider Implementation Guide

This guide shows how to implement custom article providers for your specific data sources.

## Overview

The system now uses a provider-based architecture that allows you to implement custom article sources. You can create providers for:

- Different databases (PostgreSQL, MySQL, MongoDB, etc.)
- Cloud storage (AWS S3, Azure Blob, Google Cloud Storage)
- APIs (REST, GraphQL, etc.)
- Other file formats (JSON, XML, CSV, etc.)
- Custom data sources

## Basic Implementation

### 1. Inherit from ArticleProvider

```python
from article_provider import ArticleProvider, Article
import pandas as pd
from typing import Optional

class MyCustomProvider(ArticleProvider):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        # Initialize your data source connection here
        
    def get_articles(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load articles from your data source."""
        # Your implementation here
        # Return a DataFrame with these required columns:
        # - Id: Unique article identifier
        # - Title: Article title  
        # - content: Article content/text
        # - filename: Filename or identifier
        
        articles = []
        # Your data loading logic here
        
        return pd.DataFrame(articles)
    
    def get_article_by_id(self, article_id: str) -> Optional[Article]:
        """Get a specific article by ID."""
        # Your implementation here
        # Return Article instance or None if not found
        pass
        
    def get_total_count(self) -> int:
        """Get total number of articles."""
        # Your implementation here
        return 0
        
    def is_available(self) -> bool:
        """Check if data source is accessible."""
        # Your implementation here
        # Test connection and return True/False
        return False
```

### 2. Register Your Provider

In `provider_config.py`, modify the `get_article_provider()` function:

```python
def get_article_provider() -> ArticleProvider:
    """Get the configured article provider instance."""
    
    # Option 1: Use only your custom provider
    return MyCustomProvider("your-connection-string")
    
    # Option 2: Add your provider to the fallback chain
    my_provider = MyCustomProvider("your-connection-string")
    database_provider = create_database_provider()
    file_provider = create_file_provider()
    
    return CompositeArticleProvider([my_provider, database_provider, file_provider])
```

## Example Implementations

### PostgreSQL Provider

```python
import psycopg2
import pandas as pd
from article_provider import ArticleProvider, Article

class PostgreSQLProvider(ArticleProvider):
    def __init__(self, host: str, database: str, user: str, password: str, port: int = 5432):
        self.connection_params = {
            'host': host,
            'database': database,
            'user': user,
            'password': password,
            'port': port
        }
    
    def get_articles(self, limit: Optional[int] = None) -> pd.DataFrame:
        import psycopg2
        
        conn = psycopg2.connect(**self.connection_params)
        
        query = """
        SELECT id, title, content, keywords, created_at
        FROM articles 
        WHERE content IS NOT NULL
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql(query, conn)
        
        # Create required columns
        df['filename'] = df['id'].astype(str) + '.txt'
        
        conn.close()
        return df
    
    def is_available(self) -> bool:
        try:
            conn = psycopg2.connect(**self.connection_params)
            conn.close()
            return True
        except:
            return False
    
    # Implement other required methods...
```

### JSON File Provider

```python
import json
import os
from article_provider import ArticleProvider, Article

class JSONProvider(ArticleProvider):
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
    
    def get_articles(self, limit: Optional[int] = None) -> pd.DataFrame:
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        articles = []
        for item in data:
            articles.append({
                'Id': item['id'],
                'Title': item['title'],
                'content': item['content'],
                'filename': f"{item['id']}.txt",
                # Add other fields as needed
            })
            
            if limit and len(articles) >= limit:
                break
        
        return pd.DataFrame(articles)
    
    def is_available(self) -> bool:
        return os.path.exists(self.json_file_path)
    
    # Implement other required methods...
```

### REST API Provider

```python
import requests
from article_provider import ArticleProvider, Article

class APIProvider(ArticleProvider):
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {api_key}'}
    
    def get_articles(self, limit: Optional[int] = None) -> pd.DataFrame:
        url = f"{self.base_url}/articles"
        params = {'limit': limit} if limit else {}
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        articles = []
        for item in data['articles']:
            articles.append({
                'Id': item['id'],
                'Title': item['title'],
                'content': item['content'],
                'filename': f"{item['id']}.txt",
            })
        
        return pd.DataFrame(articles)
    
    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", 
                                   headers=self.headers, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    # Implement other required methods...
```

## Required DataFrame Schema

Your `get_articles()` method must return a pandas DataFrame with these columns:

- **Id** (required): Unique article identifier
- **Title** (required): Article title
- **content** (required): Full article text/content
- **filename** (required): Filename or identifier for compatibility

Optional columns (add as needed):
- Keywords: Article keywords
- PmcId: PubMed Central ID
- DateCreated: Creation date
- Author: Article author
- Any other metadata fields

## Configuration

### Environment Variables

Create a `.env` file with your provider-specific configuration:

```env
# Custom Provider Settings
CUSTOM_DB_HOST=localhost
CUSTOM_DB_NAME=my_articles
CUSTOM_DB_USER=username
CUSTOM_DB_PASSWORD=password
CUSTOM_API_KEY=your-api-key
CUSTOM_API_URL=https://api.example.com
```

### Provider Selection

You can control which provider to use by modifying `provider_config.py`:

```python
def get_article_provider() -> ArticleProvider:
    provider_type = os.getenv('ARTICLE_PROVIDER', 'default')
    
    if provider_type == 'postgresql':
        return PostgreSQLProvider(
            host=os.getenv('CUSTOM_DB_HOST'),
            database=os.getenv('CUSTOM_DB_NAME'),
            user=os.getenv('CUSTOM_DB_USER'),
            password=os.getenv('CUSTOM_DB_PASSWORD')
        )
    elif provider_type == 'api':
        return APIProvider(
            base_url=os.getenv('CUSTOM_API_URL'),
            api_key=os.getenv('CUSTOM_API_KEY')
        )
    else:
        return create_default_provider()  # Fallback to default
```

## Testing Your Provider

Create a test script to verify your provider works:

```python
from your_provider import MyCustomProvider

def test_provider():
    provider = MyCustomProvider("your-connection-string")
    
    # Test availability
    if not provider.is_available():
        print("Provider not available")
        return
    
    # Test article loading
    df = provider.get_articles(limit=10)
    print(f"Loaded {len(df)} articles")
    print("Columns:", df.columns.tolist())
    print("Sample data:")
    print(df.head())
    
    # Test specific article retrieval
    if len(df) > 0:
        article_id = df.iloc[0]['Id']
        article = provider.get_article_by_id(str(article_id))
        if article:
            print(f"Retrieved article: {article.title}")
        else:
            print("Could not retrieve specific article")

if __name__ == "__main__":
    test_provider()
```

## Migration from Existing Code

The new system is backward compatible. Existing code will continue to work, but you can gradually migrate:

1. **Current**: `df = load_articles_from_db(limit=100)`
2. **New**: `provider = get_article_provider(); df = provider.get_articles(limit=100)`

The benefit of the new approach is that it automatically handles fallbacks and allows for easy swapping of data sources.