"""
Article Provider Interface and Implementations

This module defines an abstract interface for retrieving article content,
along with concrete implementations for different data sources.
Users can implement the ArticleProvider interface to create custom
article content providers for their specific data sources.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import pandas as pd


class Article:
    """
    Represents an article with its metadata and content.
    
    This standardized article format ensures consistency across
    different data sources.
    """
    
    def __init__(self, 
                 article_id: str,
                 title: str,
                 content: str,
                 filename: str,
                 **kwargs):
        """
        Initialize an Article instance.
        
        Args:
            article_id: Unique identifier for the article
            title: Article title
            content: Full article content/text
            filename: Filename or identifier for compatibility
            **kwargs: Additional metadata (e.g., keywords, pmc_id, etc.)
        """
        self.article_id = article_id
        self.title = title
        self.content = content
        self.filename = filename
        self.metadata = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert article to dictionary format."""
        return {
            'Id': self.article_id,
            'Title': self.title,
            'content': self.content,
            'filename': self.filename,
            **self.metadata
        }


class ArticleProvider(ABC):
    """
    Abstract base class for article content providers.
    
    This interface defines the contract that must be implemented
    by any article content provider. Users can inherit from this
    class to create custom providers for their specific data sources.
    """
    
    @abstractmethod
    def get_articles(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve articles from the data source.
        
        Args:
            limit: Optional limit on the number of articles to retrieve
            
        Returns:
            DataFrame with standardized article columns:
            - Id: Unique article identifier
            - Title: Article title
            - content: Article content/text
            - filename: Filename or identifier
            - Additional columns may vary by implementation
            
        Raises:
            Exception: If articles cannot be retrieved
        """
        pass
    
    @abstractmethod
    def get_article_by_id(self, article_id: str) -> Optional[Article]:
        """
        Retrieve a specific article by its ID.
        
        Args:
            article_id: The unique identifier of the article
            
        Returns:
            Article instance if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_total_count(self) -> int:
        """
        Get the total number of articles available.
        
        Returns:
            Total count of articles in the data source
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the data source is available and accessible.
        
        Returns:
            True if the data source is accessible, False otherwise
        """
        pass
    
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about this provider.
        
        Returns:
            Dictionary with provider metadata
        """
        return {
            'provider_type': self.__class__.__name__,
            'is_available': self.is_available(),
            'total_articles': self.get_total_count() if self.is_available() else 0
        }


class FileBasedArticleProvider(ArticleProvider):
    """
    File-based article provider that reads articles from text files.
    
    This provider reads articles from a specified directory where
    each file contains the content of one article.
    """
    
    def __init__(self, folder_path: str):
        """
        Initialize the file-based provider.
        
        Args:
            folder_path: Path to the directory containing article files
        """
        self.folder_path = folder_path
        self._cached_articles = None
    
    def get_articles(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load articles from text files in the specified folder."""
        import os
        from tqdm import tqdm
        
        if not os.path.exists(self.folder_path):
            raise Exception(f"Folder path does not exist: {self.folder_path}")
        
        articles = []
        filenames = [f for f in os.listdir(self.folder_path) if f.endswith(".txt")]
        
        if limit:
            filenames = filenames[:limit]
        
        for filename in tqdm(filenames, desc="Loading articles from files"):
            file_path = os.path.join(self.folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                article = Article(
                    article_id=filename.replace('.txt', ''),
                    title=filename.replace('.txt', '').replace('_', ' ').title(),
                    content=content,
                    filename=filename
                )
                articles.append(article.to_dict())
            except Exception as e:
                print(f"Warning: Could not read file {filename}: {e}")
        
        self._cached_articles = pd.DataFrame(articles)
        return self._cached_articles
    
    def get_article_by_id(self, article_id: str) -> Optional[Article]:
        """Get a specific article by its ID (filename without extension)."""
        import os
        
        filename = f"{article_id}.txt"
        file_path = os.path.join(self.folder_path, filename)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            return Article(
                article_id=article_id,
                title=article_id.replace('_', ' ').title(),
                content=content,
                filename=filename
            )
        except Exception:
            return None
    
    def get_total_count(self) -> int:
        """Get the total number of text files in the folder."""
        import os
        
        if not os.path.exists(self.folder_path):
            return 0
        
        return len([f for f in os.listdir(self.folder_path) if f.endswith(".txt")])
    
    def is_available(self) -> bool:
        """Check if the folder exists and is accessible."""
        import os
        return os.path.exists(self.folder_path) and os.path.isdir(self.folder_path)


class DatabaseArticleProvider(ArticleProvider):
    """
    Database-based article provider that reads articles from a SQL database.
    
    This provider connects to a database and retrieves articles from
    a specified table with the expected schema.
    """
    
    def __init__(self, connection_factory):
        """
        Initialize the database provider.
        
        Args:
            connection_factory: A callable that returns a database engine/connection
        """
        self.connection_factory = connection_factory
        self._cached_connection = None
    
    def _get_connection(self):
        """Get database connection, using cache if available."""
        if self._cached_connection is None:
            self._cached_connection = self.connection_factory()
        return self._cached_connection
    
    def get_articles(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load articles from the database."""
        engine = self._get_connection()
        if engine is None:
            raise Exception("Failed to create database connection")
        
        limit_clause = f"TOP ({limit})" if limit else ""
        
        query = f"""
        SELECT {limit_clause}
            [Id],
            [Title],
            [Content],
            [Keywords],
            [PmcId],
            [Processed],
            [DateProcessed],
            [IsFractured],
            [FracturedReason],
            [Confidence],
            [LanguageCode],
            [IsAnalyzed],
            [ArticleSummary]
        FROM [dhs].[DissertationArticles]
        WHERE [Content] IS NOT NULL AND LEN([Content]) > 0
        """
        
        print(f"Loading articles from database (limit: {limit})...")
        df = pd.read_sql(query, engine)
        
        # Create filename-like identifier for compatibility
        df['filename'] = df.apply(
            lambda row: f"{row['PmcId']}.txt" if pd.notna(row['PmcId']) and row['PmcId'] != '' 
            else f"article_{row['Id']}.txt", 
            axis=1
        )
        
        # Rename Content to content for compatibility
        df = df.rename(columns={'Content': 'content'})
        
        print(f"Loaded {len(df)} articles from database.")
        return df
    
    def get_article_by_id(self, article_id: str) -> Optional[Article]:
        """Get a specific article by its database ID."""
        engine = self._get_connection()
        if engine is None:
            return None
        
        query = """
        SELECT [Id], [Title], [Content], [Keywords], [PmcId]
        FROM [dhs].[DissertationArticles]
        WHERE [Id] = ? AND [Content] IS NOT NULL
        """
        
        try:
            df = pd.read_sql(query, engine, params=[article_id])
            if df.empty:
                return None
            
            row = df.iloc[0]
            filename = f"{row['PmcId']}.txt" if pd.notna(row['PmcId']) and row['PmcId'] != '' else f"article_{row['Id']}.txt"
            
            return Article(
                article_id=str(row['Id']),
                title=row['Title'],
                content=row['Content'],
                filename=filename,
                keywords=row.get('Keywords', ''),
                pmc_id=row.get('PmcId', '')
            )
        except Exception:
            return None
    
    def get_total_count(self) -> int:
        """Get the total number of articles in the database."""
        engine = self._get_connection()
        if engine is None:
            return 0
        
        query = """
        SELECT COUNT(*) as count 
        FROM [dhs].[DissertationArticles] 
        WHERE [Content] IS NOT NULL AND LEN([Content]) > 0
        """
        
        try:
            df = pd.read_sql(query, engine)
            return df.iloc[0]['count']
        except Exception:
            return 0
    
    def is_available(self) -> bool:
        """Check if database connection can be established."""
        try:
            engine = self._get_connection()
            if engine is None:
                return False
            
            # Test connection with a simple query
            test_query = "SELECT 1"
            pd.read_sql(test_query, engine)
            return True
        except Exception:
            return False


class CompositeArticleProvider(ArticleProvider):
    """
    Composite provider that can fallback between multiple providers.
    
    This provider tries providers in order until one is available,
    providing a robust fallback mechanism.
    """
    
    def __init__(self, providers: List[ArticleProvider]):
        """
        Initialize with a list of providers in order of preference.
        
        Args:
            providers: List of ArticleProvider instances in order of preference
        """
        self.providers = providers
        self._active_provider = None
    
    def _get_active_provider(self) -> Optional[ArticleProvider]:
        """Get the first available provider."""
        if self._active_provider and self._active_provider.is_available():
            return self._active_provider
        
        for provider in self.providers:
            if provider.is_available():
                self._active_provider = provider
                return provider
        
        return None
    
    def get_articles(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Get articles from the first available provider."""
        provider = self._get_active_provider()
        if provider is None:
            raise Exception("No article providers are available")
        
        print(f"Using provider: {provider.__class__.__name__}")
        return provider.get_articles(limit)
    
    def get_article_by_id(self, article_id: str) -> Optional[Article]:
        """Get article by ID from the active provider."""
        provider = self._get_active_provider()
        if provider is None:
            return None
        
        return provider.get_article_by_id(article_id)
    
    def get_total_count(self) -> int:
        """Get total count from the active provider."""
        provider = self._get_active_provider()
        if provider is None:
            return 0
        
        return provider.get_total_count()
    
    def is_available(self) -> bool:
        """Check if any provider is available."""
        return any(provider.is_available() for provider in self.providers)
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about all providers."""
        active_provider = self._get_active_provider()
        
        return {
            'provider_type': 'CompositeArticleProvider',
            'active_provider': active_provider.__class__.__name__ if active_provider else None,
            'available_providers': [
                provider.__class__.__name__ for provider in self.providers 
                if provider.is_available()
            ],
            'all_providers': [
                {
                    'type': provider.__class__.__name__,
                    'available': provider.is_available(),
                    'count': provider.get_total_count() if provider.is_available() else 0
                }
                for provider in self.providers
            ],
            'is_available': self.is_available(),
            'total_articles': self.get_total_count()
        }