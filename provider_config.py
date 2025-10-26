"""
Provider Configuration Module

This module provides configuration and factory functions for the article provider system.
It handles database connections, provider creation, and backward compatibility.
"""

import os
from typing import Optional
from article_provider import (
    ArticleProvider, 
    DatabaseArticleProvider, 
    FileBasedArticleProvider, 
    CompositeArticleProvider
)

# Default folder path for articles - users should override this
CHUNKS_FOLDER = "./articles"  # Generic default, users must configure their own paths


def create_database_connection():
    """
    Create a database connection using environment variables or config.
    
    Returns:
        Database connection object or None if not available
    """
    try:
        import pyodbc  # type: ignore
        
        # Try to get connection details from environment variables
        server = os.getenv('DB_SERVER', 'localhost')
        database = os.getenv('DB_DATABASE', 'cardiac_articles')
        username = os.getenv('DB_USERNAME')
        password = os.getenv('DB_PASSWORD')
        
        if username and password:
            connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
        else:
            # Try Windows Authentication
            connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'
        
        connection = pyodbc.connect(connection_string, timeout=5)
        return connection
        
    except ImportError:
        print("⚠️  pyodbc not available. Database provider disabled.")
        return None
    except Exception as e:
        print(f"⚠️  Database connection failed: {e}")
        return None


def create_default_provider() -> ArticleProvider:
    """
    Create the default article provider with fallback logic.
    
    Priority:
    1. Database provider (if available)
    2. File-based provider (if files exist)
    3. Composite provider (fallback chain)
    
    Returns:
        ArticleProvider instance
    """
    providers = []
    
    # Try database provider first
    db_connection = create_database_connection()
    if db_connection:
        try:
            db_provider = DatabaseArticleProvider(db_connection)
            if db_provider.is_available():
                providers.append(db_provider)
                print("✅ Database provider available")
        except Exception as e:
            print(f"⚠️  Database provider failed: {e}")
    
    # Add file-based provider only if CHUNKS_FOLDER exists
    try:
        if os.path.exists(CHUNKS_FOLDER):
            file_provider = FileBasedArticleProvider(CHUNKS_FOLDER)
            if file_provider.is_available():
                providers.append(file_provider)
                print("✅ File-based provider available")
        else:
            print(f"⚠️  Default folder '{CHUNKS_FOLDER}' not found. Users must configure their own article sources.")
    except Exception as e:
        print(f"⚠️  File-based provider failed: {e}")
    
    if not providers:
        print("⚠️  No providers available.")
        print("   Users must implement their own ArticleProvider or configure data sources.")
        # Create a minimal file provider that will return empty results
        return FileBasedArticleProvider(".")
    
    if len(providers) == 1:
        return providers[0]
    else:
        return CompositeArticleProvider(providers)


# Global provider instance
_provider_instance: Optional[ArticleProvider] = None


def get_article_provider() -> ArticleProvider:
    """
    Get the global article provider instance.
    
    Creates the provider on first call and caches it.
    
    Returns:
        ArticleProvider instance (never None)
    """
    global _provider_instance
    
    if _provider_instance is None:
        _provider_instance = create_default_provider()
    
    return _provider_instance


def reset_provider():
    """Reset the global provider instance (useful for testing)."""
    global _provider_instance
    _provider_instance = None


# Backward compatibility functions
def load_articles_from_database(limit: Optional[int] = None):
    """
    Backward compatibility function for database loading.
    
    Args:
        limit: Maximum number of articles to load
        
    Returns:
        DataFrame with articles
    """
    provider = get_article_provider()
    return provider.get_articles(limit)


def load_articles_from_db(limit: Optional[int] = None):
    """Alias for load_articles_from_database for backward compatibility."""
    return load_articles_from_database(limit)


def load_txt_articles(folder_path: Optional[str] = None):
    """
    Backward compatibility function for loading text articles.
    
    Args:
        folder_path: Path to folder containing text files
        
    Returns:
        DataFrame with articles
    """
    if folder_path is None:
        folder_path = CHUNKS_FOLDER
    
    file_provider = FileBasedArticleProvider(folder_path)
    return file_provider.get_articles()


def get_database_connection():
    """
    Backward compatibility function for database connection.
    
    Returns:
        Database connection or None
    """
    return create_database_connection()


def get_db_connection():
    """Alias for get_database_connection for backward compatibility."""
    return get_database_connection()