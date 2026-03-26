"""Synonym/alias engine — maps construction terms across languages and conventions."""
from src.knowledge_base.synonyms.loader import SynonymLoader
from src.knowledge_base.synonyms.schema import SynonymEntry

__all__ = ["SynonymLoader", "SynonymEntry"]
