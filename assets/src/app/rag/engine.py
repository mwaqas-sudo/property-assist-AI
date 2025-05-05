"""
RAG Engine for PropertyAssistAI
------------------------------
This module implements a high-performance Retrieval-Augmented Generation system
for real estate property information and broker knowledge.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from pydantic import BaseModel

from app.core.config import settings
from app.utils.metrics import MetricsCollector


logger = logging.getLogger(__name__)


class Document(BaseModel):
    """Represents a document in the knowledge base."""
    content: str
    metadata: Dict[str, Any]


class QueryResult(BaseModel):
    """Result of a RAG query."""
    context: str
    sources: List[Dict[str, Any]]
    relevant_doc_ids: List[str]
    query_time_ms: float


class RAGEngine:
    """
    High-performance RAG engine optimized for real estate domain knowledge.
    
    This class handles vector similarity search, context generation, and
    document retrieval with sub-millisecond performance targets.
    """
    
    def __init__(
        self,
        connection_string: str,
        collection_name: str = "property_documents",
        embedding_model: str = "paraphrase-multilingual-mpnet-base-v2",
        cache_manager = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        """
        Initialize the RAG engine.
        
        Args:
            connection_string: PostgreSQL connection string
            collection_name: Name of the vector collection
            embedding_model: HuggingFace model for embeddings
            cache_manager: Optional cache for results
            metrics_collector: Optional metrics collector
        """
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.cache = cache_manager
        self.metrics = metrics_collector
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            cache_folder=settings.MODEL_CACHE_DIR,
        )
        
        # Initialize vector store
        self.vectorstore = PGVector(
            connection_string=connection_string,
            embedding_function=self.embeddings,
            collection_name=collection_name,
            pre_delete_collection=False,
        )
        
        # Performance optimization: maintain an in-memory cache for hot queries
        self.query_cache = {}
        self.cache_hits = 0
        self.total_queries = 0
        
        logger.info(f"RAG Engine initialized with {embedding_model} embeddings")
    
    async def query(
        self, 
        query_text: str, 
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> QueryResult:
        """
        Query the knowledge base with optimized retrieval.
        
        Args:
            query_text: The query text
            filters: Optional metadata filters
            top_k: Number of documents to retrieve
            
        Returns:
            QueryResult with context and sources
        """
        start_time = time.time()
        self.total_queries += 1
        
        # Track metrics if available
        if self.metrics:
            self.metrics.increment("rag_queries")
        
        # Check in-memory and Redis cache first for exact query matches
        cache_key = f"rag:query:{query_text}:{str(filters)}:{top_k}"
        
        # Try memory cache first (fastest)
        if cache_key in self.query_cache:
            self.cache_hits += 1
            if self.metrics:
                self.metrics.increment("rag_cache_hits")
            return self.query_cache[cache_key]
        
        # Try Redis cache next
        if self.cache:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                self.cache_hits += 1
                if self.metrics:
                    self.metrics.increment("rag_cache_hits")
                return cached_result
        
        # Process the query to optimize retrieval
        processed_query = await self._preprocess_query(query_text)
        
        # Generate query embedding
        query_embedding = await asyncio.to_thread(
            self.embeddings.embed_query, 
            processed_query
        )
        
        # Prepare metadata filters
        metadata_filters = {}
        if filters:
            metadata_filters = filters
        
        # Perform similarity search with filtering
        try:
            # Use asyncio to make this non-blocking
            docs_with_scores = await asyncio.to_thread(
                self.vectorstore.similarity_search_with_score,
                processed_query,
                k=top_k,
                filter=metadata_filters
            )
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            # Return empty result on error
            return QueryResult(
                context="",
                sources=[],
                relevant_doc_ids=[],
                query_time_ms=(time.time() - start_time) * 1000
            )
        
        # Extract documents and scores
        documents = [doc for doc, _ in docs_with_scores]
        scores = [score for _, score in docs_with_scores]
        
        # Perform re-ranking if needed
        if len(documents) > 1:
            documents, scores = await self._rerank_documents(
                documents, 
                scores, 
                processed_query
            )
        
        # Generate context from documents
        context = await self._generate_context(documents, processed_query)
        
        # Extract sources metadata and doc IDs
        sources = []
        doc_ids = []
        for doc in documents:
            if doc.metadata and "source" in doc.metadata:
                sources.append(doc.metadata)
            if doc.metadata and "id" in doc.metadata:
                doc_ids.append(doc.metadata["id"])
        
        # Create result
        result = QueryResult(
            context=context,
            sources=sources,
            relevant_doc_ids=doc_ids,
            query_time_ms=(time.time() - start_time) * 1000
        )
        
        # Cache the result
        self.query_cache[cache_key] = result
        if len(self.query_cache) > settings.RAG_MEMORY_CACHE_SIZE:
            # Simple LRU by removing a random old item
            self.query_cache.pop(next(iter(self.query_cache)))
        
        # Also cache in Redis if available
        if self.cache:
            await self.cache.set(
                cache_key, 
                result, 
                expire=settings.RAG_CACHE_TTL
            )
        
        # Log performance metrics
        query_time = (time.time() - start_time) * 1000
        logger.debug(f"RAG query completed in {query_time:.2f}ms")
        
        if self.metrics:
            self.metrics.record("rag_query_time", query_time)
        
        return result
    
    async def _preprocess_query(self, query_text: str) -> str:
        """
        Preprocess the query to optimize retrieval.
        
        Args:
            query_text: Original query text
            
        Returns:
            Processed query text
        """
        # Remove filler words and normalize
        filler_words = ["the", "a", "an", "in", "on", "at", "to", "for"]
        words = query_text.lower().split()
        processed_words = [w for w in words if w not in filler_words]
        
        # Extract real estate specific entities if present
        # This is a simplified example - in production would use NER
        real_estate_terms = ["apartment", "house", "property", "rent", "buy", "sqm", "m²"]
        important_terms = [w for w in processed_words if w in real_estate_terms]
        
        # If real estate terms found, prioritize them in the query
        if important_terms:
            # Move important terms to the front
            other_terms = [w for w in processed_words if w not in important_terms]
            processed_words = important_terms + other_terms
        
        return " ".join(processed_words)
    
    async def _rerank_documents(
        self, 
        documents: List[Any], 
        scores: List[float], 
        query: str
    ) -> Tuple[List[Any], List[float]]:
        """
        Rerank documents based on additional relevance factors.
        
        Args:
            documents: Retrieved documents
            scores: Similarity scores
            query: Processed query text
            
        Returns:
            Reranked documents and scores
        """
        # Calculate document freshness scores (0-1)
        freshness_scores = []
        current_time = time.time()
        
        for doc in documents:
            # Get document timestamp or use a default
            timestamp = doc.metadata.get("timestamp", 0)
            age_days = (current_time - timestamp) / (60 * 60 * 24)
            
            # Exponential decay based on age
            if age_days <= 0:
                freshness = 1.0
            else:
                # Half-life of 30 days
                freshness = np.exp(-age_days / 30)
            
            freshness_scores.append(freshness)
        
        # Calculate quality scores based on metadata
        quality_scores = []
        for doc in documents:
            # Example quality signals
            is_verified = doc.metadata.get("verified", False)
            rating = doc.metadata.get("rating", 0)
            
            # Combine signals into quality score (0-1)
            quality = 0.5  # Default
            if is_verified:
                quality += 0.3
            quality += min(rating / 10, 0.2)  # Max 0.2 contribution
            
            quality_scores.append(quality)
        
        # Combine all factors with weights
        combined_scores = []
        for i in range(len(documents)):
            # Weighted combination (customize weights based on importance)
            relevance_weight = 0.6
            freshness_weight = 0.2
            quality_weight = 0.2
            
            combined_score = (
                relevance_weight * scores[i] +
                freshness_weight * freshness_scores[i] +
                quality_weight * quality_scores[i]
            )
            
            combined_scores.append(combined_score)
        
        # Sort documents by combined score
        sorted_pairs = sorted(
            zip(documents, combined_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Unpack sorted results
        sorted_documents = [doc for doc, _ in sorted_pairs]
        sorted_scores = [score for _, score in sorted_pairs]
        
        return sorted_documents, sorted_scores
    
    async def _generate_context(self, documents: List[Any], query: str) -> str:
        """
        Generate optimized context for the LLM from retrieved documents.
        
        Args:
            documents: Retrieved documents
            query: Processed query
            
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        # Extract relevant parts from each document
        context_parts = []
        
        for doc in documents:
            # Extract document content
            content = doc.page_content
            metadata = doc.metadata
            
            # Format based on document type
            doc_type = metadata.get("type", "general")
            
            if doc_type == "property":
                # For property listings, include structured details
                context_parts.append(
                    f"PROPERTY LISTING: {content}\n"
                    f"Location: {metadata.get('location', 'Unknown')}\n"
                    f"Price: {metadata.get('price', 'Unknown')}\n"
                    f"Features: {metadata.get('features', 'None listed')}"
                )
            elif doc_type == "faq":
                # For FAQs, include question and answer
                context_parts.append(
                    f"FAQ: {metadata.get('question', content)}\n"
                    f"Answer: {content}"
                )
            else:
                # For general information
                context_parts.append(f"INFORMATION: {content}")
        
        # Combine with separators
        context = "\n\n".join(context_parts)
        
        # Ensure context isn't too long (token limit considerations)
        max_chars = settings.MAX_CONTEXT_LENGTH
        if len(context) > max_chars:
            context = context[:max_chars]
        
        return context
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        start_time = time.time()
        
        if not documents:
            return []
        
        # Prepare documents for vector store
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        try:
            # Add to vector store (use asyncio to make non-blocking)
            doc_ids = await asyncio.to_thread(
                self.vectorstore.add_texts,
                texts,
                metadatas
            )
            
            # Log success
            add_time = (time.time() - start_time) * 1000
            logger.info(f"Added {len(documents)} documents in {add_time:.2f}ms")
            
            if self.metrics:
                self.metrics.increment("documents_added", len(documents))
                self.metrics.record("document_add_time", add_time)
            
            # Clear caches due to new documents
            self.query_cache.clear()
            if self.cache:
                await self.cache.delete_pattern("rag:query:*")
            
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            if self.metrics:
                self.metrics.increment("document_add_errors")
            return []
    
    async def delete_documents(self, doc_ids: List[str]) -> bool:
        """
        Delete documents from the knowledge base.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Success status
        """
        if not doc_ids:
            return True
            
        try:
            # Delete from vector store
            # Note: This implementation depends on PGVector's capabilities
            for doc_id in doc_ids:
                await asyncio.to_thread(
                    self._delete_document_by_id,
                    doc_id
                )
            
            # Clear caches due to document changes
            self.query_cache.clear()
            if self.cache:
                await self.cache.delete_pattern("rag:query:*")
                
            logger.info(f"Deleted {len(doc_ids)} documents")
            if self.metrics:
                self.metrics.increment("documents_deleted", len(doc_ids))
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            if self.metrics:
                self.metrics.increment("document_delete_errors")
            return False
    
    def _delete_document_by_id(self, doc_id: str) -> None:
        """
        Delete a single document by ID (implementation depends on vector store).
        
        Args:
            doc_id: Document ID to delete
        """
        # This would be implemented based on your specific vector store
        # For PostgreSQL with pgvector, you might use:
        with self.vectorstore.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"DELETE FROM {self.collection_name} WHERE metadata->>'id' = %s",
                (doc_id,)
            )
            conn.commit()
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the document collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            # Get document count (implementation depends on vector store)
            doc_count = await asyncio.to_thread(self._get_document_count)
            
            # Calculate cache hit rate
            cache_hit_rate = 0
            if self.total_queries > 0:
                cache_hit_rate = self.cache_hits / self.total_queries
                
            return {
                "document_count": doc_count,
                "cache_hit_rate": cache_hit_rate,
                "total_queries": self.total_queries,
                "cache_hits": self.cache_hits,
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {
                "error": str(e),
                "document_count": 0,
                "cache_hit_rate": 0,
                "total_queries": self.total_queries,
                "cache_hits": self.cache_hits,
            }
    
    def _get_document_count(self) -> int:
        """
        Get the total number of documents in the collection.
        
        Returns:
            Document count
        """
        # Implementation depends on vector store
        # For PostgreSQL with pgvector:
        with self.vectorstore.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {self.collection_name}")
            count = cursor.fetchone()[0]
            return count