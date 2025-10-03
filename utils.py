from langchain_core.documents import Document
from typing import List, Optional, Dict, Any
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.stores import BaseStore
from langchain.vectorstores import VectorStore
import numpy as np

class MMRParentDocumentRetriever(ParentDocumentRetriever):
    """
    Custom Parent Document Retriever that combines MMR (Maximum Marginal Relevance) 
    with Parent Document Retrieval for better diversity and context.
    """
    
    def __init__(
        self,
        vectorstore: VectorStore,
        docstore: BaseStore,
        child_splitter: Optional[Any] = None,
        parent_splitter: Optional[Any] = None,
        id_key: str = "parent_id",
        **kwargs
    ):
        super().__init__(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            id_key=id_key,
            **kwargs
        )
    
    def get_relevant_documents(
        self, 
        query: str, 
        *, 
        k: int = 5, 
        fetch_k: int = 20, 
        lambda_mult: float = 0.5,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve parent documents using MMR on child chunks.
        
        Args:
            query: The query string
            k: Number of final results to return
            fetch_k: Number of candidates to consider for MMR
            lambda_mult: Balance between relevance (1.0) and diversity (0.0)
        
        Returns:
            List of parent documents selected via MMR
        """
        
        # Step 1: MMR on child chunks
        child_docs = self.vectorstore.max_marginal_relevance_search(
            query,
            k=fetch_k,  # Get more candidates for MMR
            fetch_k=fetch_k * 2,  # Even larger pool for diversity
            lambda_mult=lambda_mult,
            **kwargs
        )
        
        # Step 2: Map child chunks back to parent docs with MMR scoring
        parent_docs = []
        parent_scores = {}
        seen_parents = set()
        
        for child_doc in child_docs:
            parent_id = child_doc.metadata.get(self.id_key)
            if parent_id and parent_id not in seen_parents:
                # Use mget instead of get for InMemoryStore
                parent_doc_results = self.docstore.mget([parent_id])
                if parent_doc_results and parent_doc_results[0] is not None:
                    parent_doc = parent_doc_results[0]
                    # Store parent doc with its relevance score
                    parent_scores[parent_id] = {
                        'doc': parent_doc,
                        'score': getattr(child_doc, 'score', 0.0)
                    }
                    seen_parents.add(parent_id)
        
        # Step 3: Sort by relevance and return top k
        sorted_parents = sorted(
            parent_scores.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        return [item['doc'] for item in sorted_parents[:k]]
    
    def get_relevant_documents_with_scores(
        self, 
        query: str, 
        *, 
        k: int = 5, 
        fetch_k: int = 20, 
        lambda_mult: float = 0.5,
        **kwargs
    ) -> List[tuple]:
        """
        Retrieve parent documents with their relevance scores.
        
        Returns:
            List of tuples (Document, score)
        """
        
        # Step 1: MMR on child chunks
        child_docs = self.vectorstore.max_marginal_relevance_search(
            query,
            k=fetch_k,
            fetch_k=fetch_k * 2,
            lambda_mult=lambda_mult,
            **kwargs
        )
        
        # Step 2: Map child chunks back to parent docs with scores
        parent_docs = []
        parent_scores = {}
        seen_parents = set()
        
        for child_doc in child_docs:
            parent_id = child_doc.metadata.get(self.id_key)
            if parent_id and parent_id not in seen_parents:
                # Use mget instead of get for InMemoryStore
                parent_doc_results = self.docstore.mget([parent_id])
                if parent_doc_results and parent_doc_results[0] is not None:
                    parent_doc = parent_doc_results[0]
                    parent_scores[parent_id] = {
                        'doc': parent_doc,
                        'score': getattr(child_doc, 'score', 0.0)
                    }
                    seen_parents.add(parent_id)
        
        # Step 3: Sort by relevance and return top k with scores
        sorted_parents = sorted(
            parent_scores.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        return [(item['doc'], item['score']) for item in sorted_parents[:k]]


def create_mmr_parent_retriever(
    documents: List[Document],
    embeddings,
    child_chunk_size: int = 400,
    child_chunk_overlap: int = 150,
    parent_chunk_size: int = 1000,
    parent_chunk_overlap: int = 400,
    vectorstore_class=None,
    persist_directory: str = None
) -> MMRParentDocumentRetriever:
    """
    Helper function to create an MMR Parent Document Retriever.
    
    Args:
        documents: List of documents to process
        embeddings: Embedding model
        child_chunk_size: Size of child chunks
        child_chunk_overlap: Overlap between child chunks
        parent_chunk_size: Size of parent chunks
        parent_chunk_overlap: Overlap between parent chunks
        vectorstore_class: Vector store class to use
        persist_directory: Directory to persist vector store
    
    Returns:
        Configured MMRParentDocumentRetriever
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma
    from langchain.storage import InMemoryStore
    
    # Create splitters
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap
    )
    
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap
    )
    
    # Create parent chunks first
    parent_chunks = parent_splitter.split_documents(documents)
    
    # Create child chunks from parent chunks
    child_chunks = []
    for i, parent_chunk in enumerate(parent_chunks):
        parent_id = f"parent_{i}"
        # Add parent_id to parent chunk metadata
        parent_chunk.metadata["parent_id"] = parent_id
        
        # Split parent chunk into child chunks
        child_chunks_from_parent = child_splitter.split_documents([parent_chunk])
        
        # Add parent_id to each child chunk
        for child_chunk in child_chunks_from_parent:
            child_chunk.metadata["parent_id"] = parent_id
            child_chunks.append(child_chunk)
    
    # Create vector store for child chunks
    if persist_directory:
        vectorstore = Chroma.from_documents(
            documents=child_chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=child_chunks,
            embedding=embeddings
        )
    
    # Create parent document store
    parent_store = InMemoryStore()
    
    # Store parent documents with IDs
    parent_ids = [f"parent_{i}" for i in range(len(parent_chunks))]
    parent_store.mset([(parent_ids[i], parent_chunks[i]) for i in range(len(parent_chunks))])
    
    # Create MMR Parent Document Retriever
    retriever = MMRParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=parent_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        id_key="parent_id"
    )
    
    return retriever
