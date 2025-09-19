# RAG--Based-Question-Answering-Assistant
# Install required packages
!pip install -q langchain faiss-cpu sentence-transformers transformers datasets accelerate
!pip install -q --upgrade huggingface_hub
!pip install -q pypdf

# Import necessary libraries
import os
import re
from typing import List, Dict, Any
import numpy as np
import faiss

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Hugging Face imports
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    pipeline
)
from sentence_transformers import SentenceTransformer

# For embedding and retrieval
from sklearn.metrics.pairwise import cosine_similarity

class RAGAssistant:
    def __init__(self, model_name: str = "google/flan-t5-base", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the RAG Assistant
        
        Args:
            model_name: Name of the LLM model to use for generation
            embedding_model: Name of the embedding model to use
        """
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        self.vectorstore = None
        self.qa_chain = None
        self.embeddings = None
        self.retriever = None
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_llm()
        
    def _initialize_embeddings(self):
        """Initialize the embedding model"""
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embedding model loaded successfully!")
    
    def _initialize_llm(self):
        """Initialize the language model"""
        print(f"Loading language model: {self.model_name}")
        
        # Use Seq2Seq model for T5 models
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype="auto"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Create text generation pipeline
        text_generation_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=0.3,
            do_sample=True,
            repetition_penalty=1.1,
            max_new_tokens=256,
        )
        
        self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        print("Language model loaded successfully!")
    
    def load_documents_from_text(self, texts: List[str], metadata: List[Dict] = None):
        """
        Load documents from a list of text strings
        
        Args:
            texts: List of text strings
            metadata: Optional list of metadata dictionaries for each text
        """
        print("Loading documents from text...")
        
        if metadata is None:
            metadata = [{"source": f"text_{i}"} for i in range(len(texts))]
        
        if len(texts) != len(metadata):
            raise ValueError("Texts and metadata must have the same length")
        
        self.documents = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadata)]
        print(f"Loaded {len(self.documents)} documents!")
        
        # Split documents into chunks
        self._split_documents()
    
    def _split_documents(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Split documents into chunks for processing"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        self.texts = text_splitter.split_documents(self.documents)
        print(f"Split into {len(self.texts)} text chunks!")
    
    def create_vectorstore(self):
        """Create FAISS vector store from documents"""
        if not hasattr(self, 'texts') or not self.texts:
            raise ValueError("No documents loaded. Please load documents first.")
        
        print("Creating FAISS vector store...")
        self.vectorstore = FAISS.from_documents(self.texts, self.embeddings)
        print("FAISS vector store created successfully!")
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    
    def create_qa_chain(self):
        """Create the QA chain with custom prompt"""
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Provide a detailed, comprehensive answer based strictly on the context.

        Context: {context}

        Question: {question}
        Answer: """
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        print("QA chain created successfully!")
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Please call create_qa_chain() first.")
        
        result = self.qa_chain({"query": question})
        return result
    
    def evaluate_retrieval(self, queries: List[str], ground_truths: List[List[str]] = None):
        """
        Evaluate the retrieval performance of the system
        
        Args:
            queries: List of query strings
            ground_truths: Optional list of ground truth relevant documents for each query
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized.")
        
        results = {}
        
        for i, query in enumerate(queries):
            # Get embeddings for query
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in vector store
            retrieved_docs = self.vectorstore.similarity_search(query, k=5)
            
            # Calculate similarity scores
            doc_embeddings = [self.embeddings.embed_query(doc.page_content) for doc in retrieved_docs]
            similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
            
            results[query] = {
                "retrieved_documents": [doc.page_content for doc in retrieved_docs],
                "similarity_scores": similarities.tolist()
            }
            
            # Compare with ground truth if available
            if ground_truths and i < len(ground_truths):
                # Simple evaluation - check if any ground truth appears in retrieved docs
                gt_found = any(
                    any(gt in doc.page_content for gt in ground_truths[i]) 
                    for doc in retrieved_docs
                )
                results[query]["ground_truth_found"] = gt_found
        
        return results
    
    def save_vectorstore(self, path: str):
        """Save the FAISS vector store to disk"""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"Vector store saved to {path}")
        else:
            print("No vector store to save.")
    
    def load_vectorstore(self, path: str):
        """Load a FAISS vector store from disk"""
        if os.path.exists(path):
            self.vectorstore = FAISS.load_local(path, self.embeddings)
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            print(f"Vector store loaded from {path}")
        else:
            print(f"Path {path} does not exist.")

# Enhanced RAG with better embeddings and optimization techniques
class EnhancedRAGAssistant(RAGAssistant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _initialize_embeddings(self):
        """Initialize with better embedding model"""
        print(f"Loading enhanced embedding model: {self.embedding_model_name}")
        
        # Create a wrapper for LangChain compatibility
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32
            }
        )
        
        # Also initialize a sentence transformer for advanced operations
        self.sentence_transformer = SentenceTransformer(self.embedding_model_name)
        print("Enhanced embedding model loaded successfully!")
    
    def optimize_retrieval(self, queries: List[str], alpha: float = 0.7):
        """
        Implement hybrid search with query expansion
        
        Args:
            queries: List of queries to optimize
            alpha: Weight for semantic vs keyword search
            
        Returns:
            Optimized retrieval results
        """
        optimized_results = {}
        
        for query in queries:
            # Query expansion - add related terms
            expanded_query = self._expand_query(query)
            
            # Hybrid search - combine semantic and keyword search
            semantic_results = self.vectorstore.similarity_search(expanded_query, k=10)
            
            # Simple keyword matching
            keyword_matches = []
            query_terms = query.lower().split()
            
            for doc in self.texts:
                content = doc.page_content.lower()
                match_score = sum(1 for term in query_terms if term in content)
                if match_score > 0:
                    keyword_matches.append((doc, match_score))
            
            # Sort by match score and take top results
            keyword_matches.sort(key=lambda x: x[1], reverse=True)
            keyword_results = [doc for doc, score in keyword_matches[:5]]
            
            # Combine results using weighted scoring
            combined_results = self._combine_results(semantic_results, keyword_results, alpha)
            optimized_results[query] = combined_results
        
        return optimized_results
    
    def _expand_query(self, query: str) -> str:
        """Simple query expansion technique"""
        # Map of terms to related concepts
        expansion_map = {
            "ai": ["artificial intelligence", "machine learning", "neural networks"],
            "ml": ["machine learning", "deep learning", "algorithms"],
            "nlp": ["natural language processing", "text processing", "language models"],
            "transformer": ["attention mechanism", "neural architecture", "BERT GPT"],
        }
        
        expanded_query = query.lower()
        for term, expansions in expansion_map.items():
            if term in expanded_query:
                expanded_query += " " + " ".join(expansions)
        
        return expanded_query
    
    def _combine_results(self, semantic_results, keyword_results, alpha):
        """Combine semantic and keyword search results"""
        all_docs = {}
        
        # Add semantic results with weight (1-alpha)
        for i, doc in enumerate(semantic_results):
            score = (1 - alpha) * (len(semantic_results) - i) / len(semantic_results)
            all_docs[doc.page_content] = all_docs.get(doc.page_content, 0) + score
        
        # Add keyword results with weight alpha
        for i, doc in enumerate(keyword_results):
            score = alpha * (len(keyword_results) - i) / len(keyword_results)
            all_docs[doc.page_content] = all_docs.get(doc.page_content, 0) + score
        
        # Sort by combined score
        sorted_docs = sorted(all_docs.items(), key=lambda x: x[1], reverse=True)
        
        # Return top documents
        return [Document(page_content=content) for content, score in sorted_docs[:5]]
    
    def rerank_results(self, query: str, documents: List[Document], top_k: int = 3):
        """
        Rerank documents based on relevance to query
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top results to return
            
        Returns:
            Reranked documents
        """
        if not documents:
            return documents
        
        # Calculate similarity scores
        query_embedding = self.sentence_transformer.encode([query])[0]
        doc_embeddings = self.sentence_transformer.encode([doc.page_content for doc in documents])
        
        # Compute cosine similarities
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # Pair documents with their similarity scores
        scored_docs = list(zip(documents, similarities))
        
        # Sort by similarity score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k documents
        return [doc for doc, score in scored_docs[:top_k]]

# Example usage and demonstration
def main():
    # Initialize the RAG assistant
    rag = EnhancedRAGAssistant(
        model_name="google/flan-t5-base",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create sample documents
    sample_docs = [
        "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans.",
        "Machine learning (ML) is a field of inquiry devoted to understanding and building methods that learn from data.",
        "Natural language processing (NLP) is a subfield of AI focused on enabling computers to understand human language.",
        "The Transformer architecture, introduced in 2017, revolutionized NLP with its attention mechanism.",
        "Deep learning uses neural networks with many layers to model complex patterns in data.",
        "Computer vision enables machines to interpret and understand visual information from the world.",
        "Reinforcement learning is a type of machine learning where agents learn by interacting with an environment.",
        "Large language models like GPT-3 and BERT have significantly advanced the field of NLP in recent years."
    ]
    
    # Load documents
    rag.load_documents_from_text(sample_docs)
    
    # Create vector store
    rag.create_vectorstore()
    
    # Create QA chain
    rag.create_qa_chain()
    
    # Ask some questions
    questions = [
        "What is artificial intelligence?",
        "What is machine learning?",
        "What is natural language processing?",
        "What is the Transformer architecture?",
        "What are large language models?"
    ]
    
    print("RAG Question Answering System")
    print("=" * 50)
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = rag.query(question)
        print(f"Answer: {result['result']}")
        print("-" * 50)
    
    # Evaluate retrieval
    print("\nRetrieval Evaluation:")
    eval_queries = ["What is AI?", "Tell me about machine learning"]
    evaluation = rag.evaluate_retrieval(eval_queries)
    
    for query, result in evaluation.items():
        print(f"Query: {query}")
        print(f"Top similarity score: {max(result['similarity_scores']):.4f}")
        print()

# Run the demo
if __name__ == "__main__":
    main()
