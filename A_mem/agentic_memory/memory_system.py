import keyword
from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime
from .llm_controller import LLMController
from .retrievers import ChromaRetriever
import json
import logging
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import word_tokenize
import pickle
from pathlib import Path
from litellm import completion
import time

logger = logging.getLogger(__name__)

class MemoryNote:
    """A memory note that represents a single unit of information in the memory system.
    
    This class encapsulates all metadata associated with a memory, including:
    - Core content and identifiers
    - Temporal information (creation and access times)
    - Semantic metadata (keywords, context, tags)
    - Relationship data (links to other memories)
    - Usage statistics (retrieval count)
    - Evolution tracking (history of changes)
    """
    
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None):
        """Initialize a new memory note with its associated metadata.
        
        Args:
            content (str): The main text content of the memory
            id (Optional[str]): Unique identifier for the memory. If None, a UUID will be generated
            keywords (Optional[List[str]]): Key terms extracted from the content
            links (Optional[Dict]): References to related memories
            retrieval_count (Optional[int]): Number of times this memory has been accessed
            timestamp (Optional[str]): Creation time in format YYYYMMDDHHMM
            last_accessed (Optional[str]): Last access time in format YYYYMMDDHHMM
            context (Optional[str]): The broader context or domain of the memory
            evolution_history (Optional[List]): Record of how the memory has evolved
            category (Optional[str]): Classification category
            tags (Optional[List[str]]): Additional classification tags
        """
        # Core content and ID
        self.content = content
        self.id = id or str(uuid.uuid4())
        
        # Semantic metadata
        self.keywords = keywords or []
        self.links = links or []
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []
        
        # Temporal information
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        
        # Usage and evolution data
        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []

class AgenticMemorySystem:
    """Core memory system that manages memory notes and their evolution.
    
    This system provides:
    - Memory creation, retrieval, update, and deletion
    - Content analysis and metadata extraction
    - Memory evolution and relationship management
    - Hybrid search capabilities
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 chroma_db_path: Optional[str] = None):  
        """Initialize the memory system.
        
        Args:
            model_name: Name of the sentence transformer model
            llm_backend: LLM backend to use (openai/ollama)
            llm_model: Name of the LLM model
            evo_threshold: Number of memories before triggering evolution
            api_key: API key for the LLM service
            api_base: Base URL for the LLM API (e.g., "http://localhost:8040/v1" for vLLM)
            chroma_db_path: Optional path to persist ChromaDB data. If None, uses in-memory mode.
        """
        self.memories = {}
        self.model_name = model_name
        
        # Initialize ChromaDB retriever
        # Note: We don't reset/delete the collection here, as it may contain existing data
        # If you want to start fresh, delete the collection manually or use clear_existing_db option
        self.retriever = ChromaRetriever(
            collection_name="memories",
            model_name=self.model_name,
            chroma_db_path=chroma_db_path
        )
        
        # Initialize LLM controller
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, api_base)
        
        # Save parameters for pickle serialization
        self._llm_backend = llm_backend
        self._llm_model = llm_model
        self._api_key = api_key
        self._api_base = api_base
        self._chroma_db_path = chroma_db_path  # 保存 chroma_db_path 以便 pickle 序列化
        
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

        # Evolution system prompt
        self._evolution_system_prompt = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.  

                                The new memory context:
                                {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories:
                                {nearest_neighbors_memories}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["neighbor_memory_ids"],
                                    "tags_to_update": ["tag_1",..."tag_n"], 
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}
                                '''
        
    def analyze_content(self, content: str) -> Dict:            
        """Analyze content using LLM to extract semantic metadata.
        
        Uses a language model to understand the content and extract:
        - Keywords: Important terms and concepts
        - Context: Overall domain or theme
        - Tags: Classification categories
        
        Args:
            content (str): The text content to analyze
            
        Returns:
            Dict: Contains extracted metadata with keys:
                - keywords: List[str]
                - context: str
                - tags: List[str]
        """
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content
        try:
            response = self.llm_controller.llm.get_completion(prompt, response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "context": {
                                    "type": "string",
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            }
                        }
                    }})
            return json.loads(response)
        except Exception as e:
            print(f"Error analyzing content: {e}")
            return {"keywords": [], "context": "General", "tags": []}

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note"""
        # Create MemoryNote without llm_controller
        if time is not None:
            kwargs['timestamp'] = time
        note = MemoryNote(content=content, **kwargs)
        
        # Update retriever with all documents
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        
        # Add to ChromaDB with essential metadata only (optimized for storage)
        # Note: retrieval_count, last_accessed, evolution_history are kept in MemoryNote object
        # but not stored in ChromaDB to reduce storage size
        metadata = {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "timestamp": note.timestamp,
            "context": note.context,
            "category": note.category,
            "tags": note.tags
        }
        self.retriever.add_document(note.content, metadata, note.id)
        
        if evo_label == True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id
    
    def batch_add_notes(self, notes_data: List[Dict[str, Any]], max_workers: int = 5, show_progress: bool = False, chromadb_batch_size: int = 100, evolution_sample_rate: int = 1) -> List[str]:
        """Add multiple memory notes, following the same logic as add_note.
        
        This method processes memories sequentially (same as add_note), where each note is
        fully processed (including LLM evolution) before moving to the next. This ensures
        that subsequent memories can find previously added memories as neighbors.
        
        Processing order for each note (same as add_note):
        1. Create MemoryNote
        2. Call process_memory (calls LLM for evolution, uses previously added memories as neighbors)
        3. Add to memories dict (using processed_note)
        4. Add to ChromaDB (using processed_note metadata)
        5. Handle evolution threshold
        
        Args:
            notes_data: List of dictionaries, each containing:
                - 'content': str (required) - The memory content
                - 'keywords': List[str] (optional) - Keywords for the memory
                - Other optional fields: 'time', 'context', 'category', 'tags', etc.
            max_workers: Unused (kept for API compatibility, processing is sequential)
            show_progress: Whether to show progress bar (requires tqdm)
            chromadb_batch_size: Unused (kept for API compatibility)
            evolution_sample_rate: Perform evolution only for every Nth note (default: 1 = all notes).
                                   Higher values speed up batch processing but reduce evolution coverage.
                                   For example, evolution_sample_rate=10 means only every 10th note gets evolution.
            
        Returns:
            List[str]: List of memory IDs for the added notes
        """
        memory_ids = []
        
        # Progress bar setup
        if show_progress:
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=len(notes_data), desc="Adding memories")
            except ImportError:
                show_progress = False
                progress_bar = None
        else:
            progress_bar = None
        
        # Two-phase approach for parallelization:
        # Phase 1: Quickly add all notes to ChromaDB (no LLM calls, just embedding and storage)
        # Phase 2: Process evolution in parallel (LLM calls can run concurrently, with locks for shared state)
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        notes_created = []  # List of (note, idx, should_evolve) tuples
        
        # Phase 1: Create all notes and add to ChromaDB quickly (sequential, but fast - no LLM)
        for idx, note_data in enumerate(notes_data):
            content = note_data.get('content', '')
            if not content:
                if progress_bar:
                    progress_bar.update(1)
                continue
            
            # Extract optional parameters
            time_str = note_data.get('time')
            keywords = note_data.get('keywords', [])
            context = note_data.get('context')
            category = note_data.get('category')
            tags = note_data.get('tags', [])
            
            # Create MemoryNote
            if time_str is not None:
                kwargs = {'timestamp': time_str}
            else:
                kwargs = {}
            if keywords:
                kwargs['keywords'] = keywords
            if context:
                kwargs['context'] = context
            if category:
                kwargs['category'] = category
            if tags:
                kwargs['tags'] = tags
            
            note = MemoryNote(content=content, **kwargs)
            
            # Add to memories dict immediately
            self.memories[note.id] = note
            memory_ids.append(note.id)
            
            # Add to ChromaDB immediately (so all notes are available for neighbor search)
            # Optimized: Only store essential fields for retrieval, not usage stats
            metadata = {
                "id": note.id,
                "content": note.content,
                "keywords": note.keywords,
                "links": note.links,
                "timestamp": note.timestamp,
                "context": note.context,
                "category": note.category,
                "tags": note.tags
            }
            self.retriever.add_document(note.content, metadata, note.id)
            
            # Check if this note should go through evolution
            should_evolve_note = (evolution_sample_rate == 1) or (idx % evolution_sample_rate == 0)
            if should_evolve_note and len(self.memories) > 1:  # Need at least one other memory
                notes_created.append((note, idx, True))
            else:
                notes_created.append((note, idx, False))
            
            if progress_bar:
                progress_bar.update(1)
        
        # Phase 2: Process evolution in parallel (LLM calls can run concurrently)
        # Use locks to protect shared state (self.memories, ChromaDB updates)
        if notes_created and any(should_evolve for _, _, should_evolve in notes_created):
            logger.info(f"Processing evolution for {sum(1 for _, _, se in notes_created if se)} memories in parallel (max_workers={max_workers})...")
            
            evolution_lock = threading.Lock()
            evolution_results = {}  # {idx: (evo_label, processed_note)}
            
            def process_evolution_with_lock(note_idx_should_evolve):
                """Process evolution for a single note (called in parallel)
                
                Optimization: Separate LLM call from ChromaDB update.
                - LLM call (process_memory) is done with minimal lock time (only when modifying self.memories)
                - ChromaDB updates are deferred and done in batch later
                """
                note, idx, should_evolve = note_idx_should_evolve
                if not should_evolve:
                    return idx, (False, note, None)
                
                try:
                    # Step 1: Call process_memory (includes LLM call)
                    # This is the slow part, but LLM I/O releases GIL allowing parallelism
                    evo_label, processed_note = self.process_memory(note)
                    
                    # Step 2: Update memories dict (need lock for thread safety)
                    with evolution_lock:
                        self.memories[processed_note.id] = processed_note
                        evolution_results[idx] = (evo_label, processed_note)
                    
                    # Step 3: Prepare metadata for ChromaDB update (done outside lock)
                    # ChromaDB update will be done in batch later
                    metadata = {
                        "id": processed_note.id,
                        "content": processed_note.content,
                        "keywords": processed_note.keywords,
                        "links": processed_note.links,
                        "timestamp": processed_note.timestamp,
                        "context": processed_note.context,
                        "category": processed_note.category,
                        "tags": processed_note.tags
                    }
                    
                    return idx, (evo_label, processed_note, metadata)
                    
                except Exception as e:
                    logger.error(f"Error processing evolution for note {idx}: {e}")
                    with evolution_lock:
                        evolution_results[idx] = (False, note)
                    return idx, (False, note, None)
            
            # Process evolution in parallel (LLM calls run concurrently, ChromaDB updates batched)
            notes_to_evolve = [(note, idx, should_evolve) for note, idx, should_evolve in notes_created if should_evolve]
            
            # Add progress bar for evolution processing
            evolution_progress_bar = None
            if show_progress:
                try:
                    from tqdm import tqdm
                    evolution_progress_bar = tqdm(
                        total=len(notes_to_evolve), 
                        desc="Evolution processing",
                        unit="memory",
                        leave=True  # Keep progress bar after completion
                    )
                except ImportError:
                    evolution_progress_bar = None
            
            # Step 1: Process all evolutions in parallel (LLM calls)
            # Collect metadata for batch ChromaDB update
            chromadb_updates = []  # List of (doc_id, document, metadata) tuples
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_evolution_with_lock, item): item for item in notes_to_evolve}
                for future in as_completed(futures):
                    idx, (evo_label, processed_note, metadata) = future.result()
                    if evolution_progress_bar:
                        evolution_progress_bar.update(1)
                    
                    # Collect metadata for batch update
                    if metadata is not None:
                        chromadb_updates.append((
                            processed_note.id,
                            processed_note.content,
                            metadata
                        ))
            
            if evolution_progress_bar:
                evolution_progress_bar.close()
            
            # Step 2: Batch update ChromaDB (much faster than individual updates)
            if chromadb_updates:
                logger.info(f"Batch updating ChromaDB for {len(chromadb_updates)} evolved memories...")
                
                # Batch delete all documents first
                doc_ids_to_delete = [doc_id for doc_id, _, _ in chromadb_updates]
                self.retriever.delete_documents_batch(doc_ids_to_delete)
                
                # Batch add all updated documents
                documents = [doc for _, doc, _ in chromadb_updates]
                metadatas = [meta for _, _, meta in chromadb_updates]
                doc_ids = [doc_id for doc_id, _, _ in chromadb_updates]
                
                self.retriever.add_documents_batch(documents, metadatas, doc_ids)
                logger.info(f"✓ Batch updated {len(chromadb_updates)} memories in ChromaDB")
            
            # Handle evolution threshold for all evolved notes (sequential to avoid conflicts)
            for idx, (evo_label, _) in evolution_results.items():
                if evo_label:
                    self.evo_cnt += 1
                    if self.evo_cnt % self.evo_threshold == 0:
                        self.consolidate_memories()
            
            logger.info(f"Completed evolution processing for {len(evolution_results)} memories")
        
        if progress_bar:
            progress_bar.close()
        
        return memory_ids
    
    def consolidate_memories(self):
        """Consolidate memories: update retriever with new documents"""
        # Reset ChromaDB collection (preserve chroma_db_path if it was set)
        # 优先使用显式保存的 _chroma_db_path
        chroma_db_path = getattr(self, '_chroma_db_path', None)
        if chroma_db_path is None:
            chroma_db_path = getattr(self.retriever, 'chroma_db_path', None)
        self.retriever = ChromaRetriever(
            collection_name="memories",
            model_name=self.model_name,
            chroma_db_path=chroma_db_path
        )
        
        # Re-add all memory documents with their complete metadata
        for memory in self.memories.values():
            # Optimized: Only store essential fields for retrieval
            metadata = {
                "id": memory.id,
                "content": memory.content,
                "keywords": memory.keywords,
                "links": memory.links,
                "timestamp": memory.timestamp,
                "context": memory.context,
                "category": memory.category,
                "tags": memory.tags
            }
            self.retriever.add_document(memory.content, metadata, memory.id)
    
    def find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[int]]:
        """Find related memories using ChromaDB retrieval"""
        if not self.memories:
            return "", []
            
        try:
            # Get results from ChromaDB
            results = self.retriever.search(query, k)
            
            # Convert to list of memories
            memory_str = ""
            indices = []
            
            if 'ids' in results and results['ids'] and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
                    # Get metadata from ChromaDB results
                    if i < len(results['metadatas'][0]):
                        metadata = results['metadatas'][0][i]
                        # Format memory string
                        memory_str += f"memory index:{i}\ttalk start time:{metadata.get('timestamp', '')}\tmemory content: {metadata.get('content', '')}\tmemory context: {metadata.get('context', '')}\tmemory keywords: {str(metadata.get('keywords', []))}\tmemory tags: {str(metadata.get('tags', []))}\n"
                        indices.append(i)
                    
            return memory_str, indices
        except Exception as e:
            logger.error(f"Error in find_related_memories: {str(e)}")
            return "", []

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories using ChromaDB retrieval in raw format"""
        if not self.memories:
            return ""
            
        # Get results from ChromaDB
        results = self.retriever.search(query, k)
        
        # Convert to list of memories
        memory_str = ""
        
        if 'ids' in results and results['ids'] and len(results['ids']) > 0:
            for i, doc_id in enumerate(results['ids'][0][:k]):
                if i < len(results['metadatas'][0]):
                    # Get metadata from ChromaDB results
                    metadata = results['metadatas'][0][i]
                    
                    # Add main memory info
                    memory_str += f"talk start time:{metadata.get('timestamp', '')}\tmemory content: {metadata.get('content', '')}\tmemory context: {metadata.get('context', '')}\tmemory keywords: {str(metadata.get('keywords', []))}\tmemory tags: {str(metadata.get('tags', []))}\n"
                    
                    # Add linked memories if available
                    links = metadata.get('links', [])
                    j = 0
                    for link_id in links:
                        if link_id in self.memories and j < k:
                            neighbor = self.memories[link_id]
                            memory_str += f"talk start time:{neighbor.timestamp}\tmemory content: {neighbor.content}\tmemory context: {neighbor.context}\tmemory keywords: {str(neighbor.keywords)}\tmemory tags: {str(neighbor.tags)}\n"
                            j += 1
                            
        return memory_str

    def read(self, memory_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to retrieve
            
        Returns:
            MemoryNote if found, None otherwise
        """
        return self.memories.get(memory_id)
    
    def update(self, memory_id: str, **kwargs) -> bool:
        """Update a memory note.
        
        Args:
            memory_id: ID of memory to update
            **kwargs: Fields to update
            
        Returns:
            bool: True if update successful
        """
        if memory_id not in self.memories:
            return False
            
        note = self.memories[memory_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)
                
        # Update in ChromaDB
        # Optimized: Only store essential fields for retrieval
        metadata = {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "timestamp": note.timestamp,
            "context": note.context,
            "category": note.category,
            "tags": note.tags
        }
        
        # Delete and re-add to update
        self.retriever.delete_document(memory_id)
        self.retriever.add_document(document=note.content, metadata=metadata, doc_id=memory_id)
        
        return True
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to delete
            
        Returns:
            bool: True if memory was deleted, False if not found
        """
        if memory_id in self.memories:
            # Delete from ChromaDB
            self.retriever.delete_document(memory_id)
            # Delete from local storage
            del self.memories[memory_id]
            return True
        return False
    
    def _search_raw(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Internal search method that returns raw results from ChromaDB.
        
        This is used internally by the memory evolution system to find
        related memories for potential evolution.
        
        Args:
            query (str): The search query text
            k (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: Raw search results from ChromaDB
        """
        results = self.retriever.search(query, k)
        return [{'id': doc_id, 'score': score} 
                for doc_id, score in zip(results['ids'][0], results['distances'][0])]
                
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using a hybrid retrieval approach."""
        # Get results from ChromaDB (only do this once)
        search_results = self.retriever.search(query, k)
        memories = []
        
        # Process ChromaDB results
        for i, doc_id in enumerate(search_results['ids'][0]):
            memory = self.memories.get(doc_id)
            if memory:
                memories.append({
                    'id': doc_id,
                    'content': memory.content,
                    'context': memory.context,
                    'keywords': memory.keywords,
                    'score': search_results['distances'][0][i]
                })
        
        return memories[:k]
    
    def _search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using a hybrid retrieval approach.
        
        This method combines results from both:
        1. ChromaDB vector store (semantic similarity)
        2. Embedding-based retrieval (dense vectors)
        
        The results are deduplicated and ranked by relevance.
        
        Args:
            query (str): The search query text
            k (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results, each containing:
                - id: Memory ID
                - content: Memory content
                - score: Similarity score
                - metadata: Additional memory metadata
        """
        # Get results from ChromaDB
        chroma_results = self.retriever.search(query, k)
        memories = []
        
        # Process ChromaDB results
        for i, doc_id in enumerate(chroma_results['ids'][0]):
            memory = self.memories.get(doc_id)
            if memory:
                memories.append({
                    'id': doc_id,
                    'content': memory.content,
                    'context': memory.context,
                    'keywords': memory.keywords,
                    'score': chroma_results['distances'][0][i]
                })
                
        # Get results from embedding retriever
        embedding_results = self.retriever.search(query, k)
        
        # Combine results with deduplication
        seen_ids = set(m['id'] for m in memories)
        for result in embedding_results:
            memory_id = result.get('id')
            if memory_id and memory_id not in seen_ids:
                memory = self.memories.get(memory_id)
                if memory:
                    memories.append({
                        'id': memory_id,
                        'content': memory.content,
                        'context': memory.context,
                        'keywords': memory.keywords,
                        'score': result.get('score', 0.0)
                    })
                    seen_ids.add(memory_id)
                    
        return memories[:k]

    def search_agentic(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using ChromaDB retrieval."""
        if not self.memories:
            return []
            
        try:
            # Get results from ChromaDB
            results = self.retriever.search(query, k)
            
            # Process results
            memories = []
            seen_ids = set()
            
            # Check if we have valid results
            if ('ids' not in results or not results['ids'] or 
                len(results['ids']) == 0 or len(results['ids'][0]) == 0):
                return []
                
            # Process ChromaDB results
            for i, doc_id in enumerate(results['ids'][0][:k]):
                if doc_id in seen_ids:
                    continue
                    
                if i < len(results['metadatas'][0]):
                    metadata = results['metadatas'][0][i]
                    
                    # Create result dictionary with all metadata fields
                    memory_dict = {
                        'id': doc_id,
                        'content': metadata.get('content', ''),
                        'context': metadata.get('context', ''),
                        'keywords': metadata.get('keywords', []),
                        'tags': metadata.get('tags', []),
                        'timestamp': metadata.get('timestamp', ''),
                        'category': metadata.get('category', 'Uncategorized'),
                        'is_neighbor': False
                    }
                    
                    # Add score if available
                    if 'distances' in results and len(results['distances']) > 0 and i < len(results['distances'][0]):
                        memory_dict['score'] = results['distances'][0][i]
                        
                    memories.append(memory_dict)
                    seen_ids.add(doc_id)
            
            # Add linked memories (neighbors)
            neighbor_count = 0
            for memory in list(memories):  # Use a copy to avoid modification during iteration
                if neighbor_count >= k:
                    break
                    
                # Get links from metadata
                links = memory.get('links', [])
                if not links and 'id' in memory:
                    # Try to get links from memory object
                    mem_obj = self.memories.get(memory['id'])
                    if mem_obj:
                        links = mem_obj.links
                        
                for link_id in links:
                    if link_id not in seen_ids and neighbor_count < k:
                        neighbor = self.memories.get(link_id)
                        if neighbor:
                            memories.append({
                                'id': link_id,
                                'content': neighbor.content,
                                'context': neighbor.context,
                                'keywords': neighbor.keywords,
                                'tags': neighbor.tags,
                                'timestamp': neighbor.timestamp,
                                'category': neighbor.category,
                                'is_neighbor': True
                            })
                            seen_ids.add(link_id)
                            neighbor_count += 1
            
            return memories[:k]
        except Exception as e:
            logger.error(f"Error in search_agentic: {str(e)}")
            return []

    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """Process a memory note and determine if it should evolve.
        
        Args:
            note: The memory note to process
            
        Returns:
            Tuple[bool, MemoryNote]: (should_evolve, processed_note)
        """
        # For first memory or testing, just return the note without evolution
        if not self.memories:
            return False, note
            
        try:
            # Get nearest neighbors
            neighbors_text, indices = self.find_related_memories(note.content, k=5)
            if not neighbors_text or not indices:
                return False, note
                
            # Format neighbors for LLM - in this case, neighbors_text is already formatted
            
            # Query LLM for evolution decision
            prompt = self._evolution_system_prompt.format(
                content=note.content,
                context=note.context,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbors_text,
                neighbor_number=len(indices)
            )
            
            try:
                # Use higher max_tokens for evolution responses (may contain multiple neighbors)
                # Default 1000 may not be enough when updating multiple neighbors
                # Increase to 3000 to handle cases with 5 neighbors × context/tags updates
                response_format_config = {
                    "type": "json_schema", 
                    "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {
                                    "type": "boolean"
                                },
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "suggested_connections": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_context_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "tags_to_update": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_tags_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                }
                            },
                            "required": ["should_evolve", "actions", "suggested_connections", 
                                      "tags_to_update", "new_context_neighborhood", "new_tags_neighborhood"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
                
                # Call LLM with increased max_tokens (3000 instead of default 1000)
                # This handles cases with multiple neighbors that need context/tag updates
                response = self.llm_controller.llm.get_completion(
                    prompt, 
                    response_format_config, 
                    temperature=0.7,
                    max_tokens=3000  # Increased from 1000 to handle longer responses with multiple neighbors
                )
                
                # Parse JSON with better error handling
                try:
                    response_json = json.loads(response)
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON decode error in process_memory: {json_err}")
                    logger.error(f"Response length: {len(response) if response else 0} chars")
                    if response:
                        logger.error(f"Response preview (first 500 chars): {response[:500]}")
                        logger.error(f"Response preview (last 500 chars): {response[-500:]}")
                        # Check if response seems truncated
                        if not response.rstrip().endswith('}'):
                            logger.warning("Response appears to be truncated (doesn't end with '}')")
                            logger.warning("This may indicate max_tokens is still too low or LLM response was incomplete")
                    raise
                should_evolve = response_json["should_evolve"]
                
                if should_evolve:
                    actions = response_json["actions"]
                    for action in actions:
                        if action == "strengthen":
                            suggest_connections = response_json["suggested_connections"]
                            new_tags = response_json["tags_to_update"]
                            note.links.extend(suggest_connections)
                            note.tags = new_tags
                        elif action == "update_neighbor":
                            new_context_neighborhood = response_json["new_context_neighborhood"]
                            new_tags_neighborhood = response_json["new_tags_neighborhood"]
                            noteslist = list(self.memories.values())
                            notes_id = list(self.memories.keys())
                            
                            for i in range(min(len(indices), len(new_tags_neighborhood))):
                                # Skip if we don't have enough neighbors
                                if i >= len(indices):
                                    continue
                                    
                                tag = new_tags_neighborhood[i]
                                if i < len(new_context_neighborhood):
                                    context = new_context_neighborhood[i]
                                else:
                                    # Since indices are just numbers now, we need to find the memory
                                    # In memory list using its index number
                                    if i < len(noteslist):
                                        context = noteslist[i].context
                                    else:
                                        continue
                                        
                                # Get index from the indices list
                                if i < len(indices):
                                    memorytmp_idx = indices[i]
                                    # Make sure the index is valid
                                    if memorytmp_idx < len(noteslist):
                                        notetmp = noteslist[memorytmp_idx]
                                        notetmp.tags = tag
                                        notetmp.context = context
                                        # Make sure the index is valid
                                        if memorytmp_idx < len(notes_id):
                                            self.memories[notes_id[memorytmp_idx]] = notetmp
                                
                return should_evolve, note
                
            except (json.JSONDecodeError, KeyError, Exception) as e:
                logger.error(f"Error in memory evolution: {str(e)}")
                return False, note
                
        except Exception as e:
            # For testing purposes, catch all exceptions and return the original note
            logger.error(f"Error in process_memory: {str(e)}")
            return False, note
    
    def __getstate__(self):
        """Custom pickle serialization - exclude retriever and llm_controller as they contain non-serializable objects"""
        state = self.__dict__.copy()
        # Save retriever configuration instead of the object itself
        # 优先使用显式保存的 _chroma_db_path，否则从 retriever 获取
        if '_chroma_db_path' not in state or state.get('_chroma_db_path') is None:
            if hasattr(self, 'retriever') and self.retriever is not None:
                state['_chroma_db_path'] = getattr(self.retriever, 'chroma_db_path', None)
        # Remove non-serializable objects
        state.pop('retriever', None)
        state.pop('llm_controller', None)
        return state
    
    def __setstate__(self, state):
        """Custom pickle deserialization - reconstruct retriever and llm_controller"""
        import time
        _perf_start = time.time()
        
        # Step 1: Update __dict__
        _step_start = time.time()
        self.__dict__.update(state)
        _step_time = time.time() - _step_start
        if _step_time > 0.1:
            logger.info(f"[__setstate__] Step 1 (update dict): {_step_time:.2f}s")
        
        # Step 2: Reconstruct retriever
        _step_start = time.time()
        chroma_db_path = state.get('_chroma_db_path', None)
        self.retriever = ChromaRetriever(
            collection_name="memories",
            model_name=self.model_name,
            chroma_db_path=chroma_db_path
        )
        _step_time = time.time() - _step_start
        if _step_time > 0.1:
            logger.info(f"[__setstate__] Step 2 (init retriever): {_step_time:.2f}s")
        
        # Step 3: Check if ChromaDB already has data (if using persistent storage)
        # Only re-add memories if ChromaDB is empty or data doesn't match
        _step_start = time.time()
        need_reload = False
        logger.info(f"[__setstate__] Step 3: Checking ChromaDB state (chroma_db_path={chroma_db_path})")
        if chroma_db_path is not None:
            # Using persistent storage - check if collection has data
            try:
                logger.info(f"[__setstate__] Step 3a: Getting collection count...")
                collection_count = self.retriever.collection.count()
                memory_count = len(self.memories)
                _check_time = time.time() - _step_start
                logger.info(f"[__setstate__] Step 3a: Collection count={collection_count}, Memory count={memory_count} (took {_check_time:.2f}s)")
                
                # If collection is empty or count doesn't match, need to reload
                if collection_count == 0:
                    need_reload = True
                    logger.warning(f"[__setstate__] ChromaDB collection is empty (count=0), will reload all {memory_count} memories")
                elif collection_count < memory_count:
                    # ChromaDB has fewer records than Memory - need to reload to add missing ones
                    need_reload = True
                    logger.warning(f"[__setstate__] Count mismatch: ChromaDB={collection_count} < Memory={memory_count}, will reload")
                elif collection_count > memory_count:
                    # ChromaDB has more records than Memory - use Memory's records, don't reload
                    # This allows using a subset of ChromaDB data without modifying local storage
                    # Verify that Memory's IDs exist in ChromaDB
                    logger.info(f"[__setstate__] ChromaDB has more records ({collection_count}) than Memory ({memory_count}), verifying Memory IDs exist in ChromaDB...")
                    sample_ids = list(self.memories.keys())[:min(10, len(self.memories))]
                    existing_ids = set(self.retriever.collection.get(ids=sample_ids)['ids'])
                    if all(mid in existing_ids for mid in sample_ids):
                        need_reload = False
                        logger.info(f"[__setstate__] ✓ Using Memory's {memory_count} records from ChromaDB ({collection_count} total), no reload needed")
                    else:
                        # Some Memory IDs don't exist in ChromaDB, need to reload
                        need_reload = True
                        missing_ids = set(sample_ids) - existing_ids
                        logger.warning(f"[__setstate__] Some Memory IDs missing in ChromaDB: {list(missing_ids)[:5]}, will reload")
                elif collection_count != memory_count:
                    # Should not reach here, but keep as fallback
                    need_reload = True
                    logger.warning(f"[__setstate__] Count mismatch: ChromaDB={collection_count} != Memory={memory_count}, will reload")
                else:
                    # Verify a few IDs match to ensure data integrity
                    logger.info(f"[__setstate__] Step 3b: Verifying ID integrity (counts match)...")
                    _verify_start = time.time()
                    sample_ids = list(self.memories.keys())[:min(10, len(self.memories))]
                    existing_ids = set(self.retriever.collection.get(ids=sample_ids)['ids'])
                    _verify_time = time.time() - _verify_start
                    logger.info(f"[__setstate__] Step 3b: Verified {len(existing_ids)}/{len(sample_ids)} sample IDs (took {_verify_time:.2f}s)")
                    
                    if not all(mid in existing_ids for mid in sample_ids):
                        need_reload = True
                        missing_ids = set(sample_ids) - existing_ids
                        logger.warning(f"[__setstate__] Data integrity check failed: missing IDs {list(missing_ids)[:5]}, will reload")
                    else:
                        logger.info(f"[__setstate__] ✓ ChromaDB persistent storage OK ({collection_count} memories), skipping reload")
            except Exception as e:
                logger.error(f"[__setstate__] Error checking ChromaDB state: {e}, will reload memories")
                import traceback
                logger.error(traceback.format_exc())
                need_reload = True
        else:
            # Using in-memory mode - always need to reload
            need_reload = True
            logger.warning(f"[__setstate__] In-memory mode (chroma_db_path=None), will reload all {len(self.memories)} memories")
        
        # Step 4: Re-add memories if necessary
        if need_reload:
            _reload_start = time.time()
            memory_count = len(self.memories)
            logger.info(f"[__setstate__] Step 4: Reloading {memory_count} memories into ChromaDB...")
            _added = 0
            for memory in self.memories.values():
                # Optimized: Only store essential fields for retrieval
                metadata = {
                    "id": memory.id,
                    "content": memory.content,
                    "keywords": memory.keywords,
                    "links": memory.links,
                    "timestamp": memory.timestamp,
                    "context": memory.context,
                    "category": memory.category,
                    "tags": memory.tags
                }
                try:
                    self.retriever.add_document(memory.content, metadata, memory.id)
                    _added += 1
                    if _added % 100 == 0:
                        _elapsed = time.time() - _reload_start
                        _rate = _added / _elapsed if _elapsed > 0 else 0
                        logger.info(f"[__setstate__] Added {_added}/{memory_count} memories ({_rate:.1f} mem/s)...")
                except Exception as e:
                    logger.warning(f"Could not re-add memory {memory.id} to ChromaDB: {e}")
            _reload_time = time.time() - _reload_start
            logger.info(f"[__setstate__] Step 4 complete: {_added} memories reloaded in {_reload_time:.2f}s")
        else:
            logger.info(f"[__setstate__] Step 4: Skipped (using persistent storage)")
        
        # Step 5: Reconstruct llm_controller
        _step_start = time.time()
        llm_backend = state.get('_llm_backend', 'openai')
        llm_model = state.get('_llm_model', 'gpt-4o-mini')
        api_key = state.get('_api_key', None)
        api_base = state.get('_api_base', None)
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, api_base)
        _step_time = time.time() - _step_start
        if _step_time > 0.1:
            logger.info(f"[__setstate__] Step 5 (init LLM controller): {_step_time:.2f}s")
        
        _total_time = time.time() - _perf_start
        if _total_time > 1.0:
            logger.info(f"[__setstate__] Total time: {_total_time:.2f}s")
