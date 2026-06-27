import logging
import time
import uuid
import hashlib
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from pymilvus import MilvusClient, DataType
from src.embeddings.embedding_generator import EmbeddedChunk
from src.vector_database.chunk_reference_tracker import get_reference_tracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 2  # Version 2: reference counting, no notebook_id field


def _get_schema_version_db_path(db_path: str) -> Path:
    """Get path for schema version tracking database."""
    return Path(db_path).parent / "schema_version.db"


def _get_current_schema_version(db_path: str) -> int:
    """Get current schema version from tracking database."""
    version_db = _get_schema_version_db_path(db_path)
    if not version_db.exists():
        return 0  # No version tracking = old version
    
    try:
        with sqlite3.connect(version_db) as conn:
            cursor = conn.execute("SELECT version FROM schema_version WHERE id = 1")
            row = cursor.fetchone()
            return row[0] if row else 0
    except Exception:
        return 0


def _set_schema_version(db_path: str, version: int) -> bool:
    """Set schema version in tracking database."""
    version_db = _get_schema_version_db_path(db_path)
    version_db.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with sqlite3.connect(version_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    id INTEGER PRIMARY KEY,
                    version INTEGER NOT NULL,
                    migrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                INSERT OR REPLACE INTO schema_version (id, version)
                VALUES (1, ?)
            """, (version,))
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to set schema version: {e}")
        return False


def _escape_filter_string(value: str) -> str:
    """Escape a string for Milvus filter expressions."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


class MilvusVectorDB:
    def __init__(
        self, 
        db_path: str = "./milvus_lite.db",
        collection_name: str = "notebook_lm",
        embedding_dim: int = 384
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.client = None
        self.collection_exists = False
        
        self._initialize_client()
        self._setup_collection()
    
    def _migrate_collection(self):
        """Migrate from old schema (notebook_id field) to new schema (reference counting)."""
        try:
            logger.info("Starting schema migration from version 1 to version 2...")
            ref_tracker = get_reference_tracker()
            
            # Export all existing chunks with their data
            results = self.client.query(
                collection_name=self.collection_name,
                output_fields=["id", "notebook_id", "source_file", "content", "vector", 
                               "source_type", "page_number", "chunk_index", "start_char", 
                               "end_char", "metadata", "embedding_model"],
                limit=10000
            )
            
            # Calculate content hashes for existing chunks
            chunks_to_migrate = []
            for chunk in results:
                chunk_id = chunk.get('id', '')
                notebook_id = chunk.get('notebook_id')
                source_file = chunk.get('source_file', 'unknown')
                content = chunk.get('content', '')
                
                if chunk_id and notebook_id is not None:
                    # Calculate content hash
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    
                    # Add reference to tracker
                    ref_tracker.add_reference(chunk_id, notebook_id, source_file)
                    
                    # Prepare chunk data for re-insertion
                    chunk_data = {
                        'id': chunk_id,
                        'vector': chunk.get('vector'),
                        'content': content,
                        'source_file': source_file,
                        'source_type': chunk.get('source_type', ''),
                        'page_number': chunk.get('page_number', -1),
                        'chunk_index': chunk.get('chunk_index', 0),
                        'start_char': chunk.get('start_char', -1),
                        'end_char': chunk.get('end_char', -1),
                        'metadata': chunk.get('metadata', {}),
                        'embedding_model': chunk.get('embedding_model', ''),
                        'content_hash': content_hash
                    }
                    chunks_to_migrate.append(chunk_data)
            
            logger.info(f"Prepared {len(chunks_to_migrate)} chunks for migration")
            
            # Drop and recreate collection without notebook_id field
            logger.info("Recreating collection with new schema...")
            self.client.drop_collection(collection_name=self.collection_name)
            self.collection_exists = False
            
            # Create new collection
            self._create_new_collection()
            
            # Re-insert all chunks with new schema
            if chunks_to_migrate:
                self.client.insert(
                    collection_name=self.collection_name,
                    data=chunks_to_migrate
                )
                logger.info(f"Re-inserted {len(chunks_to_migrate)} chunks into new collection")
            
            # Update schema version
            _set_schema_version(self.db_path, CURRENT_SCHEMA_VERSION)
            logger.info("Schema migration completed successfully")
            
        except Exception as e:
            logger.error(f"Schema migration failed: {e}")
            raise
    
    def _create_new_collection(self):
        """Create collection with new schema (no notebook_id field)."""
        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True
        )
        
        # Primary key field (chunk_id)
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            max_length=128,
            is_primary=True
        )
        
        # Vector field for embeddings
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.embedding_dim
        )
        
        # Essential fields for citations and content
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=8192
        )
        
        schema.add_field(
            field_name="source_file",
            datatype=DataType.VARCHAR,
            max_length=512
        )
        
        schema.add_field(
            field_name="source_type",
            datatype=DataType.VARCHAR,
            max_length=32
        )
        
        schema.add_field(
            field_name="page_number",
            datatype=DataType.INT32
        )
        
        schema.add_field(
            field_name="chunk_index",
            datatype=DataType.INT32
        )
        
        schema.add_field(
            field_name="start_char",
            datatype=DataType.INT32
        )
        
        schema.add_field(
            field_name="end_char", 
            datatype=DataType.INT32
        )
        
        # JSON field for additional metadata
        schema.add_field(
            field_name="metadata",
            datatype=DataType.JSON
        )
        
        schema.add_field(
            field_name="embedding_model",
            datatype=DataType.VARCHAR,
            max_length=128
        )
        
        # Content hash for deduplication
        schema.add_field(
            field_name="content_hash",
            datatype=DataType.VARCHAR,
            max_length=32
        )
        
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema
        )
        
        logger.info(f"Collection '{self.collection_name}' created with new schema")
        self.collection_exists = True
    
    def _initialize_client(self):
        try:
            self.client = MilvusClient(uri=self.db_path)
            logger.info(f"Milvus client initialized with database: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Milvus client: {str(e)}")
            raise
    
    def _setup_collection(self):
        try:
            current_version = _get_current_schema_version(self.db_path)
            
            if self.client.has_collection(collection_name=self.collection_name):
                logger.info(f"Collection '{self.collection_name}' already exists")
                self.collection_exists = True
                
                # Check if migration is needed
                if current_version < CURRENT_SCHEMA_VERSION:
                    logger.warning(f"Schema version {current_version} detected, current version is {CURRENT_SCHEMA_VERSION}")
                    self._migrate_collection()
                else:
                    logger.info(f"Schema version {current_version} is up to date")
                
                try:
                    self.client.load_collection(collection_name=self.collection_name)
                    logger.info(f"Collection '{self.collection_name}' loaded into memory")
                except Exception:
                    pass  # May not have index yet
                return
            
            # Create new collection
            self._create_new_collection()
            _set_schema_version(self.db_path, CURRENT_SCHEMA_VERSION)
            
        except Exception as e:
            logger.error(f"Error setting up collection: {str(e)}")
            raise
    
    def create_index(
        self,
        use_binary_quantization: bool = False,
        nlist: int = 1024,
        enable_refine: bool = False,
        refine_type: str = "SQ8"
    ):
        try:
            if not self.collection_exists:
                raise Exception("Collection does not exist. Setup collection first.")
            
            index_params = self.client.prepare_index_params()
            
            if use_binary_quantization:
                index_params.add_index(
                    field_name="vector",
                    index_type="IVF_RABITQ",
                    index_name="vector_index",
                    metric_type="L2",
                    params={
                        "nlist": nlist,
                        "refine": enable_refine,
                        "refine_type": refine_type if enable_refine else None
                    }
                )
            else:
                index_params.add_index(
                    field_name="vector",
                    index_type="IVF_FLAT", 
                    index_name="vector_index",
                    metric_type="L2",
                )
            
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params
            )
            logger.info("Index created successfully")
            
        except Exception as e:
            if "already exists" in str(e):
                logger.info("Index already exists, skipping creation")
            else:
                logger.error(f"Error creating index: {str(e)}")
                raise
        
        # Always load collection into memory for searching
        try:
            self.client.load_collection(collection_name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' loaded into memory")
        except Exception as e:
            logger.warning(f"Could not load collection: {e}")
    
    def insert_embeddings(self, embedded_chunks: List[EmbeddedChunk], notebook_id: int = 1) -> List[str]:
        if not embedded_chunks:
            return []
        
        # Retry logic for Windows file locking issues
        max_retries = 3
        retry_delay = 1
        ref_tracker = get_reference_tracker()
        
        for attempt in range(max_retries):
            try:
                # Calculate content hashes and prepare data
                data = []
                content_hashes = {}
                source_file = None
                
                for embedded_chunk in embedded_chunks:
                    chunk_data = embedded_chunk.to_vector_db_format()
                    
                    # Calculate MD5 hash of content for deduplication
                    content_hash = hashlib.md5(chunk_data['content'].encode()).hexdigest()
                    chunk_data['content_hash'] = content_hash
                    content_hashes[content_hash] = chunk_data
                    
                    if source_file is None:
                        source_file = chunk_data.get('source_file', 'unknown')
                    
                    # Use content hash as base ID (stable across notebooks)
                    chunk_data['id'] = f"chunk_{content_hash}"
                    chunk_data['page_number'] = chunk_data['page_number'] or -1
                    chunk_data['start_char'] = chunk_data['start_char'] or -1
                    chunk_data['end_char'] = chunk_data['end_char'] or -1
                    # Don't set notebook_id here - chunks are shared across notebooks

                    data.append(chunk_data)
                
                # Check for existing chunks by content hash
                if content_hashes:
                    hash_filter = " or ".join([f'content_hash == "{h}"' for h in content_hashes.keys()])
                    
                    try:
                        existing_chunks = self.client.query(
                            collection_name=self.collection_name,
                            filter=hash_filter,
                            output_fields=["content_hash", "vector", "id", "source_file"],
                            limit=len(content_hashes)
                        )
                        
                        existing_by_hash = {chunk['content_hash']: chunk for chunk in existing_chunks}
                        
                        # Separate new chunks from existing ones
                        new_chunks = []
                        reused_chunks = []
                        chunk_ids_to_reference = []
                        
                        for content_hash, chunk_data in content_hashes.items():
                            if content_hash in existing_by_hash:
                                # Chunk exists, reuse it
                                existing = existing_by_hash[content_hash]
                                chunk_ids_to_reference.append(existing['id'])
                                reused_chunks.append(content_hash)
                                logger.info(f"Reusing existing chunk {existing['id']} for notebook {notebook_id}")
                            else:
                                # New chunk, need to insert
                                chunk_data['vector'] = None  # Will be filled by embedding generator
                                new_chunks.append(chunk_data)
                        
                        # Add references for reused chunks
                        if chunk_ids_to_reference:
                            ref_tracker.add_references_batch(chunk_ids_to_reference, notebook_id, source_file)
                        
                        if reused_chunks:
                            logger.info(f"Reused {len(reused_chunks)} existing chunks for notebook {notebook_id}")
                        
                        # Insert only new chunks
                        if new_chunks:
                            self.client.insert(
                                collection_name=self.collection_name,
                                data=new_chunks
                            )
                            
                            # Add references for new chunks
                            new_chunk_ids = [chunk['id'] for chunk in new_chunks]
                            ref_tracker.add_references_batch(new_chunk_ids, notebook_id, source_file)
                            
                            logger.info(f"Inserted {len(new_chunks)} new chunks for notebook {notebook_id}")
                            return new_chunk_ids
                        else:
                            logger.info(f"All chunks already exist, only added references for notebook {notebook_id}")
                            return chunk_ids_to_reference
                            
                    except Exception as e:
                        logger.warning(f"Could not check for existing chunks (deduplication disabled): {e}")
                        # Fallback to normal insert if deduplication check fails
                        self.client.insert(
                            collection_name=self.collection_name,
                            data=data
                        )
                        chunk_ids = [chunk['id'] for chunk in data]
                        ref_tracker.add_references_batch(chunk_ids, notebook_id, source_file)
                        return chunk_ids

                # Fallback for no content hashes
                self.client.insert(
                    collection_name=self.collection_name,
                    data=data
                )
                inserted_ids = [item['id'] for item in data]
                ref_tracker.add_references_batch(inserted_ids, notebook_id, source_file)
                logger.info(f"Inserted {len(inserted_ids)} embeddings into database (notebook {notebook_id})")
                return inserted_ids

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Insert attempt {attempt + 1} failed, retrying in {retry_delay}s: {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Error inserting embeddings after {max_retries} attempts: {str(e)}")
                    raise
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        nprobe: int = 24,
        rbq_query_bits: int = 0,
        refine_k: float = 1.0,
        filter_expr: Optional[str] = None,
        notebook_id: Optional[int] = None,
        use_binary_quantization: bool = False
    ) -> List[Dict[str, Any]]:
        try:
            if use_binary_quantization:
                search_params = {
                    "params": {
                        "nprobe": nprobe,
                        "rbq_query_bits": rbq_query_bits,
                        "refine_k": refine_k
                    }
                }
            else:
                search_params = {
                    "params": {
                        "nprobe": nprobe
                    }
                }

            # If notebook_id is specified, get allowed chunk IDs from reference tracker
            final_filter = filter_expr
            if notebook_id is not None:
                ref_tracker = get_reference_tracker()
                allowed_chunk_ids = ref_tracker.get_chunks_for_notebook(notebook_id)
                
                if allowed_chunk_ids:
                    # Build filter for allowed chunks
                    chunk_filter = " or ".join([f'id == "{_escape_filter_string(cid)}"' for cid in allowed_chunk_ids])
                    if final_filter:
                        final_filter = f"({final_filter}) and ({chunk_filter})"
                    else:
                        final_filter = chunk_filter
                else:
                    # No chunks for this notebook, return empty results
                    logger.info(f"No chunks found for notebook {notebook_id}")
                    return []

            # Perform vector similarity search
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                anns_field="vector",
                limit=limit,
                search_params=search_params,
                filter=final_filter,
                output_fields=[
                    "content", "source_file", "source_type", "page_number",
                    "chunk_index", "start_char", "end_char", "metadata", "embedding_model",
                    "content_hash"
                ]
            )

            formatted_results = []
            if results and len(results) > 0:
                for result in results[0]:
                    formatted_result = {
                        'id': result['id'],
                        'score': result['distance'],
                        'content': result['entity']['content'],
                        'citation': {
                            'source_file': result['entity']['source_file'],
                            'source_type': result['entity']['source_type'],
                            'page_number': result['entity']['page_number'] if result['entity']['page_number'] != -1 else None,
                            'chunk_index': result['entity']['chunk_index'],
                            'start_char': result['entity']['start_char'] if result['entity']['start_char'] != -1 else None,
                            'end_char': result['entity']['end_char'] if result['entity']['end_char'] != -1 else None,
                        },
                        'metadata': result['entity']['metadata'],
                        'embedding_model': result['entity']['embedding_model']
                    }
                    formatted_results.append(formatted_result)

            logger.info(f"Search completed: {len(formatted_results)} results found")
            return formatted_results

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

    def query_by_source(
        self,
        source_file: str,
        notebook_id: Optional[int] = None,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        """Get all chunks for a source using reference tracker."""
        try:
            ref_tracker = get_reference_tracker()
            
            if notebook_id is not None:
                # Get chunks for this specific notebook
                chunk_ids = ref_tracker.get_chunks_for_source(source_file, notebook_id)
            else:
                # Get all chunks for this source across all notebooks
                chunk_ids = ref_tracker.get_chunks_for_source(source_file)
            
            if not chunk_ids:
                logger.info(f"No chunks found for source '{source_file}'")
                return []
            
            # Build filter expression for chunk IDs
            chunk_filter = " or ".join([f'id == "{_escape_filter_string(cid)}"' for cid in chunk_ids])
            
            results = self.client.query(
                collection_name=self.collection_name,
                filter=chunk_filter,
                output_fields=[
                    "content", "source_file", "source_type", "page_number",
                    "chunk_index", "start_char", "end_char", "metadata", "embedding_model",
                    "content_hash"
                ],
                limit=limit,
            )

            formatted = []
            for r in results:
                formatted.append({
                    'id': r.get('id', ''),
                    'score': 0,
                    'content': r.get('content', ''),
                    'citation': {
                        'source_file': r.get('source_file', ''),
                        'source_type': r.get('source_type', ''),
                        'page_number': r.get('page_number') if r.get('page_number', -1) != -1 else None,
                        'chunk_index': r.get('chunk_index', 0),
                        'start_char': r.get('start_char') if r.get('start_char', -1) != -1 else None,
                        'end_char': r.get('end_char') if r.get('end_char', -1) != -1 else None,
                    },
                    'metadata': r.get('metadata', {}),
                    'embedding_model': r.get('embedding_model', ''),
                })

            logger.info(f"Query by source '{source_file}': {len(formatted)} results")
            return formatted

        except Exception as e:
            logger.error(f"Error querying by source: {e}")
            return []

    def query_by_notebook(
        self,
        notebook_id: int,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """Get all chunks for a notebook using reference tracker."""
        try:
            ref_tracker = get_reference_tracker()
            chunk_ids = ref_tracker.get_chunks_for_notebook(notebook_id)
            
            if not chunk_ids:
                logger.info(f"No chunks found for notebook {notebook_id}")
                return []
            
            # Build filter expression for chunk IDs
            chunk_filter = " or ".join([f'id == "{_escape_filter_string(cid)}"' for cid in chunk_ids])
            
            results = self.client.query(
                collection_name=self.collection_name,
                filter=chunk_filter,
                output_fields=[
                    "id", "content", "source_file", "source_type", "page_number",
                    "chunk_index", "start_char", "end_char", "metadata", "embedding_model",
                    "content_hash",
                ],
                limit=limit,
            )

            formatted = []
            for r in results:
                formatted.append({
                    "id": r.get("id", ""),
                    "score": 0,
                    "content": r.get("content", ""),
                    "citation": {
                        "source_file": r.get("source_file", ""),
                        "source_type": r.get("source_type", ""),
                        "page_number": r.get("page_number") if r.get("page_number", -1) != -1 else None,
                        "chunk_index": r.get("chunk_index", 0),
                        "start_char": r.get("start_char") if r.get("start_char", -1) != -1 else None,
                        "end_char": r.get("end_char") if r.get("end_char", -1) != -1 else None,
                    },
                    "metadata": r.get("metadata", {}),
                    "embedding_model": r.get("embedding_model", ""),
                })

            logger.info(f"Query by notebook {notebook_id}: {len(formatted)} chunks")
            return formatted

        except Exception as e:
            logger.error(f"Error querying by notebook: {e}")
            return []
    
    def delete_collection(self):
        try:
            if self.client.has_collection(collection_name=self.collection_name):
                self.client.drop_collection(collection_name=self.collection_name)
                logger.info(f"Collection '{self.collection_name}' deleted")
                self.collection_exists = False
            else:
                logger.info(f"Collection '{self.collection_name}' does not exist")
                
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise
    
    def delete_by_source(self, source_file: str, notebook_id: Optional[int] = None) -> int:
        """Delete all vectors for a given source file with reference counting."""
        ref_tracker = get_reference_tracker()
        
        try:
            # Get chunk IDs for this source
            chunk_ids = ref_tracker.get_chunks_for_source(source_file, notebook_id)
            
            if not chunk_ids:
                logger.info(f"No chunks found for source {source_file}")
                return 0
            
            # Remove notebook references
            ref_tracker.delete_by_source(source_file, notebook_id)
            
            # Check which chunks are now orphaned (no notebook references)
            orphaned_chunks = set()
            for chunk_id in chunk_ids:
                if ref_tracker.get_reference_count(chunk_id) == 0:
                    orphaned_chunks.add(chunk_id)
            
            # Delete only orphaned chunks from Milvus
            if orphaned_chunks:
                # Build filter expression for orphaned chunks
                chunk_filter = " or ".join([f'id == "{_escape_filter_string(cid)}"' for cid in orphaned_chunks])
                self.client.delete(
                    collection_name=self.collection_name,
                    filter=chunk_filter
                )
                logger.info(f"Deleted {len(orphaned_chunks)} orphaned chunks from Milvus")
            else:
                logger.info(f"No orphaned chunks to delete (all chunks still referenced)")
            
            return len(chunk_ids)
            
        except Exception as e:
            logger.error(f"Error deleting vectors for source {source_file}: {e}")
            return 0

    def delete_by_notebook(self, notebook_id: int) -> int:
        """Delete all vectors for a given notebook with reference counting."""
        ref_tracker = get_reference_tracker()
        
        try:
            # Get all chunk IDs for this notebook
            chunk_ids = ref_tracker.get_chunks_for_notebook(notebook_id)
            
            if not chunk_ids:
                logger.info(f"No chunks found for notebook {notebook_id}")
                return 0
            
            # Remove all notebook references
            ref_tracker.delete_by_notebook(notebook_id)
            
            # Check which chunks are now orphaned
            orphaned_chunks = set()
            for chunk_id in chunk_ids:
                if ref_tracker.get_reference_count(chunk_id) == 0:
                    orphaned_chunks.add(chunk_id)
            
            # Delete only orphaned chunks from Milvus
            if orphaned_chunks:
                chunk_filter = " or ".join([f'id == "{_escape_filter_string(cid)}"' for cid in orphaned_chunks])
                self.client.delete(
                    collection_name=self.collection_name,
                    filter=chunk_filter
                )
                logger.info(f"Deleted {len(orphaned_chunks)} orphaned chunks from Milvus for notebook {notebook_id}")
            else:
                logger.info(f"No orphaned chunks to delete for notebook {notebook_id} (all chunks still referenced)")
            
            return len(chunk_ids)
            
        except Exception as e:
            logger.error(f"Error deleting vectors for notebook {notebook_id}: {e}")
            return 0

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        try:
            if not self.collection_exists:
                logger.warning("Collection does not exist")
                return None
            
            logger.info(f"Attempting to retrieve chunk with ID: {chunk_id}")
            
            safe_id = _escape_filter_string(chunk_id)
            results = self.client.query(
                collection_name=self.collection_name,
                filter=f'id == "{safe_id}"',
                output_fields=["id", "content", "metadata", "source_file", "source_type", "page_number", "chunk_index", "content_hash"]
            )
            
            logger.info(f"Query returned {len(results) if results else 0} results")
            
            if results and len(results) > 0:
                chunk_data = results[0]
                logger.info(f"Successfully retrieved chunk: {chunk_data.get('id')}")
                
                metadata = chunk_data.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except Exception:
                        metadata = {}
                
                return {
                    "id": chunk_data.get("id"),
                    "content": chunk_data.get("content"),
                    "metadata": metadata,
                    "source_file": chunk_data.get("source_file"),
                    "source_type": chunk_data.get("source_type"),
                    "page_number": chunk_data.get("page_number"),
                    "chunk_index": chunk_data.get("chunk_index")
                }
            
            logger.warning(f"No chunk found with ID: {chunk_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving chunk by ID {chunk_id}: {str(e)}")
            logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
            return None
    
    def close(self):
        try:
            if self.client:
                self.client.close()
                logger.info("Milvus client connection closed")
        except Exception as e:
            logger.error(f"Error closing connection: {str(e)}")


if __name__ == "__main__":
    from src.document_processing.doc_processor import DocumentProcessor
    from src.embeddings.embedding_generator import EmbeddingGenerator
    
    doc_processor = DocumentProcessor()
    embedding_generator = EmbeddingGenerator()
    vector_db = MilvusVectorDB()
    
    try:
        chunks = doc_processor.process_document("data/raft.pdf")
        embedded_chunks = embedding_generator.generate_embeddings(chunks)
        vector_db.create_index()
        
        inserted_ids = vector_db.insert_embeddings(embedded_chunks)
        print(f"Inserted {len(inserted_ids)} embeddings")
        
        query_text = "What is the main topic?"
        query_vector = embedding_generator.generate_query_embedding(query_text)
        
        search_results = vector_db.search(query_vector.tolist(), limit=5)
        
        for i, result in enumerate(search_results):
            print(f"\nResult {i+1}:")
            print(f"Score: {result['score']:.4f}")
            print(f"Content: {result['content'][:200]}...")
            print(f"Citation: {result['citation']}")
        
    except Exception as e:
        print(f"Error in example: {e}")
    
    finally:
        vector_db.close()