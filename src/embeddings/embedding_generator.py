import logging

import os

from typing import Callable, List, Optional



import numpy as np

from dataclasses import dataclass



from fastembed import TextEmbedding

from src.document_processing.document_chunk import DocumentChunk

from src.embeddings.embed_cache import EmbedCache



logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)



DEFAULT_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))





@dataclass

class EmbeddedChunk:

    """Document chunk with its embedding vector"""

    chunk: DocumentChunk

    embedding: np.ndarray

    embedding_model: str

    

    def to_vector_db_format(self) -> dict:

        return {

            'id': self.chunk.chunk_id,

            'vector': self.embedding.tolist(),

            'content': self.chunk.content,

            'source_file': self.chunk.source_file,

            'source_type': self.chunk.source_type,

            'page_number': self.chunk.page_number,

            'chunk_index': self.chunk.chunk_index,

            'start_char': self.chunk.start_char,

            'end_char': self.chunk.end_char,

            'metadata': self.chunk.metadata,

            'embedding_model': self.embedding_model

        }





class EmbeddingGenerator:

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: Optional[int] = None):

        self.model_name = model_name

        self.batch_size = batch_size or DEFAULT_BATCH_SIZE

        self.model = None

        self.embedding_dim = None

        self._cache = EmbedCache()

        self._initialize_model()

    

    def _initialize_model(self):

        try:

            logger.info(f"Initializing embedding model: {self.model_name} (batch={self.batch_size})")

            self.model = TextEmbedding(model_name=self.model_name)

            

            sample_embedding = list(self.model.embed(["test"]))[0]

            self.embedding_dim = len(sample_embedding)

            

            logger.info(f"Model initialized successfully. Embedding dimension: {self.embedding_dim}")

            

        except Exception as e:

            logger.error(f"Failed to initialize embedding model: {str(e)}")

            raise

    

    def generate_embeddings(

        self,

        chunks: List[DocumentChunk],

        batch_size: Optional[int] = None,

        on_batch: Optional[Callable[[int, int], None]] = None,

    ) -> List[EmbeddedChunk]:

        if not chunks:

            return []



        bs = batch_size or self.batch_size

        total = len(chunks)

        logger.info(f"Generating embeddings for {total} chunks (batch_size={bs})")



        try:

            embedded_chunks: List[EmbeddedChunk] = []

            for i in range(0, total, bs):

                sub_batch = chunks[i:i + bs]

                texts = [chunk.content for chunk in sub_batch]

                cached, miss_indices = self._cache.get_many(texts, self.model_name)



                to_embed_texts: List[str] = []

                to_embed_indices: List[int] = []

                for j, vec in enumerate(cached):

                    if vec is None:

                        to_embed_indices.append(j)

                        to_embed_texts.append(texts[j])



                fresh_vectors: List[np.ndarray] = []

                if to_embed_texts:

                    fresh_vectors = [

                        np.array(v, dtype=np.float32) for v in self.model.embed(to_embed_texts)

                    ]

                    self._cache.put_many(to_embed_texts, self.model_name, fresh_vectors)



                miss_map = {idx: fresh_vectors[k] for k, idx in enumerate(to_embed_indices)}

                for j, chunk in enumerate(sub_batch):

                    embedding = cached[j] if cached[j] is not None else miss_map[j]

                    embedded_chunks.append(

                        EmbeddedChunk(

                            chunk=chunk,

                            embedding=embedding,

                            embedding_model=self.model_name,

                        )

                    )



                done = min(i + len(sub_batch), total)

                if on_batch:

                    on_batch(done, total)

                logger.info(f"Embedded batch {i // bs + 1}/{(total + bs - 1) // bs}")



            logger.info(f"Successfully generated {len(embedded_chunks)} embeddings")

            return embedded_chunks



        except Exception as e:

            logger.error(f"Error generating embeddings: {str(e)}")

            raise

    

    def generate_query_embedding(self, query_text: str) -> np.ndarray:

        try:

            cached, miss = self._cache.get_many([query_text], self.model_name)

            if cached[0] is not None:

                return cached[0]

            embedding = list(self.model.embed([query_text]))[0]

            vec = np.array(embedding, dtype=np.float32)

            self._cache.put_many([query_text], self.model_name, [vec])

            return vec

            

        except Exception as e:

            logger.error(f"Error generating query embedding: {str(e)}")

            raise

    

    def get_embedding_dimension(self) -> int:

        return self.embedding_dim


