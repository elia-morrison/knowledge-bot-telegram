from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from knowledge_bot_telegram.schemas import (
    Document,
    DocumentChunk,
    EmbeddedDocumentChunk,
    EmbeddedRequest,
)


# TODO: this belongs to the cloud.
# Need to serve embedder as a service, but not for this demo.
class Embedder:
    def __init__(self):
        # fastembed does not support USER2-base
        self.embedding_dimension = 768
        self.dense_embedder = SentenceTransformer(
            "deepvk/USER2-base", device="cpu", truncate_dim=self.embedding_dimension
        )
        self.bm25_embedder = SparseTextEmbedding("Qdrant/bm25", language="russian")
        self.tokenizer = AutoTokenizer.from_pretrained("deepvk/USER2-base")
        self.source_prefix = "Источник: "
        self.chunk_prefix = " \n Отрывок: "

        self.token_overhead = len(self.tokenizer.encode(self.source_prefix)) + len(
            self.tokenizer.encode(self.chunk_prefix)
        )
        # we have a HUGE context window (8k),
        # but we don't want to use it all,
        # because shorter docs have a more concentrated meaning
        self.max_tokens_per_chunk = 512 - self.token_overhead
        self.overlap_tokens = 50

    def sparse_to_dict(self, sparse_vector: SparseEmbedding) -> dict[int, float]:
        return {i: v for i, v in zip(sparse_vector.indices, sparse_vector.values)}

    def embed_request(self, request: str) -> EmbeddedRequest:
        dense_vector = self.dense_embedder.encode(
            [request], prompt_name="search_query"
        )[0].tolist()
        bm25_vector = next(self.bm25_embedder.query_embed(request))
        return EmbeddedRequest(
            query=request,
            dense_vector=dense_vector,
            bm25_vector=self.sparse_to_dict(bm25_vector),
        )

    def embed_document(self, document: Document) -> list[EmbeddedDocumentChunk]:
        chunks = self.split_document(document)
        text = [chunk.text for chunk in chunks]
        dense_vectors = list(
            self.dense_embedder.encode(text, prompt_name="search_document")
        )
        bm25_vectors = list(
            self.sparse_to_dict(x) for x in self.bm25_embedder.embed(text)
        )
        return [
            EmbeddedDocumentChunk(
                doc_id=document.doc_id,
                text=chunk.text,
                dense_vector=dense_vector,
                bm25_vector=bm25_vector,
            )
            for chunk, dense_vector, bm25_vector in zip(
                chunks, dense_vectors, bm25_vectors, strict=True
            )
        ]

    def split_document(self, document: Document) -> list[DocumentChunk]:
        token_chunks = self._split_text_to_token_chunks(
            text=document.full_text, name=document.name
        )
        return [
            DocumentChunk(
                doc_id=document.doc_id,
                text=f"{self.source_prefix}{document.name}{self.chunk_prefix}{chunk_text}",
            )
            for chunk_text in token_chunks
        ]

    # TODO: complex chunking code does not belong here.
    # LangChain already has this, but we're forbidden to use it :\
    def _split_text_to_token_chunks(self, *, text: str, name: str) -> list[str]:
        if not text:
            return []

        tokens = self.tokenizer.encode(text)
        if not tokens:
            return []

        name_token_overhead = len(self.tokenizer.encode(name))
        max_tokens_per_chunk = self.max_tokens_per_chunk - name_token_overhead

        chunks: list[str] = []
        step = max_tokens_per_chunk - self.overlap_tokens
        for start in range(0, len(tokens), step):
            end = start + max_tokens_per_chunk
            token_slice = tokens[start:end]
            if not token_slice:
                continue
            chunk_text = self.tokenizer.decode(token_slice).strip()
            if chunk_text:
                chunks.append(chunk_text)
        return chunks
