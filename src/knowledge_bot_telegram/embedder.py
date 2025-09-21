from fastembed import SparseTextEmbedding, TextEmbedding

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
        self.dense_embedder = TextEmbedding(
            model_name="deepvk/RuModernBERT-base",
            device="cpu",
        )
        self.bm25_embedder = SparseTextEmbedding("Qdrant/bm25", language="russian")

    def embed_request(self, request: str) -> EmbeddedRequest:
        pass

    def embed_document(self, document: Document) -> list[EmbeddedDocumentChunk]:
        pass

    def split_document(self, document: Document) -> list[DocumentChunk]:
        pass
