from enum import StrEnum
from pydantic import BaseModel, UUID4


class Document(BaseModel):
    doc_id: UUID4
    name: str
    uri: str


class DocumentChunk(BaseModel):
    doc_id: UUID4
    text: str


class EmbeddedDocumentChunk(DocumentChunk):
    dense_vector: list[float]  # for semantic search
    sparse_vector: dict[int, float]  # for BM25


class Role(StrEnum):
    agent = "agent"
    user = "user"


class Message(BaseModel):
    role: Role
    text: str
