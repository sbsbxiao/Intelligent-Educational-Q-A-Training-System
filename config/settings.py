"""Application settings loaded from environment variables or .env."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM provider. Put real values in .env; do not hard-code secrets here.
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="", validation_alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="", validation_alias="OPENAI_MODEL")

    # Embedding
    embedding_model: str = Field(default="", validation_alias="EMBEDDING_MODEL")
    embedding_provider: str = Field(default="local", validation_alias="EMBEDDING_PROVIDER")
    local_embedding_path: str = Field(default="", validation_alias="LOCAL_EMBEDDING_PATH")

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687", validation_alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", validation_alias="NEO4J_USER")
    neo4j_password: str = Field(default="", validation_alias="NEO4J_PASSWORD")

    # Vector Store
    vector_store_type: str = Field(default="chroma", validation_alias="VECTOR_STORE_TYPE")
    vector_collection_name: str = Field(default="knowledge_chunks", validation_alias="VECTOR_COLLECTION_NAME")
    chroma_host: str = Field(default="localhost", validation_alias="CHROMA_HOST")
    chroma_port: int = Field(default=8000, validation_alias="CHROMA_PORT")
    pgvector_dsn: str = Field(default="", validation_alias="PGVECTOR_DSN")

    # Kafka (CDC)
    kafka_bootstrap_servers: str = Field(default="localhost:9092", validation_alias="KAFKA_BOOTSTRAP_SERVERS")
    kafka_topic_doc_changes: str = Field(default="doc-changes", validation_alias="KAFKA_TOPIC_DOC_CHANGES")
    kafka_topic_kg_updates: str = Field(default="kg-updates", validation_alias="KAFKA_TOPIC_KG_UPDATES")

    # API
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8080, validation_alias="API_PORT")

    # Document Store
    upload_dir: str = Field(default="./uploads", validation_alias="UPLOAD_DIR")

    # Conversation Memory
    conversation_history_file: str = Field(default="./runtime/conversation_history.json", validation_alias="CONVERSATION_HISTORY_FILE")
    conversation_records_file: str = Field(default="./runtime/conversation_records.json", validation_alias="CONVERSATION_RECORDS_FILE")
    short_memory_file: str = Field(default="./runtime/short_memory.json", validation_alias="SHORT_MEMORY_FILE")
    long_memory_file: str = Field(default="./runtime/long_memory.json", validation_alias="LONG_MEMORY_FILE")
    max_history_messages: int = Field(default=10, validation_alias="MAX_HISTORY_MESSAGES")
    max_conversation_records: int = Field(default=25, validation_alias="MAX_CONVERSATION_RECORDS")
    max_short_memory_items: int = Field(default=5, validation_alias="MAX_SHORT_MEMORY_ITEMS")
    max_long_memory_items: int = Field(default=10, validation_alias="MAX_LONG_MEMORY_ITEMS")

    # Local Text Generation Model
    local_text_generation_path: str = Field(default="", validation_alias="LOCAL_TEXT_GENERATION_PATH")
    local_text_generation_max_new_tokens: int = Field(default=256, validation_alias="LOCAL_TEXT_GENERATION_MAX_NEW_TOKENS")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()


