from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-70b-versatile"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = ""
    qdrant_use_tls: bool = False
    qdrant_collection: str = "python_qna"
    active_prompt_version: str = "v1"
    backend_url: str = "http://backend:8000"

    class Config:
        env_file = ".env"


settings = Settings()
