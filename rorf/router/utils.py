import json
from typing import Dict

from rorf.router.embeddings import BaseEmbeddings, VoyageEmbeddings, OpenAIEmbeddings, JinaEmbeddings

def get_embedding_model(embedding_provider: str) -> BaseEmbeddings:
    """
    Get an embedding model based on the embedding provider.
    """
    if embedding_provider == "voyage":
        embedding_model = VoyageEmbeddings()
    elif embedding_provider == "openai":
        embedding_model = OpenAIEmbeddings()
    elif embedding_provider == "jina":
        embedding_model = JinaEmbeddings()
    return embedding_model


def write_config_to_json(config: Dict, path: str) -> None:
    """
    Write the configuration to a JSON file.
    """
    with open(f"{path}/config.json", "w") as f:
        json.dump(config, f, indent=2)