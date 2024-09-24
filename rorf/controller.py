import logging
import os
import pickle
from tqdm import tqdm
from typing import List, Union, Optional

import numpy as np
from huggingface_hub import hf_hub_download

from rorf.router.utils import get_embedding_model

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class RoutingError(Exception):
    pass


class Controller:
    def __init__(
        self,
        router: str,
        model_a: str,
        model_b: str,
        threshold: float,
    ):
        """
        Initialize the RoRF controller with the specified router, model A, model B, and threshold.
        Threshold determines the percentage of calls made to model A by the router.
        """
        self._validate_router_threshold(router, threshold)
        self.embedding_provider = self._parse_model_name(router)
        logger.info(f"Initializing RoRF controller for {router} with {self.embedding_provider} embeddings...")
        self.router_model, self.embedding_model = self.load(router, self.embedding_provider)
        self.model_a, self.model_b = model_a, model_b
        self.threshold = threshold
    
    def _validate_router_threshold(
        self, router: Optional[str], threshold: Optional[float]
    ):
        """
        Validate the router and threshold.
        """
        if router is None or threshold is None:
            raise RoutingError("Router or threshold unspecified.")
        if not 0 <= threshold <= 1:
            raise RoutingError(
                f"Invalid threshold {threshold}. Threshold must be a float between 0.0 and 1.0."
            )
    
    def _parse_model_name(self, router: str):
        """
        Parse the method and embedding model provider's name from the router name.
        """
        method, embedding_provider = router.split("/")[1].split('-')[0:2]
        if not method == "rorf":
            raise RoutingError(f"Invalid method {method}. Method must be 'rorf'.")
        if not embedding_provider in ["jina", "voyage", "openai"]:
            raise RoutingError(
                f"Invalid embedding provider {embedding_provider}. Embedding provider must be 'jina', 'voyage', or 'openai'."
            )
        return embedding_provider
    
    def batch_calculate_win_rate(self, prompts: List[str]) -> List[float]:
        """
        Given a list of prompts, calculate the win rates (Model A probability) using the RoRF router.
        """
        logger.info(f"Calculating win rates for {len(prompts)}...")
        model_a_probs = []
        prompt_embeddings = self.embedding_model.get_prompt_embeddings(prompts)
        for prompt in tqdm(prompt_embeddings, total=len(prompt_embeddings)):
            prompt = prompt.reshape((1, -1))
            _, model_a_prob, _ = self.predict_proba(prompt)
            model_a_probs.append(model_a_prob)
        return model_a_probs

    def route(self, prompt: Union[str, np.ndarray]) -> str:
        """
        Given a prompt or prompt embedding, return the model recommended by the RoRF router.
        """
        recommended_model, _, _ = self.predict_proba(prompt)
        return recommended_model

    def predict_proba(self, prompt: Union[str, np.ndarray]) -> str:
        """
        Given a prompt or prompt embedding, return the model recommended by the RoRF router along with the probabilities of the two models.
        """
        if isinstance(prompt, str):
            prompt_embedding = self.embedding_model.get_prompt_embeddings([prompt])
        else:
            prompt_embedding = prompt
        label_scores = self.router_model.predict_proba(prompt_embedding)[0]
        model_a_prob = label_scores[0] + label_scores[1]
        model_b_prob = label_scores[2] + label_scores[3]

        if model_a_prob >= self.threshold:
            recommended_model = self.model_a
        else:
            recommended_model = self.model_b
        return recommended_model, model_a_prob, model_b_prob

    def load(self, repo_id: str, embedding_provider: str):
        """
        Load the RoRF router and embedding model from the specified local or Huggingface repo.
        """
        logger.info(f"Initializing {embedding_provider} embedding model...")
        embedding_model = get_embedding_model(embedding_provider)

        if os.path.exists(f"{repo_id}/classifier.pkl"):
            logger.info(f"{repo_id} exists. Loading router...")
            router_path = f"{repo_id}/classifier.pkl"
        else:
            logger.info(f"{repo_id} does not exist. Downloading router...")
            router_path = hf_hub_download(
                repo_id=repo_id,
                filename="classifier.pkl",
            )
        with open(router_path, "rb") as f:
            router_model = pickle.load(f)
        return router_model, embedding_model