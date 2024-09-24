import logging
import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from tqdm import tqdm
from typing import List

load_dotenv()

import numpy as np
import torch
import tiktoken
from openai import OpenAI
from transformers import AutoModel
import voyageai

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

class BaseEmbeddings(ABC):    
    @abstractmethod
    def get_prompt_embeddings(self, prompts: List[str]) -> np.ndarray:
        raise NotImplementedError
    

class VoyageEmbeddings(BaseEmbeddings):
    def __init__(self, model_name: str = "voyage-large-2-instruct"):
        """
        Initialize a Voyage AI client.
        """
        VOYAGE_API_KEY=os.getenv("VOYAGE_API_KEY")
        self.embedding_model = voyageai.Client(api_key=VOYAGE_API_KEY)
        self.model_name = model_name
    
    def get_prompt_embeddings(self, prompts: List[str]) -> np.ndarray:
        """
        Get embeddings from Voyage AI for a list of prompts.
        """
        if self.model_name == "voyage-large-2-instruct":
            # instructions: https://github.com/voyage-ai/voyage-large-2-instruct
            prompts = ["Cluster the text: " + str(p) for p in prompts]
        embeddings = [] 
        if len(prompts) > 1:
            batch_size = 128
            for i in tqdm(range(0, len(prompts), batch_size), total=len(prompts) // batch_size):
                embeddings += self.embedding_model.embed(
                    prompts[i:i + batch_size], model=self.model_name, input_type=None
                ).embeddings
        else:
            embeddings = self.embedding_model.embed(prompts, model=self.model_name).embeddings
        embeddings = np.array(embeddings)
        return embeddings
    

class OpenAIEmbeddings(BaseEmbeddings):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initialize an OpenAI client.
        """
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.embedding_model = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name

    def _trim_gpt_message(self, message):
        """
        Return the number of tokens used by a message.
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        encoded_message = encoding.encode(message)
        num_tokens = len(encoded_message)
        max_tokens = 8150  # the maximum for text-embedding-3-large is 8191
        if num_tokens > max_tokens:
            encoded_message = encoded_message[:max_tokens]
            decoded_messgae = encoding.decode(encoded_message)
            return decoded_messgae
        return message
    
    def get_prompt_embeddings(self, prompts: List[str]) -> np.ndarray:
        """
        Get embeddings from OpenAI for a list of prompts.
        """
        embeddings = []
        if len(prompts) > 1:
            batch_size = 2048
            for i in tqdm(range(0, len(prompts), batch_size), total=len(prompts) // batch_size):
                # replace newlines, which can negatively affect performance.
                list_of_prompts = prompts[i: i + batch_size]
                list_of_prompts = [str(prompt).replace("\n", " ").replace("<|endoftext|>", "").replace("<|endofprompt|>", "") for prompt in list_of_prompts]
                list_of_prompts = [self._trim_gpt_message(prompt) for prompt in list_of_prompts]
                data = self.embedding_model.embeddings.create(
                    input=list_of_prompts, model=self.model_name, encoding_format='float').data
                embeddings.extend([d.embedding for d in data])
        else:
            prompt = prompts[0].replace("\n", " ")
            request = self.embedding_model.embeddings.create(
                input=prompt,
                model=self.model_name,
                encoding_format="float"
            )
            embeddings.append(request.data[0].embedding)
        embeddings = np.array(embeddings)
        return embeddings
    
    
class JinaEmbeddings(BaseEmbeddings):
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v3"):
        """
        Initialize a Jina model locally.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True, 
            device_map=device
        )
        self.model_name = model_name
    
    def get_prompt_embeddings(self, prompts: List[str]) -> np.ndarray:
        """
        Get embeddings from a local Jina AI embedding model for a list of prompts.
        """
        list_of_prompts = [str(prompt) for prompt in prompts]
        embeddings = self.embedding_model.encode(list_of_prompts, task="classification")
        embeddings = np.asarray(embeddings)
        return embeddings