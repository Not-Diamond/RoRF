import logging
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict, List, Optional

from datasets import load_dataset
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    @abstractmethod
    def get_dataset(self):
        raise NotImplementedError
    

class NDEvalsDataset(BaseDataset):
    def __init__(self, dataset_path: Path, llms: List[str], eval_dataset: Optional[List[str]] = None):
        self.dataset_path = dataset_path
        self.llms = llms
        self.eval_dataset = eval_dataset
        
    def get_dataset(self, split) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Loading {self.dataset_path}")
        samples = load_dataset(self.dataset_path, split=split).to_pandas()
        logger.info(f"Number of {split} samples: {len(samples)}")
        
        prompts = []
        targets = []
        for _, row in samples.iterrows():
            prompt = row["Input"]
            target = {}
            for model in self.llms:
                target[model] = row[f"{model}/score"]
            prompts.append(prompt)
            targets.append(target)
        
        assert len(prompts) == len(targets)
        return prompts, targets, samples
    
    def get_features(self, prompts: List[str], targets: List[Dict], prompt_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Get pairwise dataset with a single target label after comparing model responses
        '''
        logger.info("Getting pairwise data")
        
        model_a, model_b = self.llms
        label_stats = {}
        label_stats['NO_MODEL'] = 0
        label_stats['BOTH_MODEL'] = 0
        for model in self.llms:
            label_stats[model] = 0

        inputs = []
        labels = []
        scores = []
        for prompt_i, target in tqdm(enumerate(targets)):
            model_a_score = target[model_a]
            model_b_score = target[model_b]
            scores.append({model_a: model_a_score, model_b: model_b_score})

            if model_a_score > model_b_score:
                label = 0
                label_stats[model_a] += 1
            elif model_a_score == model_b_score:
                if model_a_score == 0:
                    label = 1
                    label_stats["NO_MODEL"] += 1
                else:
                    label = 2
                    label_stats["BOTH_MODEL"] += 1
            else:
                label = 3
                label_stats[model_b] += 1

            features = prompt_embeddings[prompt_i]
            inputs.append(features)
            labels.append(label)

        logger.info(f"Label statistics: {label_stats}")
        return np.vstack(inputs, dtype=np.float32), np.array(labels), scores