import json
import logging
import os
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from tqdm import tqdm
from typing import Tuple, Dict, Any, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from huggingface_hub import HfApi

from rorf.router.dataloader import NDEvalsDataset
from .utils import get_embedding_model

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    @abstractmethod
    def train(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class RoRFTrainer(BaseTrainer):
    def __init__(
        self,
        llms: List,
        dataset_path: str,
        eval_dataset: str,
        embedding_provider: str,
        prompt_embedding_cache: Optional[str],
        # hyperparameters
        max_depth: int,
        max_features: Union[float, str],
        n_estimators: int,
        save_dir: str,
        model_id: str,
        model_org: str,
        *args,
        **kwargs
    ):
        self.llms = llms
        self.dataset_path = dataset_path
        self.eval_dataset = eval_dataset
        self.dataset = NDEvalsDataset(dataset_path, llms, eval_dataset)

        self.embedding_provider = embedding_provider
        logger.info(f"Initializing RoRF trainer with {embedding_provider} embeddings")
        self.embedding_model = get_embedding_model(embedding_provider)

        self.save_path = f"{save_dir}/{model_id}"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.repo_id = f"{model_org}/{model_id}"

        # args
        self.prompt_embedding_cache = prompt_embedding_cache
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.model_id = model_id
    
    def get_prompt_embeddings_dataset(self, prompts: List[str], mode: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Compute prompt embeddings for the given prompts.
        """
        logger.info(f"Creating {mode} embeddings for prompts")
        embeddings_path = f"{self.prompt_embedding_cache}/prompt_embeddings_{self.embedding_provider}_{mode}.pkl"
        if self.prompt_embedding_cache and os.path.exists(embeddings_path):
            logger.info("Loading cached prompt embeddings...")
            prompt_embeddings = self._load(self.prompt_embedding_cache, f"prompt_embeddings_{self.embedding_provider}_{mode}.pkl")
        else:
            logger.info("Prompt embedding cache does not exist. Encoding prompts...")
            prompt_embeddings = self.embedding_model.get_prompt_embeddings(prompts)

        self._save(self.save_path, prompt_embeddings, f"prompt_embeddings_{self.embedding_provider}_{mode}.pkl")
        logger.info(f"Prompt embeddings shape: {prompt_embeddings.shape}")
        return prompt_embeddings

    def _train_rf_classifier(self, features: np.ndarray, labels: np.ndarray) -> RandomForestClassifier:
        """
        Train a RandomForestClassifier on the given features and labels.
        """
        logger.info("Training RandomForestClassifier")
        rf_classifier = RandomForestClassifier(
            max_depth=self.max_depth,
            max_features=self.max_features,
            n_estimators=self.n_estimators,
            n_jobs=-1,
            random_state=420
        )
        rf_classifier.fit(features, labels)
        self._save(self.save_path, rf_classifier, "classifier.pkl")
        logger.info("RandomForestClassifier trained and saved.")
        return rf_classifier

    def _train(self) -> Tuple[RandomForestClassifier, Dict[str, np.ndarray], float]:
        """
        Train and evaluate a RandomForestClassifier on the training set.
        """
        train_prompts, train_targets, train_samples = self.dataset.get_dataset(split="train")
        train_prompt_embeddings = self.get_prompt_embeddings_dataset(train_prompts, mode="train")
        train_inputs, train_labels, train_scores = self.dataset.get_features(train_prompts, train_targets, train_prompt_embeddings)

        rf_classifier = self._train_rf_classifier(train_inputs, train_labels)
        logger.info("Training complete. Evaluating on training set...")
        train_predictions = rf_classifier.predict(train_inputs)
        train_accuracy = 1 - (sum(abs(train_predictions - train_labels)) / len(train_labels))
        logger.info(f"Train accuracy {train_accuracy}")
        return rf_classifier, train_prompt_embeddings, train_accuracy

    def train(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Train a RandomForestClassifier on the training set and evaluate on the evaluation/test set.
        """
        rf_classifier, train_prompt_embeddings, train_accuracy = self._train()
        if self.eval_dataset:
            evaluation_report = self.evaluate(rf_classifier)
        else:
            evaluation_report = dict(train_accuracy=train_accuracy)
        
        return self.push_to_hub(self.repo_id)

    def evaluate(self, rf_classifier: RandomForestClassifier) -> Dict[str, Any]:
        """
        Evaluate the RandomForestClassifier on the evaluation/test set.
        """
        logger.info(f"Evaluating model on {self.eval_dataset}")
        eval_prompts, eval_targets, eval_samples = self.dataset.get_dataset(split="eval")
        eval_prompt_embeddings = self.get_prompt_embeddings_dataset(eval_prompts, mode="eval")
        eval_inputs, eval_labels, eval_scores = self.dataset.get_features(eval_prompts, eval_targets, eval_prompt_embeddings)

        predictions = defaultdict(list)
        model_a, model_b = self.llms
        model_a_count, model_b_count = 0, 0
        test_accuracy, label_accuracy = 0, 0
        gold_labels, predicted_labels = [], []
        label_scores = []
        for i, (eval_input, eval_label) in tqdm(enumerate(zip(eval_inputs, eval_labels)), total=len(eval_labels)):
            scores = eval_scores[i]
            prompt_emb = eval_input.reshape((1, -1))
            classifier_preds = rf_classifier.predict_proba(prompt_emb)[0]
            gold_label = eval_label
            predicted_label = np.argmax(classifier_preds)
            model_a_prob = classifier_preds[0] + classifier_preds[1]
            model_b_prob = classifier_preds[2] + classifier_preds[3]
        
            gold_labels.append(gold_label)
            predicted_labels.append(predicted_label)
            label_scores.append(classifier_preds)

            if predicted_label == gold_label:
                label_accuracy += 1
            
            recommended_model = model_a if model_a_prob >= model_b_prob else model_b
            test_accuracy += scores[recommended_model]

            if recommended_model == model_a:
                model_a_count += 1
            else:
                model_b_count += 1
            
            predictions["prompt"].append(eval_prompts[i])
            predictions["model_a"].append(model_a)
            predictions["model_b"].append(model_b)
            predictions["model_a_score"].append(scores[model_a])
            predictions["model_b_score"].append(scores[model_b])
            predictions["gold_label"].append(gold_label)
            predictions["predicted_label"].append(predicted_label)
            predictions["recommended_model"].append(recommended_model)
            predictions["predicted_score"].append(scores[recommended_model])
        
        predictions_df = pd.DataFrame(predictions)
        self._save(self.save_path, predictions_df, "evaluation_predictions.csv")

        test_accuracy = test_accuracy / len(eval_labels)
        label_accuracy = label_accuracy / len(eval_labels)
        logger.info(f"Stats for {self.save_path}:")
        logger.info(f"Score Accuracy: {test_accuracy}")
        logger.info(f"Label Accuracy: {label_accuracy}")
        logger.info(f"{model_a} score: {model_a_count}")
        logger.info(f"{model_b} score: {model_b_count}")

        report = classification_report(gold_labels, predicted_labels, output_dict=True)
        results = {
            "save_path": str(self.save_path),
            "test_accuracy": test_accuracy,
            "label_accuracy": label_accuracy,
            "model_a": model_a,
            "model_b": model_b,
            "model_a_count": model_a_count,
            "model_b_count": model_b_count,
            "classification_report": report
        }
        self._save(self.save_path, results, "evaluation_results.json")
        logger.info(f"Classification report: {report}")
        return results

    def _save(self, path: str, obj: Any, obj_key: str) -> None:
        """
        Save an object to the specified path.
        """
        logger.info(f"Saving object {obj_key} to {path}")
        if "json" in obj_key:
            with open(f"{path}/{obj_key}", "w") as f:
                json.dump(obj, f, indent=2)
        elif "csv" in obj_key:
            obj.to_csv(f"{path}/{obj_key}", index=False)
        else:
            with open(f"{path}/{obj_key}", "wb") as f:
                pickle.dump(obj, f)

    def _load(self, path: str, obj_key: str) -> Any:
        """
        Load an object from the specified path.
        """
        if not os.path.exists(f"{path}/{obj_key}"):
            raise RuntimeError(f"Object {path}/{obj_key} does not exist.")

        logger.info(f"Loading {obj_key} from {path}")
        with open(f"{path}/{obj_key}", "rb") as f:
            obj = pickle.load(f)
        return obj
    
    def push_to_hub(self, repo_name: str) -> None:
        """
        Push the trained model and its configuration to the Huggingface Hub.
        """
        api = HfApi()
        repo_id = api.create_repo(repo_name, exist_ok=True, private=True)
        logger.info(f"Uploading model to {repo_id}")
        api.upload_file(
            path_or_fileobj=f"{self.save_path}/config.json",
            path_in_repo="config.json",
            repo_id=repo_name,
            repo_type="model",
        )
        return api.upload_file(
            path_or_fileobj=f"{self.save_path}/classifier.pkl",
            path_in_repo="classifier.pkl",
            repo_id=repo_name,
            repo_type="model",
        )