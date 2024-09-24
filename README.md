# RoRF - Routing on Random Forests

RoRF is a framework for training and serving random forest-based LLM routers.

Our core features include:
- 12 pre-trained routers across 6 model pairs and 2 embedding models ([jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3), [voyageai/voyage-large-2-instruct](https://docs.voyageai.com/docs/embeddings#model-choices)) that reduce costs while either maintaining or improving performance.
- Our pre-trained routers outperform existing routing solutions, including open-source and commercial offerings.

## Installation
### PyPI
```sh
pip install rorf
```
### Source
```sh
git clone https://github.com/Not-Diamond/RoRF
cd RoRF
pip install -e .
```

## Quickstart
We adopt RouteLLM's [Controller](https://github.com/lm-sys/RouteLLM/tree/main?tab=readme-ov-file#quickstart) to allow users to replace their existing routing setups with RoRF. Our `Controller` requires a `router` (available either locally or on Huggingface Hub) that routes between `model_a` (usually stronger/expensive) and `model_b` (usually weaker/cheaper). Our release includes 6 model pairs between different models and providers as described in Model Support.
```python
from rorf.controller import Controller

router = Controller(
    router="notdiamond/rorf-jina-llama31405b-llama3170b",
    model_a="llama-3.1-405b-instruct",
    model_b="llama-3.1-70b-instruct",
    threshold=0.3,
)

recommended_model = router.route("What is the meaning of life?")
print(f"Recommended model: {recommended_model}")
```
We also provide a `threshold` parameter that determines the percentage of calls made to each model, allowing users to decide their own cost vs performance tradeoffs.

## Training RoRF
We include our training framework for RoRF so that users can train custom routers on their own data and model pairs. `trainer.py` is the entry-point for training, and `run_trainer.sh` provides an example command to train a model router for `llama-3.1-405b-instruct` vs `llama-3.1-70b-instruct` on top of Jina AI's embeddings.


## Motivation
Our experiments show that:
- Routing between a pair of strong and weak models can reduce costs while maintaining the strong model's performance.
- Routing between a pair of two strong models can reduce costs while outperforming both individual models.