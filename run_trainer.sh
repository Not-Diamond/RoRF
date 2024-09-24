python trainer.py rorf-classifier \
    --llms "llama-3.1-405b-instruct" \
    --llms "llama-3.1-70b-instruct" \
    --dataset_path "notdiamond/rorf-llama31405b-llama3170b-battles" \
    --eval_dataset "notdiamond/rorf-llama31405b-llama3170b-battles" \
    --embedding_provider "jina" \
    --max_depth 20 \
    --n_estimators 100 \