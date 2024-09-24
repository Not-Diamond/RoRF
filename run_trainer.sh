python trainer.py rorf-classifier \
    --model_a "llama-3.1-405b-instruct" \
    --model_b "llama-3.1-70b-instruct" \
    --dataset_path "notdiamond/rorf-llama31405b-llama3170b-battles" \
    --eval_dataset "notdiamond/rorf-llama31405b-llama3170b-battles" \
    --embedding_provider "openai" \
    --max_depth 20 \
    --n_estimators 100 \
    --model_id "rorf-llama31405b-llama3170b" \
    --model_org "notdiamond"
