import click
from typing import Optional, Union, List

from rorf.router.utils import write_config_to_json

import sys
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO,
    stream=sys.stdout,
    datefmt="%I:%M:%S",
)

@click.group()
@click.pass_context
def run(ctx):
    ctx.obj = {}


@run.command()
@click.option("--model_a", type=str, default="llama-3.1-405b-instruct")
@click.option("--model_b", type=str, default="llama-3.1-70b-instruct")
@click.option("--dataset_path", type=str)
@click.option("--eval_dataset", type=str)
@click.option("--embedding_provider", type=str, default="jina")
@click.option("--prompt_embedding_cache", type=str, default=None)
@click.option("--max_depth", type=int, default=20)
@click.option("--max_features", default=1.0)
@click.option("--n_estimators", type=int, default=100)
@click.option("--save_dir", type=str, default="checkpoints")
@click.option("--model_id", type=str, default=None)
@click.option("--model_org", type=str)

@click.pass_context
def rorf_classifier(
    ctx,
    model_a: str,
    model_b: str,
    dataset_path: str,
    eval_dataset: str,
    embedding_provider: str,
    prompt_embedding_cache: Optional[str],
    max_depth: int,
    max_features: Union[float, str],
    n_estimators: int,
    save_dir: str,
    model_id: str,
    model_org: str,
):
    from rorf.router.rorf import RoRFTrainer

    llms = [model_a, model_b]

    configs = {
        "trainer": "RoRF",
        "llms": llms,
        "dataset_path": dataset_path,
        "eval_dataset": eval_dataset,
        "embedding_provider": embedding_provider,
        "prompt_embedding_cache": prompt_embedding_cache,
        "max_depth": max_depth,
        "max_features": max_features,
        "n_estimators": n_estimators,
        "save_dir": save_dir,
        "model_id": model_id,
        "model_org": model_org,
    }

    trainer_obj = RoRFTrainer(
        llms=llms,
        dataset_path=dataset_path,
        eval_dataset=eval_dataset,
        embedding_provider=embedding_provider,
        prompt_embedding_cache=prompt_embedding_cache,
        max_depth=max_depth,
        max_features=max_features,
        n_estimators=n_estimators,
        save_dir=save_dir,
        model_id=model_id,
        model_org=model_org,
    )
    write_config_to_json(configs, trainer_obj.save_path)
    ctx.obj["trainer"] = trainer_obj
    ctx.obj["configs"] = configs


@run.result_callback()
@click.pass_context
def process_result(ctx, result, **kwargs):
    trainer = ctx.obj.get("trainer", None)
    configs = ctx.obj.get("configs", None)
    result = trainer.train()


if __name__ == "__main__":
    run()
