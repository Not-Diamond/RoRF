import argparse
from datasets import Dataset, load_dataset
from rorf.controller import Controller

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dataset", type=str)
    parser.add_argument("--router", type=str)
    parser.add_argument("--model-a-pct", type=float)
    parser.add_argument("--task", type=str, choices=["generate", "calibrate"], default="calibrate")
    args = parser.parse_args()

    if args.task == "generate":
        calibration_df = load_dataset(args.calibration_dataset, split="train").to_pandas()
        controller = Controller(
            router=args.router,
            model_a=None,
            model_b=None,
            threshold=0,
        )

        win_rates = controller.batch_calculate_win_rate(
            calibration_df["Input"].tolist()
        )

        calibration_df[args.router] = win_rates
        Dataset.from_pandas(calibration_df).push_to_hub(
            f"{args.calibration_dataset}-rorf-thresholds",
            private=True
        )

    elif args.task == "calibrate":
        thresholds_df = load_dataset(f"{args.calibration_dataset}-rorf-thresholds", split="train").to_pandas()
        threshold = thresholds_df[args.router].quantile(q=1 - args.model_a_pct)
        print(
            f"Threshold = {round(threshold, 5)} for {args.model_a_pct * 100}% calls to Model A."
        )