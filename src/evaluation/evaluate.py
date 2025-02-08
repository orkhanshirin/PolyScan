import argparse
from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_tools.loader import load_data
from src.architectures import FCN, UNet
from src.metrics import (
    dice_score,
    hausdorff_distance_95,
    iou_score,
    sensitivity_specificity,
)
from src.visualization.overlay import overlay_mask
from src.visualization.plotting import plot_predictions

from src.helpers import load_config


def evaluate_model(config: dict[str, Any]) -> dict[str, float]:
    """main evaluation pipeline with comprehensive medical metrics"""
    # init model
    model_class = FCN if config["model_name"].lower() == "fcn" else UNet
    model = model_class(**config["model_params"]).to(config["device"])

    # load weights
    checkpoint = torch.load(config["model_path"], map_location=config["device"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # load data
    _, _, test_loader = load_data(
        batch_size=config["batch_size"],
        augmentations=None,  # no augmentation for evaluation
    )

    # init metric trackers
    metrics = {"dice": [], "iou": [], "sensitivity": [], "specificity": [], "hd95": []}

    # store for visualization
    images, predictions, ground_truths = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="evaluating"):
            img, mask = batch
            img = img.to(config["device"])

            # forward pass
            outputs = model(img)
            preds = torch.argmax(outputs, dim=1).cpu()
            probs = torch.softmax(outputs, dim=1).cpu()

            # store for visualization
            images.extend(img.cpu().numpy().transpose(0, 2, 3, 1))  # nhwc format
            ground_truths.extend(mask.numpy())
            predictions.extend(preds.numpy())

            # calculate metrics
            mask = mask.to(config["device"])
            if "dice" in config["metrics"]:
                metrics["dice"].append(
                    dice_score(preds, mask, config["model_params"]["num_classes"])
                )

            if "iou" in config["metrics"]:
                metrics["iou"].append(
                    iou_score(preds, mask, config["model_params"]["num_classes"])
                )

            if "sensitivity" in config["metrics"] or "specificity" in config["metrics"]:
                sens, spec = sensitivity_specificity(
                    preds, mask, config["model_params"]["num_classes"]
                )
                metrics["sensitivity"].append(sens)
                metrics["specificity"].append(spec)

            if config.get("compute_hd95"):
                # convert to one-hot format for hd95
                hd_mask = F.one_hot(
                    mask, config["model_params"]["num_classes"]
                ).permute(0, 3, 1, 2)
                metrics["hd95"].append(
                    hausdorff_distance_95(
                        probs, hd_mask.float(), config["model_params"]["num_classes"]
                    )
                )

    # aggregate results
    results = {}
    class_names = ["background", "polyp"]

    for metric in metrics:
        if metrics[metric]:  # only process calculated metrics
            stacked = torch.stack(metrics[metric])
            mean_values = stacked.mean(dim=0)

            for idx, name in enumerate(class_names):
                results[f"{metric}_{name}"] = mean_values[idx].item()

            results[f"{metric}_mean"] = mean_values.mean().item()

    # visualization
    if config["visualization"]["enable"]:
        output_dir = Path(config["visualization"]["output_path"])
        output_dir.mkdir(parents=True, exist_ok=True)

        overlays = [overlay_mask(img, pred) for img, pred in zip(images, predictions)]

        plot_predictions(
            images=images[: config["visualization"]["num_samples"]],
            masks=[
                m.squeeze()
                for m in ground_truths[: config["visualization"]["num_samples"]]
            ],
            overlays=overlays[: config["visualization"]["num_samples"]],
            save_path=output_dir / "predictions.png",
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate medical segmentation model")
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="override experiment name in config",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="direct path to model checkpoint (overrides config)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    with mlflow.start_run():
        metrics = evaluate_model(config)
        mlflow.log_metrics(metrics)

        print("\nmedical evaluation metrics:")
        for k, v in metrics.items():
            print(f"{k:25}: {v:.4f}")

        if config["visualization"]["enable"]:
            mlflow.log_artifacts(config["visualization"]["output_path"])
