import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

from helpers import load_config, resolve_model_path
from architectures import FCN, UNet
from visualization import overlay_mask, tensor_to_numpy


def load_inference_model(config_path: str) -> torch.nn.Module:
    """load trained model based on config file

    args:
        config_path: path to model config yaml

    returns:
        loaded model ready for inference
    """
    config = load_config(config_path)
    config["model"]["path"] = resolve_model_path(config["model"])

    # init model
    model_class = FCN if config["model"]["name"] == "fcn" else UNet
    model = model_class(**config["model"]["params"]).to(config["device"])

    # load weights
    try:
        checkpoint = torch.load(config["model"]["path"], map_location=config["device"])
        model.load_state_dict(checkpoint["model_state"])
    except Exception as e:
        raise RuntimeError(f"failed loading model: {str(e)}")

    model.eval()
    return model, config


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (256, 256),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """preprocess medical image for model input

    args:
        image_path: path to input image
        target_size: model input dimensions
        mean: normalization mean values
        std: normalization std values

    returns:
        preprocessed image tensor
    """
    # load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"could not load image at {image_path}")

    # handle grayscale images
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize and normalize
    img_resized = cv2.resize(img, target_size)
    img_normalized = (img_resized.astype(np.float32) / 255.0 - mean) / std

    # convert to tensor
    return torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)


def run_inference(
    config_path: str,
    image_path: str,
    output_path: str = "output.png",
    overlay_alpha: float = 0.4,
) -> None:
    """run full medical image segmentation pipeline

    args:
        config_path: path to model config yaml
        image_path: path to input medical image
        output_path: path to save visualization
        overlay_alpha: transparency for prediction overlay
    """
    # load model and config
    model, config = load_inference_model(config_path)

    # preprocess image
    img_tensor = preprocess_image(
        image_path,
        target_size=config["data"].get("input_size", (256, 256)),
        mean=config["data"].get("mean", (0.485, 0.456, 0.406)),
        std=config["data"].get("std", (0.229, 0.224, 0.225)),
    ).to(config["device"])

    # run inference
    with torch.inference_mode():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).squeeze()

    # load original image
    original_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # create overlay
    overlayed_img = overlay_mask(
        image=original_img,
        mask=tensor_to_numpy(prediction, is_mask=True),
        color=config["visualization"].get("overlay_color", (255, 0, 0)),
        alpha=overlay_alpha,
    )

    # save result
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(overlayed_img, cv2.COLOR_RGB2BGR))
    print(f"inference result saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="medical image segmentation inference pipeline"
    )
    parser.add_argument(
        "--config", default="configs/inference.yml", help="path to model config file"
    )
    parser.add_argument("image", help="path to input medical image")
    parser.add_argument(
        "--output", default="results/prediction.png", help="path to save visualization"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.4, help="overlay transparency (0.0-1.0)"
    )

    args = parser.parse_args()

    try:
        run_inference(
            config_path=args.config,
            image_path=args.image,
            output_path=args.output,
            overlay_alpha=args.alpha,
        )
    except Exception as e:
        print(f"inference failed: {str(e)}")
        exit(1)
