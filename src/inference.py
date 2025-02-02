import torch
import cv2
import numpy as np
import argparse
from architectures import FCN, UNet
from helpers import overlay_segmentation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def run_inference(model_name, image_path, output_path="output.png"):
    """Runs inference on a single image and saves the result."""
    # Load model dynamically
    if model_name.lower() == "fcn":
        model = FCN().to(device)
    elif model_name.lower() == "unet":
        model = UNet().to(device)
    else:
        raise ValueError("Invalid model name! Choose either 'fcn' or 'unet'.")

    model_path = f"models/best_{model_name.lower()}.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Overlay prediction on original image
    overlayed_image = overlay_segmentation(img, prediction)

    # Save output
    cv2.imwrite(output_path, cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))
    print(f"Inference complete. Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an image using a trained model.")
    parser.add_argument("model", type=str, help="Model to use: 'fcn' or 'unet'")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save output")

    args = parser.parse_args()
    run_inference(args.model, args.image, args.output)
