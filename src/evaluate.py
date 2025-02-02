import torch
import argparse
from architectures import FCN, UNet
from data_loader import load_dataset
from helpers import dice_coeff, visualize_predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def evaluate_model(model_name):
    """Evaluates trained model and computes Dice coefficient."""
    # Load dataset
    data_img, data_seg, _, _, idx_test = load_dataset()

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

    imgs_testset = data_img[idx_test]
    segs_testset = data_seg[idx_test]

    # Compute predictions
    with torch.no_grad():
        features = model(imgs_testset)
        prediction = torch.argmax(features, dim=1)

    # Compute Dice Score
    d0 = torch.zeros(len(imgs_testset), 2).to(device)
    for i in range(len(imgs_testset)):
        pred = prediction[i]
        target = segs_testset[i]
        d0[i] = dice_coeff(target, pred, n_classes=2)

    print(f"Test set size = {len(imgs_testset)}")
    print(f"Dice BG mean = {d0[:,0].mean().item():.4f}")
    print(f"Dice Polyp mean = {d0[:,1].mean().item():.4f}")
    print(f"Dice overall mean = {d0.mean().item():.4f}")

    # Visualize predictions
    visualize_predictions(imgs_testset, prediction, idx_test, save_outputs=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on test data.")
    parser.add_argument("model", type=str, default="unet", help="Model to evaluate: 'fcn' or 'unet'")
    args = parser.parse_args()

    evaluate_model(args.model)
