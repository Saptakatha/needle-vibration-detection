import os
import argparse
import cv2
import torch
import numpy as np
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

class GaugeModel(nn.Module):
    """
    Neural network model architecture for predicting the needle tip and dial center coordinates
    of a pressure gauge.
    """
    def __init__(self):
        super(GaugeModel, self).__init__()
        # Load a pre-trained ResNet18 model
        self.backbone = models.resnet18(pretrained=True)
        # Modify the final fully connected layer to output 4 coordinates
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 4)

    def forward(self, x):
        return self.backbone(x)

def load_model(model_path, device):
    """
    Load the trained model from the specified path.

    Args:
        model_path (str): Path to the trained model file.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded model.
    """
    model = GaugeModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def process_image(image, model, device, transform):
    """
    Process a single image to predict the coordinates and overlay them on the image.

    Args:
        image (np.ndarray): Input image.
        model (torch.nn.Module): Trained model.
        device (torch.device): Device to perform inference on.
        transform (callable): Transformations to apply to the image.

    Returns:
        np.ndarray: Image with predicted coordinates overlaid.
    """
    # Get the original image dimensions
    original_height, original_width = image.shape[:2]

    # Apply transformations
    transformed_image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(transformed_image).cpu().numpy().flatten()

    # Extract coordinates
    needle_tip = (int(output[0]), int(output[1]))
    dial_center = (int(output[2]), int(output[3]))

    # Calculate scaling factors
    scale_x = original_width / 128
    scale_y = original_height / 128

    # Transform coordinates back to original image scale
    needle_tip = (int(needle_tip[0] * scale_x), int(needle_tip[1] * scale_y))
    dial_center = (int(dial_center[0] * scale_x), int(dial_center[1] * scale_y))

    # Print the predicted coordinates
    print(f"Needle Tip: {needle_tip}, Dial Center: {dial_center}")

    # Overlay coordinates on the original image
    cv2.circle(image, needle_tip, 1, (0, 0, 255), -1)  # Red for needle tip
    cv2.circle(image, dial_center, 1, (0, 255, 0), -1)  # Green for dial center

    return image

def main(args):
    # Define transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Define the device to be used for inference (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    model = load_model(args.model_path, device)

    # Read the input image
    image = cv2.imread(args.input_image)
    if image is None:
        print(f"Error: Unable to read image file {args.input_image}")
        return

    # Process the image
    processed_image = process_image(image, model, device, transform)

    # Save the processed image
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = os.path.join(args.output_dir, args.input_image.split('/')[-1])
    cv2.imwrite(output_path, processed_image)
    print(f"Processed image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Infer the trained model on an image and save the output with predicted coordinates overlaid.')
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output image.')

    args = parser.parse_args()
    main(args)