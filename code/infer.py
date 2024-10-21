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

def process_frame(frame, model, device, transform):
    """
    Process a single frame to predict the coordinates and overlay them on the frame.

    Args:
        frame (np.ndarray): Input frame.
        model (torch.nn.Module): Trained model.
        device (torch.device): Device to perform inference on.
        transform (callable): Transformations to apply to the frame.

    Returns:
        np.ndarray: Frame with predicted coordinates overlaid.
        tuple: Predicted needle tip coordinates.
    """
    # Get the original frame dimensions
    original_height, original_width = frame.shape[:2]

    # Apply transformations
    transformed_frame = transform(frame).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(transformed_frame).cpu().numpy().flatten()

    # Extract coordinates
    needle_tip = (int(output[0]), int(output[1]))
    dial_center = (int(output[2]), int(output[3]))

    # Calculate scaling factors
    scale_x = original_width / 128
    scale_y = original_height / 128

    # Transform coordinates back to original frame scale
    needle_tip = (int(needle_tip[0] * scale_x), int(needle_tip[1] * scale_y))
    dial_center = (int(dial_center[0] * scale_x), int(dial_center[1] * scale_y))

    # Print the predicted coordinates
    print(f"Needle Tip: {needle_tip}, Dial Center: {dial_center}")

    # Overlay coordinates on the original frame
    cv2.circle(frame, needle_tip, 5, (0, 0, 255), -1)  # Red for needle tip
    cv2.circle(frame, dial_center, 5, (0, 255, 0), -1)  # Green for dial center

    return frame, needle_tip

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

    # Open the input video
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {args.input_video}")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create the locus directory if it doesn't exist
    locus_dir = os.path.join(args.output_dir, 'locus')
    if not os.path.exists(locus_dir):
        os.makedirs(locus_dir)

    frame_count = 0
    needle_tip_locus = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        processed_frame, needle_tip = process_frame(frame, model, device, transform)
        needle_tip_locus.append(needle_tip)

        # Save the processed frame
        output_path = os.path.join(args.output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(output_path, processed_frame)
        frame_count += 1

    # Overlay the locus on the initial frame
    if needle_tip_locus:
        initial_frame = cv2.imread(os.path.join(args.output_dir, "frame_0000.png"))
        for point in needle_tip_locus:
            cv2.circle(initial_frame, point, 2, (255, 0, 0), -1)  # Blue for locus points

        # Save the initial frame with the locus overlaid
        locus_output_path = os.path.join(locus_dir, "initial_frame_with_needle_tip_locus.png")
        cv2.imwrite(locus_output_path, initial_frame)

    cap.release()
    print(f"Processed {frame_count} frames. Output saved to {args.output_dir}")
    print(f"Locus overlaid on initial frame saved to {locus_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Infer the trained model on each frame of a video and save the output with predicted coordinates overlayed.')
    parser.add_argument('--input_video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output frames.')

    args = parser.parse_args()
    main(args)