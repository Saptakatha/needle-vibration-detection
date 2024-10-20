import os
import json
import pandas as pd
import cv2
from tqdm import tqdm

def data_prep(image_dir, json_path, output_dir, split):
    """
    Prepare the dataset by extracting and saving resized images and their corresponding labels.

    Args:
        image_dir (str): Directory containing the images.
        json_path (str): Path to the JSON file containing annotations.
        output_dir (str): Directory to save the resized images and labels.
        split (str): Dataset split (e.g., 'train', 'val').

    Returns:
        None
    """
    # Load the JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Create a dictionary to map image IDs to file names
    id_img_dict = {element['id']: element['file_name'].split('/')[-1] for element in data['images']}

    # Initialize a DataFrame to store labels
    labels_df = pd.DataFrame(columns=['file_name', 'needle_tip_x', 'needle_tip_y', 'dial_center_x', 'dial_center_y'])

    # Process each annotation
    for item in tqdm(data['annotations'], desc="Processing annotations"):
        img_id = item['image_id']
        image_name = id_img_dict[img_id]
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Unable to read image {image_path}. Skipping.")
            continue

        # Extract bounding box and crop the image
        x, y, w, h = map(int, item['bbox'])
        crop_image = image[y:y+h, x:x+w]
        crop_height, crop_width = crop_image.shape[:2]

        # Extract keypoints and adjust them relative to the cropped image
        kpts = [int(i) for i in item['keypoints']]
        kpts = [(kpts[i] - x, kpts[i + 1] - y) for i in range(0, len(kpts), 3)]
        dial_center = kpts[2]
        needle_tip = kpts[3]

        # Resize the cropped image
        resized_image = cv2.resize(crop_image, (128, 128))
        resized_height, resized_width = resized_image.shape[:2]

        # Adjust keypoints to the resized image
        resized_dial_center = (int(dial_center[0] * resized_width / crop_width), int(dial_center[1] * resized_height / crop_height))
        resized_needle_tip = (int(needle_tip[0] * resized_width / crop_width), int(needle_tip[1] * resized_height / crop_height))

        # # Overlay keypoints on the resized image
        # cv2.circle(resized_image, resized_dial_center, radius=2, color=(0, 255, 0), thickness=-1)
        # cv2.circle(resized_image, resized_needle_tip, radius=2, color=(0, 0, 255), thickness=-1)

        # Save the resized image
        resized_image_name = f"{split}_{img_id}.png"
        resized_image_path = os.path.join(output_dir, resized_image_name)
        cv2.imwrite(resized_image_path, resized_image)

        # Append the labels to the DataFrame
        temp_df = pd.DataFrame({
            'file_name': [resized_image_name],
            'needle_tip_x': [resized_needle_tip[0]],
            'needle_tip_y': [resized_needle_tip[1]],
            'dial_center_x': [resized_dial_center[0]],
            'dial_center_y': [resized_dial_center[1]]
        })
        labels_df = pd.concat([labels_df, temp_df], axis=0, ignore_index=True)

    # Save the labels DataFrame to a CSV file
    labels_csv_path = os.path.join(output_dir, f"{split}_annotations.csv")
    labels_df.to_csv(labels_csv_path, index=False)
    print(f"Labels saved to {labels_csv_path}")

if __name__ == "__main__":
    image_dir = "../Kaggle_analog_gauge_synth_data/sample_synth_datasets/ds5.0/data"
    for split in ['train', 'val']:
        json_path = f"../Kaggle_analog_gauge_synth_data/sample_synth_datasets/ds5.0/{split}__kpts_coco.json"
        output_dir = f"../input_data_annotated/{split}"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        data_prep(image_dir, json_path, output_dir, split)