# Pressure Gauge Needle Vibration Detection

This repository contains scripts for Pressure Gauge Needle vibration detection by predicting the needle tip and dial center coordinates in pressure gauges using a deep learning model. The workflow includes data preparation, model training, and inference on both images and videos.


### Installation <a name="install"></a>
First, clone the repository and navigate to the project directory:
```python
git clone https://github.com/Saptakatha/needle-vibration-detection.git
cd needle-vibration-detection
```

Install the required packages using ```requirements.txt```:
```python
pip install -r requirements.txt
```

### Data Preparation <a name="prepare_data"></a>
##### Preparing Training and Validation Data
The dataset used for training and validation is prepared from a synthetic dataset available in Kaggle. The `dataset_preparation.py` script processes the dataset to extract and resize images, and generate label annotations for the needle tip and dial center coordinates.

1. Download the Kaggle synthetic dataset  and place it in the appropriate directory.
2. Run the `dataset_preparation.py` script to prepare the training and validation data:
```python
python python dataset_preparation.py
```

This script will:
	+ Load the JSON annotations from the Kaggle dataset.
	+ Extract bounding boxes and keypoints for needle tip and dial center.
	+ Resize the cropped images to 128x128 pixels.
	+ Adjust the keypoints to the resized images.
	+ Save the resized images and corresponding labels in the specified output directories and .csv files.

##### Example usage
```python
if __name__ == "__main__":
    image_dir = "../Kaggle_analog_gauge_synth_data/sample_synth_datasets/ds5.0/data"
    for split in ['train', 'val']:
        json_path = f"../Kaggle_analog_gauge_synth_data/sample_synth_datasets/ds5.0/{split}__kpts_coco.json"
        output_dir = f"../input_data/{split}"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        data_prep(image_dir, json_path, output_dir, split)
```

### Model Training <a name="train_model"></a>
Using the prepared dataset (images and labels), you can train a model to predict the needle tip and dial center coordinates using the `train.py` script.

##### Training the Model
1. Ensure the prepared dataset is available in the specified directories.
2. Run the `train.py` script to train the model:

```python
python train.py --train_images <path_to_train_images> --train_labels <path_to_train_labels.csv> --val_images <path_to_val_images> --val_labels <path_to_val_labels.csv> --output_model_dir <path_to_save_trained_model>
```

##### Example usage
```python
python train.py --train_images ../input_data/train --train_labels ../input_data/train/train_annotations.csv --val_images ../input_data/val --val_labels ../input_data/val/val_annotations.csv --output_model_dir ../models
```

This script will:

	+ Load the training and validation datasets.
	+ Define and train a neural network model using a pre-trained ResNet18 backbone.
	+ Train the model and save the trained model to the specified output directory.

