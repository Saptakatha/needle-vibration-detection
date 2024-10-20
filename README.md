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



```python
python train.py --train_images path/to/train/images/dir --train_labels path/to/train/label/file --val_images path/to/val/images/dir --val_labels path/to/train/label/file --output_model_dir path/to/save/model
```
