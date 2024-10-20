import os
import argparse
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm

class GaugeDataset(Dataset):
    """
    Custom Dataset for loading gauge images and their corresponding labels.
    """
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to fetch.
        
        Returns:
            dict: Sample containing the image and its corresponding labels.
        """
        img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = self.data.iloc[idx, 1:].values.astype('float32')
        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

class GaugeModel(nn.Module):
    """
    Neural network model for predicting the needle tip and dial center coordinates
    of a pressure gauge using a pre-trained ResNet18 backbone.
    """
    def __init__(self):
        super(GaugeModel, self).__init__()
        # Load a pre-trained ResNet18 model
        self.backbone = models.resnet18(pretrained=True)
        # Modify the final fully connected layer to output 4 coordinates
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 4)

    def forward(self, x):
        return self.backbone(x)

class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after certain epochs.
    """
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        """
        Args:
            val_loss (float): Current validation loss.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def main(args):
    # Define transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Create datasets
    train_dataset = GaugeDataset(csv_file=args.train_labels, image_dir=args.train_images, transform=transform)
    val_dataset = GaugeDataset(csv_file=args.val_labels, image_dir=args.val_images, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Instantiate the model
    model = GaugeModel()

    # Define the device to be used for training (GPU if available, otherwise CPU)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model.to(device)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, verbose=True)

    # Training loop
    num_epochs = 50

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        print(f'Validation Loss: {val_loss:.4f}')

        # Step the learning rate scheduler
        scheduler.step()

        # Check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Save the trained model
    if not os.path.exists(args.output_model_dir):
        os.makedirs(args.output_model_dir)
    torch.save(model.state_dict(), os.path.join(args.output_model_dir, 'gauge_model.pth'))
    print(f'Model saved to {os.path.join(args.output_model_dir, "gauge_model.pth")}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model to predict needle tip and dial center of pressure gauge.')
    parser.add_argument('--train_images', type=str, required=True, help='Path to the directory containing training images.')
    parser.add_argument('--train_labels', type=str, required=True, help='Path to the CSV file containing training labels.')
    parser.add_argument('--val_images', type=str, required=True, help='Path to the directory containing validation images.')
    parser.add_argument('--val_labels', type=str, required=True, help='Path to the CSV file containing validation labels.')
    parser.add_argument('--output_model_dir', type=str, required=True, help='Directory to save the trained model.')

    args = parser.parse_args()
    main(args)