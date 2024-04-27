import os
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms
from torchvision.datasets import CocoDetection

# Constants
DATA_DIR = '/path/to/your/dataset'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')

# Assuming the dataset is in COCO format after downloading and unzipping from Roboflow
def get_transform(train):
    transforms_list = []
    transforms_list.append(transforms.ToTensor())
    if train:
        # Example of more augmentations during training
        transforms_list.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transforms_list)

class CustomDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super(CustomDataset, self).__init__(root, annFile)
        self.transform = transform

    def __getitem__(self, index):
        img, target = super(CustomDataset, self).__getitem__(index)
        target = {k: v for k, v in target.items() if k in ['boxes', 'labels']}
        if self.transform:
            img = self.transform(img)
        return img, target

# Load the training and validation datasets
train_dataset = CustomDataset(TRAIN_DIR, os.path.join(TRAIN_DIR, '_annotations.coco.json'),
                              transform=get_transform(train=True))
val_dataset = CustomDataset(VAL_DIR, os.path.join(VAL_DIR, '_annotations.coco.json'),
                            transform=get_transform(train=False))

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4,
                          collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4,
                        collate_fn=lambda x: tuple(zip(*x)))

def create_model(num_classes):
    # Load a model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replace the classifier with a new one for finetuning
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

# Initialize the model
num_classes = 2  # 1 class + background
model = create_model(num_classes)

# Move model to the right device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Training
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {losses.item()}")

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Train for 10 epochs
num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch)
    torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
