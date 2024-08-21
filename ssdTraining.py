import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
import numpy as np

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
        self.class_to_idx = {'your_class_name': 1, 'background': 0}  # Include background class

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        ann_path = os.path.join(self.root, "Annotations", self.annotations[idx])

        img = Image.open(img_path).convert("RGB")

        try:
            tree = ET.parse(ann_path)
            root = tree.getroot()
        except ET.ParseError:
            print(f"Warning: Skipping corrupted file {ann_path}")
            return None, None

        boxes = []
        labels = []

        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            label = obj.find("name").text

            labels.append(self.class_to_idx.get(label, 0))

        if not labels:  # If no valid labels found, return a background class
            labels.append(0)
            boxes.append([0, 0, 0, 0])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms:
            img = self.transforms(img)  # Apply transformations only to the image

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]  # Filter out None values
    return tuple(zip(*batch))

if __name__ == '__main__':
    # Paths
    data_dir = "VOCdevkit/VOC2024"

    # Dataset
    dataset = VOCDataset(data_dir, transforms=get_transform(train=True))
    dataset_test = VOCDataset(data_dir, transforms=get_transform(train=False))

    # Split dataset
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_val = torch.utils.data.Subset(dataset_test, indices[-50:])

    # Data loaders
    data_loader = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Load the pre-trained SSD model
    model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

    # Replace the head of the model with the number of classes you have (including background)
    num_classes = 2  # 1 class + background
    model.head.classification_head.num_classes = num_classes

    # Model and optimizer
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)  # Adjusted learning rate

    # Path to save the model weights
    save_path = "model_weights.pth"

    # Training loop
    num_epochs = 10
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in data_loader:
            if images is None or targets is None:
                continue

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            try:
                loss_dict = model(images, targets)
            except Exception as e:
                print(f"Exception occurred during model forward pass: {e}")
                continue

            # Check if loss_dict is a dictionary
            if isinstance(loss_dict, dict):
                try:
                    losses = sum(loss for loss in loss_dict.values())
                except Exception as e:
                    print(f"Exception occurred while summing losses: {e}")
                    continue
            elif isinstance(loss_dict, list):
                try:
                    losses = sum(loss_dict)
                except Exception as e:
                    print(f"Exception occurred while summing losses from list: {e}")
                    continue
            else:
                print(f"Unexpected type for loss_dict: {type(loss_dict)}")
                continue

            if torch.isnan(losses) or torch.isinf(losses):
                print("Warning: Detected NaN or Inf in losses.")
                continue

            losses.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            running_loss += losses.item()

        epoch_loss = running_loss / len(data_loader)
        print(f"Epoch: {epoch}, Loss: {epoch_loss}")

        # Validation
        model.eval()
        with torch.no_grad():
            running_val_loss = 0.0
            for images, targets in data_loader_val:
                if images is None or targets is None:
                    continue

                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                try:
                    loss_dict = model(images, targets)
                except Exception as e:
                    print(f"Exception occurred during model forward pass: {e}")
                    continue

                # Check if loss_dict is a dictionary
                if isinstance(loss_dict, dict):
                    try:
                        losses = sum(loss for loss in loss_dict.values())
                    except Exception as e:
                        print(f"Exception occurred while summing losses: {e}")
                        continue
                elif isinstance(loss_dict, list):
                    try:
                        losses = sum(loss_dict)
                    except Exception as e:
                        print(f"Exception occurred while summing losses from list: {e}")
                        continue
                else:
                    print(f"Unexpected type for loss_dict: {type(loss_dict)}")
                    continue

                if torch.isnan(losses) or torch.isinf(losses):
                    print("Warning: Detected NaN or Inf in validation losses.")
                    continue

                running_val_loss += losses.item()

            epoch_val_loss = running_val_loss / len(data_loader_val)
            print(f"Validation Loss: {epoch_val_loss}")

            # Save the model if validation loss has improved
            if epoch_val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss} to {epoch_val_loss}. Saving model...")
                best_val_loss = epoch_val_loss
                torch.save(model.state_dict(), save_path)

    print("Training complete!")
