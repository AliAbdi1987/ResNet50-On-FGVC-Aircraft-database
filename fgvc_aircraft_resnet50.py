import torch
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from torchsummary import summary
import csv
import os

# Function to find the latest checkpoint file
def find_latest_checkpoint(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pth')]
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    return os.path.join(checkpoint_dir, latest_checkpoint)

class AircraftDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = sorted(self.img_labels.iloc[:, 1].unique())
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.classes.index(self.img_labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomAffine(0, shear=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomAdjustSharpness(2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Combine both train and test dataset
train_df = pd.read_csv('./data/archive/train.csv')
test_df = pd.read_csv('./data/archive/test.csv')
combined_df = pd.concat([train_df, test_df])
combined_df.to_csv('./data/archive/combined_train_test.csv', index=False)

train_dataset = AircraftDataset(csv_file='./data/archive/combined_train_test.csv', img_dir='./data/archive/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images', transform=data_transforms['train'])
val_dataset = AircraftDataset(csv_file='./data/archive/val.csv', img_dir='./data/archive/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images', transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

model_ft = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model_ft.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model_ft.fc.in_features, len(train_dataset.classes)))
model_ft = model_ft.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.00005, weight_decay=0.0005)
exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.3, patience=3, verbose=True)

def save_checkpoint(model, optimizer, scheduler, epoch, filename):
    checkpoint = {
        'epoch': epoch +1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, scheduler, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']

checkpoint_dir = './'
checkpoint_path = find_latest_checkpoint(checkpoint_dir)
start_epoch = 0
if checkpoint_path and os.path.exists(checkpoint_path): 
    start_epoch = load_checkpoint(model_ft, optimizer_ft, exp_lr_scheduler, checkpoint_path) + 1
    print(f"Resuming training from epoch {start_epoch}")
else:
    print("No checkpoint found, starting training from scratch")

def train_model(model, criterion, optimizer, scheduler, num_epochs, start_epoch, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    end_epoch = start_epoch + num_epochs
    log_file = 'training_log.csv'
    fieldnames = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc']

    # Initialize the log file with the fieldnames
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    with open(log_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(start_epoch, end_epoch):
        print(f'\nEpoch {epoch + 1}/{end_epoch}')
        print('-' * 10)

        # Temporary storage for the current epoch's loss and accuracy
        epoch_data = {
        'train_loss' : 0,
        'val_loss' : 0,
        'train_acc' : 0,
        'val_acc' : 0
        }

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            epoch_data[f'{phase}_loss'] = epoch_loss
            epoch_data[f'{phase}_acc'] = epoch_acc * 100 # Convert to percentage

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc * 100:.2f}%')


            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())


        with open(log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({
                'epoch': epoch + 1,
                'train_loss': epoch_data['train_loss'],
                'val_loss': epoch_data['val_loss'],
                'train_acc': epoch_data['train_acc'].item() if torch.is_tensor(epoch_data['train_acc']) else epoch_data['train_acc'], 
                'val_acc': epoch_data['val_acc'].item() if torch.is_tensor(epoch_data['val_acc']) else epoch_data['val_acc']
                })
        
        if ((epoch +1) % 5 == 0 or epoch == end_epoch - 1):
            checkpoint_filename = f'checkpoint_{epoch}.pth'
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_filename)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1, start_epoch=start_epoch)
summary(model_ft, (3, 224, 224))

# Save the model
torch.save(model_ft.state_dict(), 'fgvc_aircraft_resnet50.pth')