
import os
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics import Accuracy
from PIL import Image
from sklearn.model_selection import train_test_split

CHECKPOINT_FOLDER = './work_dir'

GROUND_TRUTH_TEXT_LABELS = {
    'bicycle': ['ECGPCG0059', 'ECGPCG0060', 'ECGPCG0061', 'ECGPCG0062', 'ECGPCG0064',
                'ECGPCG0065', 'ECGPCG0066', 'ECGPCG0067', 'ECGPCG0068', 'ECGPCG0069'],
    'treadmill': ['ECGPCG0035', 'ECGPCG0036', 'ECGPCG0037', 'ECGPCG0038', 'ECGPCG0039',
                  'ECGPCG0046', 'ECGPCG0047', 'ECGPCG0052', 'ECGPCG0054', 'ECGPCG0055', 'ECGPCG0056'],
    'stationary bicycle': ['ECGPCG0001', 'ECGPCG0002', 'ECGPCG0024', 'ECGPCG0025', 'ECGPCG0026',
                 'ECGPCG0027', 'ECGPCG0028','ECGPCG0029','ECGPCG0030', 'ECGPCG0031',
                 'ECGPCG0032','ECGPCG0033', 'ECGPCG0034'],
    'walking at constant speed': ['ECGPCG0041', 'ECGPCG0042', 'ECGPCG0043', 'ECGPCG0044', 'ECGPCG0045',
                                  'ECGPCG0048', 'ECGPCG0049', 'ECGPCG0050', 'ECGPCG0051', 'ECGPCG0053',
                                  'ECGPCG0057', 'ECGPCG0058', 'ECGPCG0063'],
    'laying on bed': ['ECGPCG0013', 'ECGPCG0014', 'ECGPCG0015', 'ECGPCG0016', 'ECGPCG0020',
                      'ECGPCG0021', 'ECGPCG0022', 'ECGPCG0023'],
    'sitting on armchair': ['ECGPCG0003', 'ECGPCG0004', 'ECGPCG0005', 'ECGPCG0006', 'ECGPCG0007',
                            'ECGPCG0008', 'ECGPCG0009', 'ECGPCG0010', 'ECGPCG0011', 'ECGPCG0012',
                            'ECGPCG0040']
}

GROUND_TRUTH = {
    '1': ['ECGPCG0059', 'ECGPCG0060', 'ECGPCG0061', 'ECGPCG0062', 'ECGPCG0064',
                'ECGPCG0065', 'ECGPCG0066', 'ECGPCG0067', 'ECGPCG0068', 'ECGPCG0069'],
    '2': ['ECGPCG0035', 'ECGPCG0036', 'ECGPCG0037', 'ECGPCG0038', 'ECGPCG0039',
                  'ECGPCG0046', 'ECGPCG0047', 'ECGPCG0052', 'ECGPCG0054', 'ECGPCG0055', 'ECGPCG0056'],
    '3': ['ECGPCG0001', 'ECGPCG0002', 'ECGPCG0024', 'ECGPCG0025', 'ECGPCG0026',
                 'ECGPCG0027', 'ECGPCG0028','ECGPCG0029','ECGPCG0030', 'ECGPCG0031',
                 'ECGPCG0032','ECGPCG0033', 'ECGPCG0034'],
    '4': ['ECGPCG0041', 'ECGPCG0042', 'ECGPCG0043', 'ECGPCG0044', 'ECGPCG0045',
                                  'ECGPCG0048', 'ECGPCG0049', 'ECGPCG0050', 'ECGPCG0051', 'ECGPCG0053',
                                  'ECGPCG0057', 'ECGPCG0058', 'ECGPCG0063'],
    '5': ['ECGPCG0013', 'ECGPCG0014', 'ECGPCG0015', 'ECGPCG0016', 'ECGPCG0020',
                      'ECGPCG0021', 'ECGPCG0022', 'ECGPCG0023'],
    '6': ['ECGPCG0003', 'ECGPCG0004', 'ECGPCG0005', 'ECGPCG0006', 'ECGPCG0007',
                            'ECGPCG0008', 'ECGPCG0009', 'ECGPCG0010', 'ECGPCG0011', 'ECGPCG0012',
                            'ECGPCG0040']
}

LABEL_IDS = {"ECGPCG0059": 1, "ECGPCG0060": 1, "ECGPCG0061": 1, "ECGPCG0062": 1, "ECGPCG0064": 1, "ECGPCG0065": 1, "ECGPCG0066": 1, "ECGPCG0067": 1, "ECGPCG0068": 1, "ECGPCG0069": 1, "ECGPCG0035": 2, "ECGPCG0036": 2, "ECGPCG0037": 2, "ECGPCG0038": 2, "ECGPCG0039": 2, "ECGPCG0046": 2, "ECGPCG0047": 2, "ECGPCG0052": 2, "ECGPCG0054": 2, "ECGPCG0055": 2, "ECGPCG0056": 2, "ECGPCG0001": 3, "ECGPCG0002": 3, "ECGPCG0024": 3, "ECGPCG0025": 3, "ECGPCG0026": 3, "ECGPCG0027": 3, "ECGPCG0028": 3, "ECGPCG0029": 3, "ECGPCG0030": 3, "ECGPCG0031": 3, "ECGPCG0032": 3, "ECGPCG0033": 3, "ECGPCG0034": 3, "ECGPCG0041": 4, "ECGPCG0042": 4, "ECGPCG0043": 4, "ECGPCG0044": 4, "ECGPCG0045": 4, "ECGPCG0048": 4, "ECGPCG0049": 4, "ECGPCG0050": 4, "ECGPCG0051": 4, "ECGPCG0053": 4, "ECGPCG0057": 4, "ECGPCG0058": 4, "ECGPCG0063": 4, "ECGPCG0013": 5, "ECGPCG0014": 5, "ECGPCG0015": 5, "ECGPCG0016": 5, "ECGPCG0020": 5, "ECGPCG0021": 5, "ECGPCG0022": 5, "ECGPCG0023": 5, "ECGPCG0003": 6, "ECGPCG0004": 6, "ECGPCG0005": 6, "ECGPCG0006": 6, "ECGPCG0007": 6, "ECGPCG0008": 6, "ECGPCG0009": 6, "ECGPCG0010": 6, "ECGPCG0011": 6, "ECGPCG0012": 6, "ECGPCG0040": 6}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
model = 'alex'
num_epochs = 20
batch_size = 16
learning_rate = 0.001
optim_method = 'SGD'
best_accuracy = 0
best_epoch = 0
correct = None

def get_image_paths_and_labels(directory):
    image_paths = []
    image_labels = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                image_id = file.split('_')[0]
                for label, ids in GROUND_TRUTH.items():
                    if image_id in ids:
                        image_paths.append(os.path.join(root, file))
                        image_labels.append(label)
                        break
    return image_paths, image_labels

def load_process_data(directory='./output', test_size=0.1, random_state=42):

    image_paths, image_labels = get_image_paths_and_labels(directory)
    # print('Total labels: ', set(image_labels))
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, image_labels, test_size=test_size, random_state=random_state, stratify=image_labels
    )

    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    return X_train, X_test, y_train, y_test


class ECGPCGDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        if 'output' in image_path:
            image_cv = cv2.imread(image_path)
            image_cv_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_cv_rgb)
            if self.transform:
                image = self.transform(image)

            return image, int(self.labels[idx])

# Model initialization
if model.lower() == 'alexnet':
    from model import AlexNet
    model = AlexNet()
    image_size = 227
else:
    from model import VisionTransformer
    model = VisionTransformer()
    image_size = 256

model.to(device)

# Load and preprocess data
train_data, test_data, train_labels, test_labels = load_process_data()

# Create datasets and dataloaders
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ECGPCGDataset(train_data, train_labels, transform=transform)
test_dataset = ECGPCGDataset(test_data, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

accuracy = Accuracy(task='multiclass', num_classes=6).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()

if optim_method == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
else:
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=0.9, weight_decay=0.0005)

os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

if os.listdir(CHECKPOINT_FOLDER):

    # Evaluation
    latest_checkpoint = max((f for f in os.listdir(CHECKPOINT_FOLDER) if f.startswith('checkpoint_')),
                            key=lambda f: int(f.split('_')[1].split('.')[0]))
    latest_checkpoint = f'{CHECKPOINT_FOLDER}/{latest_checkpoint}'

    model.load_state_dict(torch.load(latest_checkpoint), strict=False)

    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:

            images, labels = images.to(device), labels.clone().detach().to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            accuracy.update(predicted, labels)

            best_accuracy = accuracy

    print(f'Accuracy on the test images: {accuracy.compute().item() * 100}%')
else:
    start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.clone().detach().to(device)

            # Forward
            outputs = model(images)
            labels= labels - 1

            loss = criterion(outputs, labels)
            print('Loss: ', loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if best_accuracy != 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}. Accuracy on previous epoch {best_accuracy}. The best accuracy {best_accuracy} at epoch {best_epoch}')
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}.')

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
        }, f'{CHECKPOINT_FOLDER}/checkpoint_{epoch+1}.pth')

        # Validation
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:

                images, labels = images.to(device), labels.clone().detach().to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                accuracy.update(predicted, labels)

        print(f'Accuracy on the test images: {accuracy.compute().item() * 100}%')

        if accuracy.compute().item() > best_accuracy:
            best_accuracy = accuracy.compute().item()
            best_epoch = epoch + 1

    end_time = time.time()
    training_time = (end_time - start_time) / 60
    print(f'Training time: {training_time:.2f} minutes')

print(f'\nBest Accuracy: {best_accuracy}% at epoch {best_epoch}.')