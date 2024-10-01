import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ChestXrayDataset, test_transform
from model import create_model
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test():
    # Create test dataset and dataloader
    test_dataset = ChestXrayDataset(root_dir='data/test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Load the trained model
    model = create_model().to(DEVICE)
    model.load_state_dict(torch.load('swin_transformer_chest_xray_1.pth'))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print classification report and confusion matrix
    print(classification_report(all_labels, all_preds, target_names=['normal', 'pneumonia']))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    test()