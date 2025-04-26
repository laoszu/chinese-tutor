import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ToneDataset import ToneDataset
from ToneCNN import ToneCNN

from config import *

def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dataset = ToneDataset(TRAIN_DIR)
    test_dataset = ToneDataset(TEST_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = ToneCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Test Loss: {test_loss/len(test_loader):.4f} | "
              f"Accuracy: {100*correct/total:.2f}%")
    
    torch.save(model.state_dict(), "tone_cnn_model.pth")
    return model

if __name__ == "__main__":
    train_model()