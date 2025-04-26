import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import librosa as lb

from ToneDataset import ToneDataset
from ToneCNN import ToneCNN

from config import *
from train import train_model

def evaluate_model(model_path="tone_cnn_model.pth", test_dir="tone-perfect/test"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Training new model...")
        model = train_model()
    else:
        model = ToneCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
    
    test_dataset = ToneDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    if len(test_dataset) == 0:
        print(f"No test data found in {test_dir}")
        return
        
    all_labels = []
    all_predictions = []
    incorrect_samples = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(y.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            start_idx = batch_idx * test_loader.batch_size
            end_idx = min(start_idx + len(y), len(test_dataset.filenames))
            batch_filenames = test_dataset.filenames[start_idx:end_idx]
            
            for i in range(len(y)):
                if predicted[i] != y[i]:
                    incorrect_samples.append({
                        "filename": batch_filenames[i] if i < len(batch_filenames) else f"unknown-{i}",
                        "true": y[i].item(),
                        "predicted": predicted[i].item(),
                        "confidence": probabilities[i][predicted[i]].item()
                    })
    
    accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))
    
    unique_labels = np.unique(np.array(all_labels))
    unique_preds = np.unique(np.array(all_predictions))
    all_classes = sorted(set(unique_labels) | set(unique_preds))
    
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    print(f"Number of incorrect samples: {len(incorrect_samples)}/{len(test_dataset)}")
    
    if len(all_classes) > 1:
        cm = confusion_matrix(all_labels, all_predictions)
        
        tone_names = {
            0: "First Tone",
            1: "Second Tone",
            2: "Third Tone", 
            3: "Fourth Tone"
        }
        class_names = [tone_names.get(i, f"Unknown-{i}") for i in all_classes]
        
        report = classification_report(all_labels, all_predictions, 
                                      target_names=class_names)
        print(f"\nClassification Report:\n{report}")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    else:
        print(f"\nWARNING: Only one class ({all_classes[0]}) found in the data.")
        print("Cannot generate confusion matrix or classification report.")

evaluate_model()

if __name__ == "__main__":
    evaluate_model()