import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os
import tqdm
import time
import csv
import matplotlib.pyplot as plt

# --------- Initialize Parameters -------------
num_classes = 16# dataset includes 16 crop classes
epochs = 30
learning_rate = 0.001
batch_size = 64
weight_decay = 0.0005
model_name = "transformer"
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# --------- Select CPU or GPU to run ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#--------- Check the name of the GPU --------
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Load the processed data
X_train = torch.load("train_X.pt")
y_train = torch.load("train_y.pt")
X_val = torch.load("val_X.pt")
y_val = torch.load("val_y.pt")
X_test = torch.load("test_X.pt")
y_test = torch.load("test_y.pt")

# Create DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

# Transformer model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_classes, num_heads, num_layers, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)       # [B, T, input_dim] -> [B, T, model_dim]
        x = self.transformer_encoder(x)  # [B, T, model_dim]
        x = x.mean(dim=1)            # Mean over time
        return self.classifier(x)
    
# Model setup
input_dim = X_train.shape[2]
model_dim = 128
num_heads = 4
num_layers = 2
model = TimeSeriesTransformer(
    input_dim,
    model_dim,
    num_classes,
    num_heads,
    num_layers
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# print("Model OK")

# Training loop with validation
best_val_f1 = 0.0
best_model_state = None

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
train_f1s, val_f1s = [], []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    train_preds, train_targets = [], []
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
    for xb, yb in loop:
        xb, yb = xb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
        train_preds.extend(preds.cpu().numpy())
        train_targets.extend(yb.cpu().numpy())
        
        loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)
    
    scheduler.step()
    
    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)
    
    train_accuracy = 100 * correct / total
    train_accuracies.append(train_accuracy)
    
    train_f1 = f1_score(train_targets, train_preds, average='weighted', zero_division=0)*100
    train_f1s.append(train_f1)
    
    # Validation
    model.eval()
    val_loss = 0.0
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)
    
    val_accuracy = accuracy_score(y_true, y_pred) * 100
    val_accuracies.append(val_accuracy)
    
    val_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)*100
    val_f1s.append(val_f1)
    
    print(f"\nEpoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}, Train F1: {train_f1: .4f}%")
    print(f"\nEpoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}, Val F1: {val_f1: .4f}%\n")
    
    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_state = model.state_dict().copy()
        print("New best model saved")
        
# Save Model
model.load_state_dict(best_model_state)
torch.save(model.state_dict(), os.path.join(results_dir, f"{model_name}_trained.pth"))
print("Saved model")
    
# Evaluation function
start_time = time.time()
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        outputs = model(xb)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(yb.cpu().numpy())

inference_time = time.time() - start_time

acc = accuracy_score(all_targets, all_preds)
prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

print(f"Final Results: Accuracy: {acc: .4f}, Precision: {prec: .4f}, Recall: {rec: .4f}, F1-score: {f1: .4f}, Inference Time: {inference_time: .2f}s")

#--------------- Save Results to a csv file ------------------
csv_file = os.path.join(results_dir, "eval_results.csv")
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode="a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Inference_Time"])
    writer.writerow([model_name, acc, prec, rec, f1, inference_time])

print("Saved Results")

#---------------- Plotting -----------------
# Train/Val Loss
plt.figure(figsize=(12,8), num=1)
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(np.arange(0, epochs+1, 5))
plt.legend()
plt.grid()
plt.title('Loss Curve')
plt.savefig(os.path.join(results_dir, "loss_curve.png"))

# Train/Val Accuracy
plt.figure(figsize=(12,8), num=2)
plt.plot(range(1, epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.xticks(np.arange(0, epochs+1, 5))
plt.legend()
plt.grid()
plt.title('Accuracy Curve')
plt.savefig(os.path.join(results_dir, "accuracy_curve.png"))

# Train/Val F1-Score
plt.figure(figsize=(12,8), num=3)
plt.plot(range(1, epochs+1), train_f1s, label='Train F1-Score')
plt.plot(range(1, epochs+1), val_f1s, label='Validation F1-Core')
plt.xlabel('Epoch')
plt.ylabel('F1-Score (%)')
plt.xticks(np.arange(0, epochs+1, 5))
plt.legend()
plt.grid()
plt.title('F1-Score Curve')
plt.savefig(os.path.join(results_dir, "f1_score_curve.png"))

# Confusion Matrix
class_names = ['Legumes', 'Grassland', 'Maize', 'Potato', 'Sunflower', 'Soy', 'Winter Barley',
               'Winter Caraway', 'Rye', 'Rapeseed', 'Beet', 'Spring Cereals', 'Winter Wheat',
               'Winter Triticale', 'Permanent Plantation', 'Other Crops']
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(12,8), num=4)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(values_format='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xticks(rotation=90)
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
