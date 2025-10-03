import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from utils.data_loader import get_data_loaders
from models.model import CNN

def train_model(epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Создаем папку для сохранения результатов
    os.makedirs('outputs', exist_ok=True)

    train_loader, test_loader = get_data_loaders()
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        # ===== TRAINING =====
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (data, targets) in enumerate(loop):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Обновляем progress bar
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ===== VALIDATION =====
        test_acc = evaluate_model(model, test_loader, device)
        test_accs.append(test_acc)

        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

    # Сохраняем модель
    torch.save(model.state_dict(), 'outputs/trained_model.pth')
    
    # Строим графики
    plot_training_curves(train_losses, train_accs, test_accs)
    
    return model, train_losses, train_accs, test_accs

def evaluate_model(model, test_loader, device):
    """Функция для оценки модели на тестовых данных"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def plot_training_curves(train_losses, train_accs, test_accs):
    """Визуализация процесса обучения"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../outputs/training_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    model, train_losses, train_accs, test_accs = train_model(epochs=15)