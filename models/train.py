import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.data_loader import get_data_loaders
from models.model import CNN

def train_model(epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_data_loaders()
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    train_accs = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Обучение с progress bar
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

        # Сохраняем метрики для графиков
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

    print('Training Finished')
    # Здесь же можно добавить валидацию на test_loader и сохранение модели
    # torch.save(model.state_dict(), 'outputs/trained_model.pth')
    return model, train_losses, train_accs

if __name__ == '__main__':
    train_model()