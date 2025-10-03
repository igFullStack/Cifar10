import torch
from models.model import CNN
from utils.data_loader import get_data_loaders
import matplotlib.pyplot as plt
import numpy as np

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Загружаем модель
    model = CNN()
    model.load_state_dict(torch.load('outputs/trained_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    # Загружаем данные
    _, test_loader = get_data_loaders(batch_size=1000)
    
    # Классы CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Оцениваем на всем тестовом наборе
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
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Покажем несколько примеров с предсказаниями
    show_predictions(model, test_loader, device, classes)

def show_predictions(model, test_loader, device, classes, num_examples=12):
    """Показывает примеры изображений с предсказаниями"""
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images[:num_examples])
        _, predicted = outputs.max(1)
    
    # Денормализуем изображения для отображения
    images = images.cpu().numpy()
    images = images / 2 + 0.5  # unnormalize
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for i, ax in enumerate(axes.flat):
        if i < num_examples:
            ax.imshow(np.transpose(images[i], (1, 2, 0)))
            ax.set_title(f'True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}', 
                        color='green' if predicted[i] == labels[i] else 'red')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    test_model()