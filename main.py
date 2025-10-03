import sys
import os

# Добавляем пути для импортов
sys.path.append('./models')
sys.path.append('./utils')

from models.train import train_model
from models.test import test_model

def main():
    print("=== CIFAR-10 CNN Classifier ===")
    print("1. Train model")
    print("2. Test model")
    print("3. Train and test")
    
    choice = input("Choose option (1/2/3): ").strip()
    
    if choice == '1':
        print("\nStarting training...")
        train_model(epochs=15)
        print("Training completed! Check 'outputs' folder.")
        
    elif choice == '2':
        print("\nStarting testing...")
        test_model()
        
    elif choice == '3':
        print("\nStarting training...")
        train_model(epochs=15)
        print("\nStarting testing...")
        test_model()
        
    else:
        print("Invalid choice!")

if __name__ == '__main__':
    main()