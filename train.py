import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os

from model import SimpleCNN
from utils import (
    show_sample_images, plot_training_history, show_predictions,
    calculate_accuracy, plot_confusion_matrix, get_classification_report,
    save_model, CIFAR10_CLASSES
)


def load_data(batch_size=32, num_workers=2):
    """
    Carga y prepara los datasets de CIFAR-10
    
    Args:
        batch_size (int): Tamaño del batch
        num_workers (int): Número de workers para DataLoader
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    print("Cargando datos CIFAR-10...")
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(10),            
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform_test
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers
    )
    
    print(f"Dataset cargado:")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Test samples: {len(test_dataset)}")
    print(f"- Classes: {len(CIFAR10_CLASSES)}")
    print(f"- Batch size: {batch_size}")
    
    return train_loader, test_loader


def train_model(model, train_loader, device, num_epochs=10, learning_rate=0.001):
    """
    Entrena el modelo
    
    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        device: Device (cuda/cpu)
        num_epochs (int): Número de épocas
        learning_rate (float): Learning rate
        
    Returns:
        tuple: (train_losses, train_accuracies)
    """
    print(f"\nIniciando entrenamiento...")
    print(f"Device: {device}")
    print(f"Épocas: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print("-" * 50)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predictions = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predictions == labels).sum().item()
            
            if (batch_idx + 1) % 200 == 0:
                batch_acc = 100 * correct_predictions / total_samples
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {batch_acc:.2f}%')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct_predictions / total_samples
        epoch_time = time.time() - start_time
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] COMPLETED:')
        print(f'  Average Loss: {epoch_loss:.4f}')
        print(f'  Accuracy: {epoch_acc:.2f}%')
        print(f'  Time: {epoch_time:.1f}s')
        print('-' * 50)
    
    print("¡Entrenamiento completado!")
    return train_losses, train_accuracies


def evaluate_model(model, test_loader, device):
    """
    Evalúa el modelo en el conjunto de test
    
    Args:
        model: Modelo entrenado
        test_loader: DataLoader de test
        device: Device (cuda/cpu)
    """
    print("\nEvaluando modelo en conjunto de test...")
    
    test_accuracy = calculate_accuracy(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    print("\nReporte de Clasificación:")
    print("=" * 50)
    report = get_classification_report(model, test_loader, device)
    print(report)
    
    print("\nGenerando matriz de confusión...")
    plot_confusion_matrix(model, test_loader, device)
    
    print("\nEjemplos de predicciones:")
    show_predictions(model, test_loader, device, num_samples=8)


def main():
    """Función principal"""
    print("="*60)
    print("CIFAR-10 Image Classification with PyTorch")
    print("="*60)
    
    
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    NUM_WORKERS = 2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    train_loader, test_loader = load_data(
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS
    )
    
    print("\nMostrando imágenes de ejemplo del dataset:")
    show_sample_images(train_loader)
    
    model = SimpleCNN(num_classes=10).to(device)
    print(f"\nModelo creado:")
    print(model.get_model_summary())
    
    train_losses, train_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
    print("\nMostrando historia del entrenamiento...")
    plot_training_history(train_losses, train_accuracies)
    
    evaluate_model(model, test_loader, device)
    
    model_dir = 'saved_models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'cifar10_cnn_model.pth')
    
    save_model(
        model=model,
        filepath=model_path,
        epoch=NUM_EPOCHS,
        train_loss=train_losses[-1],
        train_acc=train_accuracies[-1]
    )
    
    print(f"\n¡Proyecto completado exitosamente!")
    print(f"Accuracy final en test: {calculate_accuracy(model, test_loader, device):.2f}%")


if __name__ == "__main__":
    plt.ion() 
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nEntrenamiento interrumpido por el usuario.")
    except Exception as e:
        print(f"\nError durante la ejecución: {e}")
        raise
    finally:
        plt.ioff()  