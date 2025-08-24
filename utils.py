import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def imshow(img: torch.Tensor, title: str = None):
    """
    Muestra una imagen de tensor de PyTorch
    
    Args:
        img (torch.Tensor): Imagen tensor de shape (C, H, W)
        title (str): Título opcional para la imagen
    """
    img = img / 2 + 0.5  
    npimg = img.numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def show_sample_images(dataloader, num_samples: int = 8):
    """
    Muestra imágenes de ejemplo del dataloader
    
    Args:
        dataloader: PyTorch DataLoader
        num_samples (int): Número de imágenes a mostrar
    """
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        img = images[i] / 2 + 0.5
        npimg = img.numpy().transpose((1, 2, 0))
        
        axes[i].imshow(npimg)
        axes[i].set_title(f'Label: {CIFAR10_CLASSES[labels[i]]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_training_history(train_losses: List[float], train_accuracies: List[float]):
    """
    Grafica la historia del entrenamiento
    
    Args:
        train_losses (List[float]): Lista de pérdidas por época
        train_accuracies (List[float]): Lista de accuracies por época
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_accuracies, 'r-', label='Training Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def show_predictions(model, dataloader, device, num_samples: int = 8):
    """
    Muestra predicciones del modelo vs etiquetas reales
    
    Args:
        model: Modelo PyTorch entrenado
        dataloader: DataLoader de test
        device: Device (cuda/cpu)
        num_samples (int): Número de predicciones a mostrar
    """
    model.eval()
    
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    images = images.cpu()
    labels = labels.cpu()
    predictions = predictions.cpu()
    
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        img = images[i] / 2 + 0.5
        npimg = img.numpy().transpose((1, 2, 0))
        
        is_correct = predictions[i] == labels[i]
        title_color = 'green' if is_correct else 'red'
        
        axes[i].imshow(npimg)
        axes[i].set_title(
            f'Real: {CIFAR10_CLASSES[labels[i]]}\n'
            f'Pred: {CIFAR10_CLASSES[predictions[i]]}',
            color=title_color,
            fontsize=10
        )
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def calculate_accuracy(model, dataloader, device) -> float:
    """
    Calcula la accuracy del modelo en un dataset
    
    Args:
        model: Modelo PyTorch
        dataloader: DataLoader del dataset
        device: Device (cuda/cpu)
        
    Returns:
        float: Accuracy como porcentaje
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def plot_confusion_matrix(model, dataloader, device):
    """
    Genera y muestra la matriz de confusión
    
    Args:
        model: Modelo PyTorch entrenado
        dataloader: DataLoader de test
        device: Device (cuda/cpu)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CIFAR10_CLASSES,
                yticklabels=CIFAR10_CLASSES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def get_classification_report(model, dataloader, device) -> str:
    """
    Genera un reporte de clasificación detallado
    
    Args:
        model: Modelo PyTorch entrenado
        dataloader: DataLoader de test
        device: Device (cuda/cpu)
        
    Returns:
        str: Reporte de clasificación
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    report = classification_report(
        all_labels, all_predictions,
        target_names=CIFAR10_CLASSES,
        digits=3
    )
    
    return report


def save_model(model, filepath: str, epoch: int, train_loss: float, train_acc: float):
    """
    Guarda el modelo y metadata de entrenamiento
    
    Args:
        model: Modelo PyTorch
        filepath (str): Ruta donde guardar el modelo
        epoch (int): Época actual
        train_loss (float): Pérdida de entrenamiento
        train_acc (float): Accuracy de entrenamiento
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'train_loss': train_loss,
        'train_accuracy': train_acc,
    }, filepath)
    print(f"Modelo guardado en: {filepath}")


def load_model(model, filepath: str, device):
    """
    Carga un modelo guardado
    
    Args:
        model: Instancia del modelo
        filepath (str): Ruta del modelo guardado
        device: Device donde cargar el modelo
        
    Returns:
        dict: Información del checkpoint cargado
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Modelo cargado desde: {filepath}")
    print(f"Época: {checkpoint['epoch']}")
    print(f"Train Loss: {checkpoint['train_loss']:.4f}")
    print(f"Train Accuracy: {checkpoint['train_accuracy']:.2f}%")
    
    return checkpoint