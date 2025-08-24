import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
import sys

from model import SimpleCNN
from utils import (
    show_predictions, calculate_accuracy, plot_confusion_matrix,
    get_classification_report, CIFAR10_CLASSES, load_model
)
from train import load_data


def predict_single_image(model, image_path, device):
    """
    Predice la clase de una imagen individual
    
    Args:
        model: Modelo entrenado
        image_path (str): Ruta de la imagen
        device: Device (cuda/cpu)
    
    Returns:
        tuple: (predicted_class, confidence)
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0) 
        image_tensor = image_tensor.to(device)
        
        # Predicción
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = CIFAR10_CLASSES[predicted_idx.item()]
            confidence_score = confidence.item()
            
        return predicted_class, confidence_score
        
    except Exception as e:
        print(f"Error procesando imagen {image_path}: {e}")
        return None, None


def evaluate_saved_model(model_path, device):
    """
    Evalúa un modelo guardado en el conjunto de test
    
    Args:
        model_path (str): Ruta del modelo guardado
        device: Device (cuda/cpu)
    """
    print(f"Evaluando modelo: {model_path}")
    
    _, test_loader = load_data(batch_size=32, num_workers=2)
    
    model = SimpleCNN(num_classes=10).to(device)
    checkpoint = load_model(model, model_path, device)
    
    print(f"\nResumen del modelo:")
    print(model.get_model_summary())
    
    test_accuracy = calculate_accuracy(model, test_loader, device)
    print(f"\nTest Accuracy: {test_accuracy:.2f}%")
    
    print("\nReporte de Clasificación:")
    print("="*60)
    report = get_classification_report(model, test_loader, device)
    print(report)
    
    print("\nGenerando visualizaciones...")
    plot_confusion_matrix(model, test_loader, device)
    show_predictions(model, test_loader, device, num_samples=8)


def interactive_demo(model_path, device):
    """
    Demo interactivo para clasificar imágenes
    
    Args:
        model_path (str): Ruta del modelo guardado
        device: Device (cuda/cpu)
    """
    print("=== DEMO INTERACTIVO - CIFAR-10 CLASSIFIER ===")
    print("Cargando modelo...")
    
    model = SimpleCNN(num_classes=10).to(device)
    load_model(model, model_path, device)
    
    print("\nModelo cargado exitosamente!")
    print("Clases disponibles:", ", ".join(CIFAR10_CLASSES))
    print("\nIngresa la ruta de una imagen (o 'quit' para salir):")
    
    while True:
        try:
            image_path = input("\nRuta de imagen: ").strip()
            
            if image_path.lower() in ['quit', 'exit', 'q']:
                print("¡Hasta luego!")
                break
                
            if not os.path.exists(image_path):
                print("Archivo no encontrado. Intenta de nuevo.")
                continue
                
            # Predicción
            predicted_class, confidence = predict_single_image(
                model, image_path, device
            )
            
            if predicted_class is not None:
                print(f"Predicción: {predicted_class}")
                print(f"Confianza: {confidence:.2%}")
            else:
                print("Error procesando la imagen.")
                
        except KeyboardInterrupt:
            print("\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Cargar y usar modelo CIFAR-10 entrenado'
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        default='saved_models/cifar10_cnn_model.pth',
        help='Ruta del modelo guardado'
    )
    
    parser.add_argument(
        '--evaluate', 
        action='store_true',
        help='Evaluar modelo en conjunto de test'
    )
    
    parser.add_argument(
        '--demo', 
        action='store_true',
        help='Ejecutar demo interactivo'
    )
    
    parser.add_argument(
        '--image', 
        type=str,
        help='Ruta de imagen individual para clasificar'
    )
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(args.model_path):
        print(f"Error: Modelo no encontrado en {args.model_path}")
        print("¿Has entrenado el modelo con 'python train.py'?")
        sys.exit(1)
    
    try:
        if args.evaluate:
            evaluate_saved_model(args.model_path, device)
            
        elif args.image:
            model = SimpleCNN(num_classes=10).to(device)
            load_model(model, args.model_path, device)
            
            predicted_class, confidence = predict_single_image(
                model, args.image, device
            )
            
            if predicted_class is not None:
                print(f"Imagen: {args.image}")
                print(f"Predicción: {predicted_class}")
                print(f"Confianza: {confidence:.2%}")
            else:
                print("Error procesando la imagen.")
                
        elif args.demo:
            interactive_demo(args.model_path, device)
            
        else:
            print("Uso: python run.py [--evaluate] [--demo] [--image path]")
            print("Ejecutando evaluación por defecto...")
            evaluate_saved_model(args.model_path, device)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()