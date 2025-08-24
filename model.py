import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for CIFAR-10 classification
    
    Architecture:
    - 3 Convolutional layers with ReLU activation and MaxPooling
    - 2 Fully connected layers
    - Dropout for regularization
    """
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward pass del modelo
        
        Args:
            x (torch.Tensor): Input tensor de shape (batch_size, 3, 32, 32)
            
        Returns:
            torch.Tensor: Output logits de shape (batch_size, num_classes)
        """
        x = self.pool1(F.relu(self.conv1(x)))
        
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x
    
    def get_model_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        summary = f"""
        Simple CNN Model Summary:
        ========================
        Conv1: 3 -> 32 channels (3x3 kernel)
        Conv2: 32 -> 64 channels (3x3 kernel)  
        Conv3: 64 -> 128 channels (3x3 kernel)
        FC1: 512 -> 256
        FC2: 256 -> 10 (output classes)
        
        Total parameters: {total_params:,}
        Trainable parameters: {trainable_params:,}
        """
        return summary


if __name__ == "__main__":
    model = SimpleCNN()
    print(model.get_model_summary())
    
    dummy_input = torch.randn(4, 3, 32, 32)  
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")