import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast # For mixed precision

from cifar10_model import SimpleBitNetCNN # Make sure this import works based on your file structure

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 128 # Can be larger if GPU memory allows
    learning_rate = 1e-3 # May need tuning for BitNet-like models
    num_epochs = 20 # Increase for better results

    # CIFAR-10 Data Loading and Augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size*2, shuffle=False, num_workers=4, pin_memory=True)

    # Model, Loss, Optimizer
    model = SimpleBitNetCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01) # AdamW often preferred
    
    # Optional: Learning rate scheduler
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Mixed precision training scaler
    scaler = GradScaler(enabled=torch.cuda.is_available()) # Only enable if CUDA is used

    print(f"Model: {model}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Mixed precision context
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], "
                      f"Loss: {loss.item():.4f}, Train Acc: {(100*correct_train/total_train):.2f}%")
        
        train_accuracy = 100 * correct_train / total_train
        epoch_loss = running_loss / len(trainloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Avg Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # if scheduler:
        #     scheduler.step()

        # Evaluation Loop
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast(enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct_test / total_test
        print(f"Test Accuracy after Epoch {epoch+1}: {test_accuracy:.2f}%")
        print("-" * 30)

    print("Finished Training")

    # Final evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast(enabled=torch.cuda.is_available()):
                 outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Final accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')

if __name__ == '__main__':
    main()
