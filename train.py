import argparse
import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader


def get_input_args():
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset")

    # Image  directory 
    parser.add_argument('data_dir', type=str, help="Path to the dataset directory with train, valid and test subfolders")

    # checkpoint directory argument
    parser.add_argument('--save_dir', type=str, default='checkpointFiles', help="Directory to save checkpoints")

    # CNN Architecture 
    parser.add_argument('--arch', type=str, default='vgg16', help="Choose architecture (e.g., vgg16)")

    # Hyperparameters 
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for training")
    parser.add_argument('--hidden_units', type=int, default=512, help="Number of hidden units in the classifier")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs for training")

    # GPU
    parser.add_argument('--gpu', action='store_true', help="Use GPU for training if available")

    return parser.parse_args()

def main():
    # Get arguments
    args = get_input_args()
    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'

    train_transforms = transforms.Compose([
    transforms.RandomRotation(30),     
    transforms.RandomResizedCrop(224),  
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),              
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
    ])
    valid_test_transforms = transforms.Compose([
    transforms.Resize(256),           
    transforms.CenterCrop(224),        
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=valid_test_transforms)
    
    batch_size = 64  
    
    # data loaders for training, validation, and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Print 
    print(f"Number of batches in training loader: {len(train_loader)}")
    print(f"Total number of samples in training loader: {len(train_loader.dataset)}")

    print(f"Number of batches in validation loader: {len(valid_loader)}")
    print(f"Total number of samples in validation loader: {len(valid_loader.dataset)}")

    print(f"Number of batches in test loader: {len(test_loader)}")
    print(f"Total number of samples in test loader: {len(test_loader.dataset)}")

    # Iterate to confirm the loading
    for inputs, labels in train_loader:
        print("Batch of inputs shape:", inputs.shape)
        print("Batch of labels shape:", labels.shape)
        break  # Print the first batch 
  
    
    # Check for GPU 
    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")

    
    # Load  pre-trained model
    model = getattr(models, args.arch)(pretrained=True)

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define a classifier
    classifier_input_size = model.classifier[0].in_features
    classifier = nn.Sequential(
        nn.Linear(classifier_input_size, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, len(train_dataset.classes)),  
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Move model to the appropriate device
    model.to(device)
    # Rubric : train.py successfully trains a new network on a dataset of images and saves the model to a checkpoint
    # Training loop
    print("Training started...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            print("Training")
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        # Rubric: The training loss, validation loss, and validation accuracy are printed out as a network trains
        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {running_loss / len(train_loader):.4f}")
        
    # validate  model
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    print(f"Validation Accuracy: {accuracy/len(valid_loader)}")
    # Save to checkpoint
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {
        'arch': args.arch,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    checkpoint_path = f'{args.save_dir}/{args.arch}_checkpoint.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved as {checkpoint_path}")

if __name__ == "__main__":
    main()
