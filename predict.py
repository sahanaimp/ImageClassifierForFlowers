import argparse
import json
import torch
import numpy as np
from torchvision import transforms, datasets, models
from PIL import Image


def get_input_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image")

    # Input image 
    parser.add_argument('input', type=str, help="Path to the input image")

    # Checkpoint
    parser.add_argument('checkpoint', type=str, help="Path to the checkpoint file")

    # Top K classes 
    parser.add_argument('--top_k', type=int, default=1, help="Return the top K most likely classes")

    # Category names 
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help="Path to the category names mapping JSON file")

    # GPU 
    parser.add_argument('--gpu', action='store_true', help="Use GPU for inference if available")

    return parser.parse_args()

#  load the model checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

#  process the input image
def process_image(image_path):
    img = Image.open(image_path)
    
    # Resize the image while preserving aspect ratio
    img.thumbnail((256, 256))
    
    # Center-crop the image to 224x224
    left_margin = (img.width - 224) / 2
    top_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    bottom_margin = top_margin + 224
    img = img.crop((left_margin, top_margin, right_margin, bottom_margin))
    
    # Convert PIL image to NumPy array
    np_image = np.array(img) / 255.0  # Normalize pixel values to 0-1 range
    
    # Normalize the image using mean and standard deviation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions to match PyTorch's expectation (color channel as the first dimension)
    np_image = np_image.transpose((2, 0, 1))
    
    # Convert NumPy array to a PyTorch tensor
    torch_image = torch.tensor(np_image, dtype=torch.float32)
    
    return torch_image
#  predict the top K classes and their probabilities
def predict(image_path, model, topk):
    model.eval()
    image = process_image(image_path).unsqueeze(0)

    if torch.cuda.is_available():
        image = image.cuda()

    with torch.no_grad():
        output = model(image)

    
    # Calculate the class probabilities
    probabilities = torch.exp(output)
    print(probabilities)
    
    # Get the top K probabilities and their corresponding class indices
    top_probs, top_indices = probabilities.topk(topk)
    
    return top_probs.cpu().numpy()[0], top_indices.cpu().numpy()[0]

#  convert class indices to class labels
def class_indices_to_labels(indices, model):
    class_to_idx = model.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    labels = [idx_to_class[i] for i in indices]
    return labels

#  map class labels to category names
def labels_to_category_names(labels, category_names_mapping):
    with open(category_names_mapping, 'r') as f:
        cat_to_name = json.load(f)

    category_names = [cat_to_name[label] for label in labels]
    return category_names


def main():
    # Get command-line arguments
    args = get_input_args()

    # Check for GPU 
    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")

    # Load  checkpoint
    model = load_checkpoint(args.checkpoint)
    model.to(device)

    # Predict the top K classes and their probabilities
    probabilities, indices = predict(args.input, model, args.top_k)

    # Convert class indices to class labels
    labels = class_indices_to_labels(indices, model)

    # Map class labels to category names
    category_names = labels_to_category_names(labels, args.category_names)
    

    # Print the top K predictions
    for i in range(args.top_k):
        print(f"Prediction {i + 1}: { category_names[i]} (Probability: {probabilities[i] * 100:.2f}%)")

if __name__ == "__main__":
    main()
