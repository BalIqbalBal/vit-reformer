import os
import torch
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import argparse

from torchvision import transforms
from reformer.reformer_pytorch import ViRWithArcMargin

# Define the transformations for the dataset
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_features_from_images(model, image_paths):
    features = []
    model.eval()
    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            image = test_transform(image).unsqueeze(0).cuda()  # Transform and move to GPU
            output = model.extract_features(image)  # Get feature embeddings directly
            output = output.data.cpu().numpy()
            features.append(output[0])

    return np.array(features)  # Return as a numpy array

def load_model(model, model_path):
    """
    Load pretrained model weights.
    """
    print(f"Loading model from {model_path}")
    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()

    # Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)  # Update the model with pretrained weights
    model.load_state_dict(model_dict)

def cosine_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)

def test_performance(features, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        idx1, idx2, label = int(splits[0]), int(splits[1]), int(splits[2])
        fe_1 = features[idx1]
        fe_2 = features[idx2]
        sim = cosine_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th

def lfw_test(model, image_paths, pair_list):
    s = time.time()
    features = get_features_from_images(model, image_paths)
    print(features.shape)
    t = time.time() - s
    print('Total time is {}, average time is {}'.format(t, t / len(image_paths)))
    acc, th = test_performance(features, pair_list)
    print('LFW face verification accuracy: ', acc, 'threshold: ', th)
    return acc

def generate_lfw_pair_list(dataset_root, pair_file, num_pairs=6000):
    """
    Generate a pair list for LFW testing.

    :param dataset_root: Root directory of the LFW dataset.
    :param pair_file: File to save the generated pair list.
    :param num_pairs: Number of pairs to generate.
    """
    label_to_images = {}

    # Collect image file paths by label
    for root, _, files in os.walk(dataset_root):
        for file in files:
            if file.endswith('.jpg'):
                label = os.path.basename(root)
                if label not in label_to_images:
                    label_to_images[label] = []
                label_to_images[label].append(os.path.join(root, file))

    pairs = []

    # Generate positive pairs
    for label, image_paths in label_to_images.items():
        if len(image_paths) < 2:
            continue
        for i in range(len(image_paths)):
            for j in range(i + 1, len(image_paths)):
                pairs.append((image_paths[i], image_paths[j], 1))

    # Generate negative pairs
    labels = list(label_to_images.keys())
    while len(pairs) < num_pairs:
        label1 = random.choice(labels)
        label2 = random.choice(labels)
        if label1 == label2:
            continue
        img1 = random.choice(label_to_images[label1])
        img2 = random.choice(label_to_images[label2])
        pairs.append((img1, img2, 0))

    # Save pairs to file
    with open(pair_file, 'w') as f:
        for img1, img2, label in pairs:
            f.write(f"{img1} {img2} {label}\n")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a face recognition model on LFW")
    parser.add_argument('--test_model_path', type=str, required=True, help="Path to the pretrained model (.pth file)")
    parser.add_argument('--dataset_root', type=str, required=True, help="Root directory for the LFW dataset")
    parser.add_argument('--pair_file', type=str, required=True, help="File to save the generated pair list")

    args = parser.parse_args()

    # Initialize the ViRWithArcMargin model
    model = ViRWithArcMargin(image_size=224, patch_size=32, bucket_size=5, num_classes=5749, dim=256, depth=12, heads=8, num_mem_kv=0)
    load_model(model, args.test_model_path)  # Load the pretrained model weights
    model.to(torch.device("cuda"))  # Move the model to GPU

    # Generate the pair list
    generate_lfw_pair_list(args.dataset_root, args.pair_file)

    # Collect all image paths
    image_paths = []
    for root, _, files in os.walk(args.dataset_root):
        for file in files:
            if file.endswith('.jpg'):
                image_paths.append(os.path.join(root, file))

    # Perform LFW test
    lfw_test(model, image_paths, args.pair_file)

if __name__ == '__main__':
    main()
