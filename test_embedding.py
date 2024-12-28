import os
import torch
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from reformer.reformer_pytorch import ViRWithArcMargin
from utils.datasets import get_lpfw_dataloaders
import argparse
from sklearn.preprocessing import LabelEncoder


def get_features(model, test_loader, batch_size=10):
    features = []
    model.eval()
    with torch.no_grad():
        for images, label in test_loader:
            images = images.cuda()  # Move images to GPU
            output = model.extract_features(images)  # Get feature embeddings directly
            output = output.data.cpu().numpy()
            features.append(output)
    
    return np.vstack(features)  # Stack features into a single array


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


def get_feature_dict(test_loader, features):
    fe_dict = {}
    for i, (images, _) in enumerate(test_loader):
        for j, image_tensor in enumerate(images):
            fe_dict[f"image_{i}_{j}"] = features[i][j]  # Unique key for each image
    return fe_dict


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


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosine_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, test_loader, pair_list, batch_size):
    s = time.time()
    features = get_features(model, test_loader, batch_size=batch_size)
    print(features.shape)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / len(test_loader)))
    fe_dict = get_feature_dict(test_loader, features)
    acc, th = test_performance(fe_dict, pair_list)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc


def generate_pair_list(test_loader, num_pairs=6000):
    """
    Generate a pair list for LFW testing from the test loader.

    :param test_loader: DataLoader object containing the images and labels.
    :param num_pairs: Number of pairs to generate for testing.
    :return: List of pairs, where each pair is a tuple (image1, image2, label).
    """
    # Collect image tensors and labels from DataLoader
    image_tensors = []
    labels = []

    for images, label in test_loader:
        image_tensors.extend(images)
        labels.extend(label)

    # Group image tensors by label
    label_to_images = {}
    for i, label in enumerate(labels):
        if label not in label_to_images:
            label_to_images[label] = []
        label_to_images[label].append(image_tensors[i])

    # Create pairs (positive and negative)
    pairs = []

    # Create positive pairs (same label)
    for label, img_list in label_to_images.items():
        if len(img_list) < 2:
            continue  # Skip if not enough images for positive pairs

        # Create positive pairs (same identity)
        for i in range(len(img_list)):
            for j in range(i+1, len(img_list)):
                pairs.append((img_list[i], img_list[j], 1))  # Positive pair with label 1

    # Create negative pairs (different labels)
    all_labels = list(label_to_images.keys())
    while len(pairs) < num_pairs:
        label1 = random.choice(all_labels)
        label2 = random.choice(all_labels)
        if label1 == label2:
            continue  # Skip if the labels are the same (we need negative pairs)

        img1 = random.choice(label_to_images[label1])
        img2 = random.choice(label_to_images[label2])
        pairs.append((img1, img2, 0))  # Negative pair with label 0

    return pairs


def save_pair_list(pair_list, file_path):
    """
    Save the generated pair list to a file.

    :param pair_list: List of pairs to save.
    :param file_path: File path to save the pair list.
    """
    with open(file_path, 'w') as f:
        for img1, img2, label in pair_list:
            f.write(f"{img1} {img2} {label}\n")


def visualize_tsne(features, labels, filename='tsne_plot.png', max_samples=1000):
    """
    Visualize the feature embeddings using t-SNE and save it as an image file.

    :param features: Feature embeddings to visualize.
    :param labels: Corresponding labels for the features.
    :param filename: The file name to save the figure.
    :param max_samples: Maximum number of samples to visualize for large datasets.
    """
    # Optionally, downsample the data if it's too large to visualize
    if len(features) > max_samples:
        sample_indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[sample_indices]
        labels = labels[sample_indices]

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)

    # Encode labels into integers for coloring
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    # Create the t-SNE plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=encoded_labels, cmap='jet', s=50)
    plt.colorbar(scatter, label='Class label')
    plt.title('t-SNE Visualization of Face Embeddings')
    plt.savefig(filename)
    plt.close()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a face recognition model on LFW")
    parser.add_argument('--test_model_path', type=str, required=True, help="Path to the pretrained model (.pth file)")
    parser.add_argument('--lfw_pair_list', type=str, required=True, help="Path to save the generated LFW pair list")
    parser.add_argument('--test_batch_size', type=int, default=10, help="Batch size for testing")

    args = parser.parse_args()

    # Initialize the ViRWithArcMargin model
    model = ViRWithArcMargin(image_size=224, patch_size=32, bucket_size=5, num_classes=5749, dim=256, depth=12, heads=8, num_mem_kv=0)
    load_model(model, args.test_model_path)  # Load the pretrained model weights
    model.to(torch.device("cuda"))  # Move the model to GPU

    _, test_loader = get_lpfw_dataloaders(batch_size=args.test_batch_size)  # Use the imported function here

    # Generate the pair list if it doesn't exist
    pair_list = generate_pair_list(test_loader)
    save_pair_list(pair_list, args.lfw_pair_list)  # Save the pair list to the file

    model.eval()  # Set the model to evaluation mode
    features = get_features(model, test_loader, batch_size=args.test_batch_size)

    # Visualize the features using t-SNE
    visualize_tsne(features, np.zeros(features.shape[0]), 'hasil/tsne.jpg')  # Placeholder labels as we don't have label info here for visualization

    lfw_test(model, test_loader, args.lfw_pair_list, args.test_batch_size)


if __name__ == '__main__':
    main()