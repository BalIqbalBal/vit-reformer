import os
import torch
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from itertools import permutations


from torchvision import transforms
from torchvision.transforms import InterpolationMode
from reformer.reformer_pytorch import ViRWithArcMargin

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Define the transformations for the dataset
test_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def save_features(features, save_path):
    """
    Save extracted features to a file.
    
    Args:
        features (dict): Dictionary mapping file paths to feature vectors
        save_path (str): Path where features will be saved
    """
    # Convert features dictionary to a format suitable for saving
    paths = list(features.keys())
    feature_vectors = np.array([features[path] for path in paths])
    
    # Save both the feature vectors and the corresponding paths
    np.savez(save_path,
             features=feature_vectors,
             paths=paths)
    print(f"Features saved to {save_path}")

def load_features(load_path):
    """
    Load features from a saved file.
    
    Args:
        load_path (str): Path to the saved features file
    
    Returns:
        dict: Dictionary mapping file paths to feature vectors
    """
    data = np.load(load_path, allow_pickle=True)
    features = {}
    for path, feature_vector in zip(data['paths'], data['features']):
        features[str(path)] = feature_vector
    return features

def get_features_from_images(model, image_paths, save_path=None):
    """
    Extract features from images and return them in a dictionary.
    
    Args:
        model: The neural network model
        image_paths (list): List of image file paths
        save_path (str, optional): If provided, save features to this path
    
    Returns:
        dict: Dictionary mapping file paths to feature vectors
    """
    features = {}
    model.eval()
    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            image = test_transform(image).unsqueeze(0).cuda()
            output = model.extract_features(image)
            output = output.data.cpu().numpy()
            features[image_path] = output[0]

    if save_path:
        save_features(features, save_path)

    return features

def load_model(model, model_path):
    """
    Load pretrained model weights.
    
    Args:
        model: The neural network model
        model_path (str): Path to the pretrained weights file
    """
    print(f"Loading model from {model_path}")
    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()

    # Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)  # Update the model with pretrained weights
    model.load_state_dict(model_dict)

def cosine_metric(x1, x2):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        x1 (numpy.ndarray): First vector
        x2 (numpy.ndarray): Second vector
    
    Returns:
        float: Cosine similarity score
    """
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cal_accuracy(y_score, y_true):
    """
    Calculate accuracy and find the best threshold.
    
    Args:
        y_score (list): List of similarity scores
        y_true (list): List of true labels
    
    Returns:
        tuple: (best_accuracy, best_threshold)
    """
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
    """
    Test performance using cosine similarity between pairs of features.
    
    Args:
        features (dict): Dictionary mapping file paths to their feature vectors
        pair_list (str): Path to the file containing pairs to test
    
    Returns:
        tuple: (accuracy, threshold)
    """
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.strip().split()
        path1, path2, label = splits[0], splits[1], int(splits[2])
        
        # Get features using file paths as keys
        fe_1 = features[path1]
        fe_2 = features[path2]
        sim = cosine_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th

def generate_lfw_pair_list(dataset_root, pair_file, num_pairs=6000):
    """
    Generate a pair list for LFW testing.

    Args:
        dataset_root (str): Root directory of the LFW dataset
        pair_file (str): File to save the generated pair list
        num_pairs (int): Number of pairs to generate
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
    positive_pairs = []
    for label, image_paths in label_to_images.items():
        if len(image_paths) < 2:
            continue
        for i in range(len(image_paths)):
            for j in range(i + 1, len(image_paths)):
                positive_pairs.append((image_paths[i], image_paths[j], 1))

    # Limit the number of positive pairs to match num_pairs
    num_positive_pairs = min(len(positive_pairs), num_pairs // 2)  # Balance between positive and negative pairs
    pairs.extend(positive_pairs[:num_positive_pairs])

    # Generate negative pairs
    labels = list(label_to_images.keys())
    negative_label_combinations = list(permutations(labels, 2))
    random.shuffle(negative_label_combinations)

    # Add negative pairs to reach num_pairs
    while len(pairs) < num_pairs:
        for label1, label2 in negative_label_combinations:
            if len(pairs) >= num_pairs:
                break
            img1 = random.choice(label_to_images[label1])
            img2 = random.choice(label_to_images[label2])
            pairs.append((img1, img2, 0))

    # Save pairs to file
    with open(pair_file, 'w') as f:
        for img1, img2, label in pairs:
            f.write(f"{img1} {img2} {label}\n")

    print(f"Generated {len(pairs)} pairs, saved to {pair_file}")

def visualize_with_tsne(features, save_path=None):
    """
    Visualizes the feature embeddings using t-SNE with colors and labels based on person names.

    Args:
        features (dict): A dictionary mapping image paths to feature vectors
        save_path (str or None): If provided, saves the plot to this path. If None, the plot is shown
    """
    # Convert features dictionary to a list of feature vectors
    feature_vectors = list(features.values())
    file_paths = list(features.keys())
    
    # Extract person names from file paths
    person_names = []
    for path in file_paths:
        # Split path and get the directory name which is the person's name
        parts = path.split(os.sep)
        # Find the person name in the path (it's usually between lfw_funneled and the image name)
        for i, part in enumerate(parts):
            if part == "lfw_funneled" and i + 1 < len(parts):
                person_names.append(parts[i + 1])
                break
        else:
            # If lfw_funneled is not found, take the second to last component
            person_names.append(parts[-2])

    # Convert feature vectors to a NumPy array
    feature_vectors = np.array(feature_vectors)

    # Apply t-SNE to reduce dimensionality to 2D
    print("Applying t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(feature_vectors)

    # Get unique names and assign colors
    unique_names = list(set(person_names))
    num_colors = len(unique_names)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))
    name_to_color = dict(zip(unique_names, colors))

    # Create color array for scatter plot
    point_colors = [name_to_color[name] for name in person_names]

    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Plot points
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                         c=point_colors, 
                         s=50,
                         alpha=0.6)

    # Add title and labels
    plt.title("t-SNE Visualization of Face Feature Embeddings")
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")

    # Create legend for unique persons
    # To avoid too many labels, limit to top 20 persons with most images
    name_counts = {}
    for name in person_names:
        name_counts[name] = name_counts.get(name, 0) + 1
    
    # Sort by count and take top 20
    top_names = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    top_names = [name for name, _ in top_names]
    
    # Create legend handles for top names
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=name_to_color[name], 
                                label=name, markersize=10)
                      for name in top_names]
    
    # Add legend
    plt.legend(handles=legend_elements, 
              title="Top 20 People",
              bbox_to_anchor=(1.05, 1),
              loc='upper left',
              borderaxespad=0.)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_with_umap(features, num_people=5, random_seed=42, save_path=None):
    """
    Enhanced UMAP visualization with better person selection and filtering for high-dimensional features.

    Args:
        features (dict): A dictionary mapping image paths to feature vectors
        num_people (int): Number of random people to select for visualization
        random_seed (int): Random seed for reproducibility
        save_path (str or None): If provided, saves the plot to this path. If None, the plot is shown
    """
    np.random.seed(random_seed)
    
    # Extract and process names
    file_paths = list(features.keys())
    person_names = []
    for path in file_paths:
        parts = path.split(os.sep)
        for i, part in enumerate(parts):
            if part == "lfw_funneled" and i + 1 < len(parts):
                person_names.append(parts[i + 1])
                break
        else:
            person_names.append(parts[-2])

    # Get unique names and their counts
    name_counts = {}
    for name in person_names:
        name_counts[name] = name_counts.get(name, 0) + 1
    
    # Filter to include only people with multiple images
    min_images = 3  # Minimum number of images per person
    qualified_names = [name for name, count in name_counts.items() if count >= min_images]
    
    if len(qualified_names) < num_people:
        print(f"Warning: Only {len(qualified_names)} people have {min_images}+ images. Using all of them.")
        selected_names = qualified_names
    else:
        selected_names = np.random.choice(qualified_names, size=num_people, replace=False)
    
    # Filter features
    filtered_features = []
    filtered_names = []
    for path, feature in features.items():
        person_name = None
        parts = path.split(os.sep)
        for i, part in enumerate(parts):
            if part == "lfw_funneled" and i + 1 < len(parts):
                person_name = parts[i + 1]
                break
        else:
            person_name = parts[-2]
            
        if person_name in selected_names:
            filtered_features.append(feature)
            filtered_names.append(person_name)

    # Convert to numpy array and normalize
    feature_vectors = np.array(filtered_features)
    
    # Normalize features (important for dimensionality reduction)
    feature_vectors = feature_vectors / np.linalg.norm(feature_vectors, axis=1, keepdims=True)
    
    print("Applying UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced_features = reducer.fit_transform(feature_vectors)

    # Plotting
    plt.figure(figsize=(15, 10))
    
    # Create color scheme
    unique_filtered_names = list(set(filtered_names))
    num_colors = len(unique_filtered_names)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))
    name_to_color = dict(zip(unique_filtered_names, colors))
    point_colors = [name_to_color[name] for name in filtered_names]
    
    # Create scatter plot
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                         c=point_colors, 
                         s=100,  # Larger points for better visibility
                         alpha=0.7)  # Increased opacity

    plt.title(f"UMAP Visualization of Face Feature Embeddings\n({len(selected_names)} People)")
    plt.xlabel("UMAP component 1")
    plt.ylabel("UMAP component 2")

    # Sort by count and create legend
    selected_names_sorted = sorted(selected_names, key=lambda x: name_counts[x], reverse=True)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=name_to_color[name], 
                                label=f"{name} ({name_counts[name]} images)", 
                                markersize=10)
                      for name in selected_names_sorted]
    
    plt.legend(handles=legend_elements, 
              title="People (Image Count)",
              bbox_to_anchor=(1.05, 1),
              loc='upper left',
              borderaxespad=0.)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def lfw_test_with_features(model, image_paths, pair_list, feature_save_path=None, tsne_save_path=None):
    """
    Perform LFW test and optionally save features and t-SNE visualization.
    
    Args:
        model: The neural network model
        image_paths (list): List of image paths to process
        pair_list (str): Path to the file containing pairs to test
        feature_save_path (str, optional): Path to save extracted features
        tsne_save_path (str, optional): Path to save t-SNE visualization
    
    Returns:
        float: Accuracy score
    """
    s = time.time()
    features = get_features_from_images(model, image_paths, feature_save_path)
    print(f"Features extracted for {len(features)} images")
    t = time.time() - s
    print('Total time is {}, average time is {}'.format(t, t / len(image_paths)))

    if tsne_save_path:
        visualize_with_tsne(features, tsne_save_path)

    acc, th = test_performance(features, pair_list)
    print('LFW face verification accuracy: ', acc, 'threshold: ', th)
    return acc

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a face recognition model on LFW")
    parser.add_argument('--test_model_path', type=str, required=True, help="Path to the pretrained model (.pth file)")
    parser.add_argument('--dataset_root', type=str, required=True, help="Root directory for the LFW dataset")
    parser.add_argument('--pair_file', type=str, required=True, help="File to save the generated pair list")
    parser.add_argument('--tsne_save_path', type=str, required=True, help="File to save tsne_plot")
    parser.add_argument('--feature_save_path', type=str, help="Path to save extracted features")

    args = parser.parse_args()

    # Initialize model
    model = ViRWithArcMargin(
        image_size=224,
        patch_size=32,
        bucket_size=5,
        num_classes=5749,
        dim=256,
        depth=12,
        heads=8,
        num_mem_kv=0
    )
    
    # Load pretrained weights and move model to GPU
    load_model(model, args.test_model_path)
    model.to(torch.device("cuda"))

    # Generate pair list for testing
    generate_lfw_pair_list(args.dataset_root, args.pair_file)

    # Collect all image paths
    image_paths = []
    for root, _, files in os.walk(args.dataset_root):
        for file in files:
            if file.endswith('.jpg'):
                image_paths.append(os.path.join(root, file))

    # Perform LFW test
    lfw_test_with_features(
        model,
        image_paths,
        args.pair_file,
        args.feature_save_path,
        args.tsne_save_path
    )

if __name__ == '__main__':
    main()