# File Name: face_recognition.py

import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Helper Function for Plotting Image Grids ---
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Plots a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0.01, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        if i >= len(images):
            break
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=10)
        plt.xticks(())
        plt.yticks(())

def load_data(dataset_path):
    print("Loading dataset...")
    search_pattern = os.path.join(dataset_path, "**", "*.jpg")
    image_paths = glob.glob(search_pattern, recursive=True)
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {dataset_path}")

    # Read first image to get target dimensions
    first_img = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    h, w = first_img.shape
    
    X = []
    y = []
    
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # Auto-resize to match first image dimensions
        if img.shape != (h, w):
            img = cv2.resize(img, (w, h))
            
        X.append(img.flatten())
        label = os.path.basename(os.path.dirname(path))
        y.append(label)
        
    return np.array(X), np.array(y), h, w

def main(dataset_path):
    # 1. Load Data
    X, y, h, w = load_data(dataset_path)
    
    # 2. Split Data (60% Train, 40% Test to satisfy PDF requirements)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)
    
    # 3. Test multiple K values for the Accuracy Graph
    k_values_to_test = [5, 10, 15, 20, 30, 40, 50]
    # Ensure K isn't larger than our dataset size
    k_values_to_test = [k for k in k_values_to_test if k <= X_train.shape[0]] 
    accuracies = []

    print("\n--- Starting K-Value Evaluation ---")
    for k in k_values_to_test:
        # Train PCA
        pca = PCA(n_components=k, svd_solver='randomized', whiten=True).fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        # Train ANN (MLP)
        clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
        clf.fit(X_train_pca, y_train)
        
        # Predict & Score
        acc = accuracy_score(y_test, clf.predict(X_test_pca))
        accuracies.append(acc * 100)
        print(f"K = {k:2d} | Classification Accuracy: {acc * 100:.2f}%")

    # --- REQUIREMENT A: Plot the K-Value Accuracy Graph ---
    print("\nPlotting Accuracy Graph (Close the window to continue...)")
    plt.figure(figsize=(8, 5))
    plt.plot(k_values_to_test, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Classification Accuracy vs. K Value (PCA Components)')
    plt.xlabel('Number of Eigenvectors (k)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.show() # Halts script until closed

    # --- VISUAL OUTPUTS: Generate Grids using the highest K value (50) ---
    best_k = k_values_to_test[-1]
    print(f"\n--- Generating Visuals for K = {best_k} ---")
    pca = PCA(n_components=best_k, svd_solver='randomized', whiten=True).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Generate Eigenfaces
    eigenfaces = pca.components_.reshape((best_k, h, w))
    eigenface_titles = [f"eigenface {i}" for i in range(eigenfaces.shape[0])]
    
    print("Plotting Eigenfaces Gallery (Close the window to continue...)")
    plot_gallery(eigenfaces, eigenface_titles, h, w)
    plt.show()

    # Train final classifier
    print("Training final Multi-Layer Perceptron...")
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, verbose=True, random_state=42).fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    y_prob = clf.predict_proba(X_test_pca)
    
    # Generate Prediction Titles
    def title_generator(y_pred, y_test, y_prob, i):
        pred_name = y_pred[i]
        true_name = y_test[i]
        prob = np.max(y_prob[i]) 
        return f"pred: {pred_name}, pr: {prob:.2f}\ntrue: {true_name}"
        
    prediction_titles = [title_generator(y_pred, y_test, y_prob, i) for i in range(y_pred.shape[0])]
    
    print("Plotting Final Predictions...")
    plot_gallery(X_test, prediction_titles, h, w)
    plt.show()

if __name__ == "__main__":
    # Automatically locate the correct 'faces' folder
    current_script_folder = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIRECTORY = os.path.join(current_script_folder, "dataset", "faces")
    
    try:
        main(DATASET_DIRECTORY)
    except Exception as e:
        print(f"An error occurred: {e}")