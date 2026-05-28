# File Name: face_recognition_pca_ann.py

import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_face_database(dataset_path):
    """Step 1: Generate the face database (with auto-resizing)"""
    search_pattern = os.path.join(dataset_path, "**", "*.jpg")
    image_paths = glob.glob(search_pattern, recursive=True)
    p = len(image_paths)
    
    if p == 0:
        raise ValueError(f"No images found. Check your dataset path: {dataset_path}")
        
    first_img = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    if first_img is None:
         raise ValueError(f"Failed to read image at {image_paths[0]}")
         
    # m = height (rows), n = width (columns)
    m, n = first_img.shape 
    
    Face_Db = np.zeros((m * n, p))
    labels = []
    
    for i, path in enumerate(image_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # --- THE FIX: Resize image if it doesn't match the first one ---
        if img.shape != (m, n):
            # Note: OpenCV resize requires (width, height) which is (n, m)
            img = cv2.resize(img, (n, m)) 
            
        Face_Db[:, i] = img.flatten()
        label = os.path.basename(os.path.dirname(path))
        labels.append(label)
        
    return Face_Db, np.array(labels), (m, n, p)

def evaluate_system(dataset_path):
    print("Loading dataset...")
    Face_Db, labels, dimensions = load_face_database(dataset_path)
    m, n, total_p = dimensions
    
    # --- REQUIREMENT B: Setup Imposters ---
    unique_labels = np.unique(labels)
    imposter_label = unique_labels[-1] # Let's pick the last person alphabetically as the imposter
    print(f"Assigning '{imposter_label}' as the UNENROLLED Imposter.")
    
    enrolled_mask = (labels != imposter_label)
    imposter_mask = (labels == imposter_label)
    
    Face_Db_enrolled = Face_Db[:, enrolled_mask]
    y_enrolled = labels[enrolled_mask]
    
    Face_Db_imposter = Face_Db[:, imposter_mask]
    p = Face_Db_enrolled.shape[1]
    
    # --- PCA: Steps 2 to 5 (Computed on Enrolled Database) ---
    print("Computing PCA on enrolled database...")
    M = np.mean(Face_Db_enrolled, axis=1).reshape(-1, 1)     # Step 2: Mean Calculation
    Delta = Face_Db_enrolled - M                             # Step 3: Mean Zero
    C = np.dot(Delta.T, Delta)                               # Step 4: Surrogate Covariance
    eigenvalues, eigenvectors = eigh(C)                      # Step 5: Eigen decomposition
    sorted_indices = np.argsort(eigenvalues)[::-1]
    
    # --- REQUIREMENT A: Test multiple K values ---
    k_values_to_test = [5, 10, 15, 20, 30, 40, 50]
    # Ensure we don't pick a K larger than our number of images
    k_values_to_test = [k for k in k_values_to_test if k <= p] 
    
    accuracies = []
    
    print("\n--- Starting K-Value Evaluation ---")
    for k_value in k_values_to_test:
        # Step 6: Find best directions (Psi)
        Psi = eigenvectors[:, sorted_indices[:k_value]] 
        
        # Step 7: Generate Eigenfaces (Phi)
        Phi = np.dot(Psi.T, Delta.T)
        
        # Step 8: Generate Signatures for Enrolled Faces
        omega = np.dot(Phi, Delta)
        X_features = omega.T 
        
        # --- Splitting Data (60% Train, 40% Test) ---
        X_train, X_test, y_train, y_test = train_test_split(X_features, y_enrolled, test_size=0.40, random_state=42)
        
        # Step 9: Train ANN
        ann_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
        ann_classifier.fit(X_train, y_train)
        
        # Evaluate standard test accuracy
        y_pred = ann_classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc * 100)
        
        print(f"K = {k_value} | Classification Accuracy: {acc * 100:.2f}%")
        
        # --- Testing the Imposter for this K value ---
        if k_value == k_values_to_test[-1]: # Only print imposter results for the last/best K
            print("\n--- Testing Imposter Recognition ---")
            
            # Step 1 & 2 (Testing Phase): Mean zero the imposter faces
            Delta_imposter = Face_Db_imposter - M
            
            # Step 3 (Testing Phase): Project test face to eigenfaces
            Omega_imposter = np.dot(Phi, Delta_imposter)
            X_imposter = Omega_imposter.T
            
            # Step 4 (Testing Phase): Use ANN predict_proba to catch imposters
            # predict_proba returns the confidence percentage for each known class
            probabilities = ann_classifier.predict_proba(X_imposter)
            max_confidences = np.max(probabilities, axis=1)
            
            # If the highest confidence is below a threshold (e.g., 60%), flag as unenrolled
            threshold = 0.60 
            imposters_caught = 0
            for i, conf in enumerate(max_confidences):
                if conf < threshold:
                    imposters_caught += 1
                
            print(f"Tested {len(max_confidences)} imposter images of '{imposter_label}'.")
            print(f"System successfully blocked {imposters_caught} out of {len(max_confidences)} unauthorized attempts (Confidence Threshold: {threshold*100}%).")
            print("------------------------------------\n")

    # --- Plotting the Graph (Requirement A) ---
    print("Generating Graph...")
    plt.figure(figsize=(8, 5))
    plt.plot(k_values_to_test, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Classification Accuracy vs. K Value')
    plt.xlabel('Number of Eigenvectors (k)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.show() # This will pop open a window showing your graph!

if __name__ == "__main__":
    # 1. Automatically find the exact folder this Python script is sitting in
    current_script_folder = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Build the path dynamically: (Current Folder) -> "dataset" -> "faces"
    DATASET_DIRECTORY = os.path.join(current_script_folder, "dataset", "faces")
    
    # 3. Print a debug message so we can see EXACTLY where it is looking
    print("\n--- Path Debugging ---")
    print(f"Script location: {current_script_folder}")
    print(f"Target dataset path: {DATASET_DIRECTORY}")
    print("----------------------\n")
    
    K_DIMENSIONS = 20 # You can change this k value as per your assignment requirements
    
    try:
        evaluate_system(DATASET_DIRECTORY)
    except Exception as e:
        print(f"An error occurred: {e}")