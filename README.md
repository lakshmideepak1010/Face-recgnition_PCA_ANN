# Face Recognition using PCA and ANN

This project implements a complete Face Recognition pipeline using Principal Component Analysis (PCA) for dimensionality reduction and an Artificial Neural Network (ANN) for classification. 

It was built to evaluate how different numbers of Principal Components ($k$-values) affect classification accuracy, and it includes visual outputs to demonstrate the extracted "Eigenfaces" and final predictions.

## 🚀 Features
* **Automated Data Loading:** Recursively loads and auto-resizes images from subdirectories, using the folder names as classification labels.
* **PCA & Eigenfaces:** Uses `scikit-learn` to perform PCA, extracting the most important features (Eigenfaces) from the dataset.
* **Multi-Layer Perceptron (ANN):** Trains a hidden-layer neural network to classify faces based on the reduced PCA feature set.
* **Evaluation & Visualizations:** * Automatically tests multiple $k$-values (e.g., 5, 10, 20, 50) and plots an Accuracy vs. K-Value line graph.
  * Generates a visual grid of the ghostly "Eigenfaces".
  * Generates a final test grid comparing True Labels vs. Predicted Labels alongside the network's confidence probability.

## 🛠️ Prerequisites
This script requires **Python 3.x** and the following libraries:
* `numpy`
* `opencv-python` (`cv2`)
* `matplotlib`
* `scikit-learn`

You can install all dependencies at once using pip:
```bash
pip install numpy opencv-python matplotlib scikit-learn

📂 Project Structure
To run this code, your directory must be structured like this so the script can dynamically locate your images:

Plaintext
├── README.md
├── face_recognition.py
└── dataset/
    └── faces/
        ├── Person_1/
        │   ├── img1.jpg
        │   └── img2.jpg
        ├── Person_2/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── ...
💻 How to Run
Ensure your dataset is extracted and placed in the correct dataset/faces folder hierarchy.

Open your terminal and navigate to the project directory.

Run the script:

Bash
python face_recognition.py
Note on Visuals: The script will pause execution to show you the generated Matplotlib graphs. You must close the graph window for the script to continue to the next phase of training!

📊 OutputsTerminal Logs: Displays the dataset loading status, the accuracy for each tested $k$-value, and the live iteration loss during the final ANN training.Accuracy Graph: A line graph showing how the number of PCA components impacts the final test accuracy.Eigenfaces Gallery: A visual representation of the features the machine is learning.Predictions Gallery: A grid of the test images showing how accurately the network predicted the unenrolled faces.
