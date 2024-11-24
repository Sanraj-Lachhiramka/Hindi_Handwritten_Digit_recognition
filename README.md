# Handwritten Digit Recognition using K-Nearest Neighbors (KNN)

## Description
This project implements a Handwritten Digit Recognition system using the K-Nearest Neighbors (KNN) algorithm. The system uses the MNIST dataset, a widely-used dataset containing 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels in size. The model is trained to classify these digits with high accuracy.

The code also includes a user-friendly functionality to predict handwritten digits from user-provided image files. The user can provide an image, and the model will preprocess and classify the digit based on the trained KNN model.

---

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```

2. **Install dependencies:**
   Make sure you have Python 3.7+ installed. Install the required Python libraries by running:
   ```bash
   pip install numpy
   pip install pandas
   pip install matplotlib
   pip install Pillow
   pip install scikit-learn
   ```
  
3. **Dataset download:**
   The MNIST dataset is automatically fetched using the `fetch_openml` function from `scikit-learn`.

---

## Usage
### Train and Evaluate the Model
1. Run the main script to train the KNN model, evaluate its accuracy, and determine the best value of `k` using the elbow method:
   ```bash
   python main.py
   ```

2. The script will:
   - Load the MNIST dataset.
   - Split it into training and testing sets.
   - Find the optimal value of `k` using the elbow method.
   - Train the KNN classifier with the optimal `k`.
   - Evaluate the model using metrics such as accuracy and confusion matrix.

### Predict Custom Images
1. Prepare your custom handwritten digit image:
   - Ensure the image is grayscale.
   - Save the image in a file format like `.png` or `.jpg`.

2. Run the script and provide the image path when prompted:
   ```bash
   python main.py
   ```
   - Enter the path to your image file when prompted.

3. The script will preprocess the image and output the predicted digit.

---

## Example
### Output after training and testing:
- **Optimal `k`**: The elbow method determines the best value for `k`, for example: `k = 1`.
- **Accuracy**: The script outputs the accuracy of the model, e.g., `Accuracy with k = 1: 97.06%`.
- **Confusion Matrix**: A confusion matrix is displayed to show the classification performance.

### Output for a custom image:
- After preprocessing and prediction:
  ```
  Enter the path to image: path/to/your/image.png
  Predicted digit: 5
  ```

---

## Notes
- Ensure your custom images are clear and have a resolution similar to the MNIST dataset (28x28).
- Modify the script if needed to experiment with different values of `k` or preprocessing steps.
- For any questions or issues, feel free to open an issue in the repository.
