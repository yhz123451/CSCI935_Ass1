import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

matplotlib.use("TkAgg")

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
import os
import sys
from skimage import feature as ft

# Function to display color channels (HSV or YCrCb) of an input image
def show_channel(img_path, color_space):
    img = cv2.imread(img_path)
    # Convert the image to the specified color space
    if color_space.upper() == 'HSV':
        image_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        titles = ['H', 'S', 'V']
    elif color_space.upper() == 'YCRCB':
        image_channel = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        titles = ['Y', 'Cr', 'Cb']
    else:
        print('Invalid color space')
        return

    # Split into individual channels
    c1, c2, c3 = cv2.split(image_channel)

    plt.figure(figsize=(12, 6))
    # Display original image (converted to RGB for matplotlib)
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    # Display each channel separately in grayscale
    plt.subplot(2, 2, 2)
    plt.imshow(c1, cmap="gray")
    plt.title(titles[0])
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(c2, cmap="gray")
    plt.title(titles[1])
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(c3, cmap="gray")
    plt.title(titles[2])
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Function to extract and visualize SIFT keypoints
def show_sift(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Create SIFT detector
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    output_image = img.copy()
    # Draw keypoints with cross, circle, and orientation line
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])  # coordinates of the keypoint
        size = int(kp.size)  # keypoint scale (size)
        angle = np.deg2rad(kp.angle)  # keypoint orientation (degrees to radians)

        # Draw circle around keypoint
        cv2.circle(output_image, (x, y), size, (0, 0, 255), 1)

        # Draw cross symbol
        cv2.line(output_image, (x - size, y), (x + size, y), (0, 255, 0), 1)
        cv2.line(output_image, (x, y - size), (x, y + size), (0, 255, 0), 1)

        # Draw orientation line
        end_x = int(x + size * np.cos(angle))
        end_y = int(y + size * np.sin(angle))
        cv2.line(output_image, (x, y), (end_x, end_y), (255, 0, 0), 1)

    img_org = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_output = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    # Display original and processed image side by side
    fig_w = 12
    plt.figure(figsize=(fig_w, fig_w * 0.38), dpi=120)
    plt.subplot(1, 2, 1)
    plt.imshow(img_org)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_output)
    plt.title("Image with overlapped keypoints")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Function to extract and visualize HOG features
def show_HOG(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Extract HOG features and visualization
    hog_features, hog_image = ft.hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True,
    )
    # Display original and HOG image
    fig_w = 12
    plt.figure(figsize=(fig_w, fig_w * 0.38), dpi=120)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(hog_image, cmap="gray")
    plt.title("HOG of the Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Function to list all images by class (subfolders of dataset)
def list_images_by_class(main_folder="Animals10"):
    list_images = {}
    for cls in os.listdir(main_folder):
        class_path = os.path.join(main_folder, cls)
        if not os.path.isdir(class_path):
            continue
        images = []
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(".jpg"):
                images.append(os.path.join(class_path, img_file))
        list_images[cls] = sorted(images)
    return list_images

# Function to extract HOG feature vector for a given image
def hog_features(img_path, size=(128, 128)):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    feat = ft.hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )
    return feat.astype(np.float32)

# Function to train and evaluate SVM classifier on HOG features
def svm(main_folder="Animals10", kernel="linear", C=1.0, runs=5):
    images_by_cls = list_images_by_class(main_folder)
    X, y = [], []
    label_map = {}
    # Create label map for classes
    for idx, cls in enumerate(images_by_cls.keys()):
        label_map[cls] = idx
    # Extract features and labels
    for cls, paths in images_by_cls.items():
        for p in paths:
            X.append(hog_features(p))
            y.append(label_map[cls])
    X = np.array(X)
    y = np.array(y)

    acc_list = []
    # Repeat classification for a given number of runs
    for r in range(runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=r
        )
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train SVM classifier
        clf = SVC(kernel=kernel, C=C)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        acc_list.append(acc)
        print(f"Run {r + 1}: Accuracy = {acc:.2f}% (kernel={kernel}, C={C})")

    # Print mean and standard deviation of accuracies
    print(f"\nMean = {np.mean(acc_list):.2f}% ± {np.std(acc_list):.2f}%")

# Main entry point to handle command-line arguments
if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "-c":
        show_channel(sys.argv[3], sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "-f":
        if sys.argv[2].upper() == "SIFT":
            show_sift(sys.argv[3])
        elif sys.argv[2].upper() == "HOG":
            show_HOG(sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "-r":
        main_folder = sys.argv[2]
        svm(main_folder, runs=5)
    else:
        # Usage guide for the program
        print("Usage:")
        print("python a1.py -c HSV|YCrCb <image-path>")
        print("python a1.py -f SIFT|HOG <image-path>")
        print("python a1.py -r <main-folder>")
