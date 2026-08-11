# Import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Google Colab upload
from google.colab import files

uploaded = files.upload()

# Get uploaded filename
filename = list(uploaded.keys())[0]

print("Uploaded image:", filename)

# Read image
img = cv2.imread(filename)

# Check image
if img is None:
    print("Error: Image could not be loaded.")
else:

    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Otsu thresholding
    ret, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV +
        cv2.THRESH_OTSU
    )

    # Create kernel
    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    # Morphological closing
    closing = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Dilation
    sure_bg = cv2.dilate(
        closing,
        kernel,
        iterations=3
    )

    # Display results
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(rgb_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(thresh, cmap="gray")
    plt.title("Otsu Threshold")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(closing, cmap="gray")
    plt.title("Morphological Closing")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(sure_bg, cmap="gray")
    plt.title("Dilation")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
