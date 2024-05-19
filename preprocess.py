import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# Function to extract GLCM features from an image
def extract_features(image):
    # Calculate GLCM
    glcm = graycomatrix(image, distances=[2], angles=[2 * np.pi/4], levels=256,
                        symmetric=True, normed=True)

    # Extract GLCM features
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]

    # Extract edge features
    edges = cv2.Canny(image, threshold1=30, threshold2=70)
    edge_mean = np.mean(edges)
    edge_std = np.std(edges)
    edge_max = np.max(edges)
    edge_min = np.min(edges)
    edge_skewness = np.mean((edges - np.mean(edges)) ** 3) / np.mean((edges - np.mean(edges)) ** 2) ** (3 / 2)
    edge_kurtosis = np.mean((edges - np.mean(edges)) ** 4) / np.mean((edges - np.mean(edges)) ** 2) ** 2
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])

    all_features = [contrast, energy, homogeneity, correlation, dissimilarity, edge_mean, edge_std, edge_max, edge_min, edge_skewness, edge_kurtosis, edge_density]

    return all_features