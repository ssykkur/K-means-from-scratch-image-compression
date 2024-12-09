import numpy as np
import matplotlib.pyplot as plt
from utils import *


def process_data(original_img, png=False):
    X_img = np.reshape(original_img, (original_img.shape[0]*original_img.shape[1], 3))
    if not png:
        X_img = X_img/255
    print("X_img shape is:", X_img.shape)
    return X_img


def init_centroids(X, K):
    # Shuffle training examples
    randidx = np.random.permutation(X.shape[0])
    # Set first K examples as centroids
    centroids = X[randidx[:K]]
    return centroids


def find_closest_centroids(X, centroids):

    K = centroids.shape[0]
    m = X.shape[0]

    idx = np.zeros(m)
    distance = np.zeros((m, K))
    for i in range(m):
        for j in range(K):
            distance[i, j] = np.linalg.norm(X[i] - centroids[j])

    # Choose the index of the smallest distance that corresponds to a centroid
    idx = np.argmin(distance, axis=1)
    return idx


def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))

    # get idices of the training examples corresponding to each centroid
    for j in range(K):
        indices = np.where(idx==j)
        centroid_indices = X[indices]
        # calculate means across columns = adjusted centroid
        centroids[j] = np.mean(centroid_indices, axis=0)

    return centroids


def run_kMeans(X, initial_centroids, max_iters=10):

    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)

    for i in range(max_iters):
        
        print(f"K-Means iteration {i}/{max_iters-1}")
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
     
    return centroids, idx


# compares the original and compressed images, saves the new one
def plot_orig_vs_comp(original_img, comp_img):
    # original image
    fix, ax = plt.subplots(1,2, figsize=(10,10))
    plt.axis('off')

    ax[0].imshow(original_img)
    ax[0].set_title('Original')
    ax[0].set_axis_off()

    # compressed one
    ax[1].imshow(comp_img)
    ax[1].set_title(f'Compressed with {K} colours')
    ax[1].set_axis_off()

    plt.show()

    f = comp_img 
    f = open('compressed_image.png', 'w')


if __name__ == "__main__":
    
    original_img = plt.imread('hg.jpg')
    K = 16
    max_iters = 10
   
    X_img = process_data(original_img)
    initial_centroids = init_centroids(X_img, 16)
    centroids, idx = run_kMeans(X_img, initial_centroids, max_iters)

    idx_c = find_closest_centroids(X_img, centroids)
    X_img_compressed = centroids[idx, :]
    X_img_compressed = np.reshape(X_img_compressed, original_img.shape)

    plot_orig_vs_comp(original_img, X_img_compressed)

 
