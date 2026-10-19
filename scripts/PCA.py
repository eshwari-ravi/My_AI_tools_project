from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
import math
import random


def load_and_center_dataset(filename):
    """
    Load dataset from .npy file and center it by subtracting the mean.

    Args:
        filename (str): Path to the .npy file

    Returns:
        numpy.ndarray: Centered dataset (n x d matrix)
    """
    data = np.load(filename)
    mean = np.mean(data, axis = 0)
    centered_data = data - mean
    """print(len(centered_data))
    print(len(centered_data[0]))"""
    return centered_data

    raise NotImplementedError
#x = load_and_center_dataset("celeba_60x50.npy")

def get_covariance(dataset):
    """
    Calculate the sample covariance matrix of the dataset.

    Args:
        dataset (numpy.ndarray): Centered dataset (n x d matrix)

    Returns:
        numpy.ndarray: Covariance matrix (d x d matrix)
    """
    x = np.array(dataset)
    y= np.transpose(x)
    n = x.shape[0]
    cov_data = (1 /(n-1)) * np.dot(y,x)
    """print("The size of the covariance matrix is:", cov_data.shape)
    print("The size of the covariance matrix is:", cov_data.size)"""
    return cov_data
    
    raise NotImplementedError
#z = get_covariance(x)

def get_eig(S, k):
    """
    Get the k largest eigenvalues and corresponding eigenvectors.

    Args:
        S (numpy.ndarray): Covariance matrix (d x d)
        k (int): Number of largest eigenvalues/eigenvectors to return

    Returns:
        tuple: (Lambda, U) where Lambda is diagonal matrix of eigenvalues
               and U is matrix of corresponding eigenvectors as columns
    """
    d = S.shape[0]
    
    start_idx = d - k
    end_idx = d - 1
    evals_subset, evecs_subset = eigh(S, subset_by_index=(start_idx, end_idx))
    evals_descending = evals_subset[::-1]
    U = evecs_subset[:, ::-1]
    Lambda = np.diag(evals_descending)
    #print(Lambda)
    return Lambda, U
    
    raise NotImplementedError
#eigvals, U = get_eig(z, 3)

def get_eig_prop(S, prop):
    """
    Get eigenvalues and eigenvectors that explain more than prop proportion of variance.

    Args:
        S (numpy.ndarray): Covariance matrix (d x d)
        prop (float): Minimum proportion of variance to explain (0 <= prop <= 1)

    Returns:
        tuple: (Lambda, U) where Lambda is diagonal matrix of eigenvalues
               and U is matrix of corresponding eigenvectors as columns
    """
    evals_all, evecs_all = eigh(S)
    evals_descending = evals_all[::-1]
    evecs_descending = evecs_all[:, ::-1]
    total_variance = np.sum(evals_descending)    
    cumulative_variance = np.cumsum(evals_descending)/total_variance
    indices = np.where((evals_descending / total_variance) > prop)[0]
    #indices = np.searchsorted(cumulative_variance, prop) + 1
    U = evecs_descending[:, indices]
    eigenvalue = np.diag(evals_descending[indices])
    """print(eigenvalue)
    print(U)"""
    return eigenvalue, U
        
    raise NotImplementedError
"""eigvals_prop, U_prop = get_eig_prop(z, 0.07)
print(eigvals_prop)
print(U_prop)
"""
def project_and_reconstruct_image(image, U):
    """
    Project image to PCA subspace and reconstruct it back to original dimension.

    Args:
        image (numpy.ndarray): Flattened image vector (d x 1)
        U (numpy.ndarray): Matrix of eigenvectors (d x m)

    Returns:
        numpy.ndarray: Reconstructed image as flattened d x 1 vector
    """
    U_trans = np.transpose(U)
    alpha = np.dot(image, U)
    x_pca = np.dot(alpha, U_trans)
    """print("x:", x.shape)
    print("U:", U.shape)
    print("U.T:", U.T.shape)"""
    return x_pca
    
    raise NotImplementedError
#x_pca = project_and_reconstruct_image(x, U)

def project_reconstruct_with_gaussian_noise(image, U, sigma=0.1, seed=0):
    """
    Project image to PCA subspace, add Gaussian noise to coefficients, and reconstruct.

    Args:
        image (numpy.ndarray): Flattened image vector (d x 1)
        U (numpy.ndarray): Matrix of eigenvectors (d x m)
        sigma (float): Standard deviation of Gaussian noise to add to coefficients
        seed (int): Random seed for reproducibility
    Returns:        
        numpy.ndarray: Reconstructed image with noise as flattened d x 1 vector
    """
    np.random.seed(seed)
    U_trans = np.transpose(U)
    alpha = np.dot(image, U)
    epsilon = np.random.normal(0, sigma, size=alpha.shape)
    
    alpha_noise = alpha + epsilon
    x_pca = np.dot(alpha_noise, U_trans)
    """print("x:", x.shape)
    print("U:", U.shape)
    print("U.T:", U.T.shape)"""
    return x_pca
    raise NotImplementedError
#x_noise_pca = project_reconstruct_with_gaussian_noise(x, U)

def display_image(im_orig_fullres, im_orig, im_reconstructed):
    """
    Display three images side by side: original high-res, original, and reconstructed.

    Args:
        im_orig_fullres (numpy.ndarray): Original high-resolution image
        im_orig (numpy.ndarray): Original low-resolution image
        im_reconstructed (numpy.ndarray): Reconstructed image from PCA

    Returns:
        tuple: (fig, ax1, ax2, ax3) matplotlib figure and axes objects
    """

    # Please use the format below to ensure grading consistency
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9,3), ncols=3)
    fig.tight_layout()
    im_orig_fullres = im_orig_fullres.reshape(218, 178, 3)
    im_orig = im_orig.reshape(60, 50)
    im_reconstructed = im_reconstructed.reshape(60, 50)


    ax1.imshow(im_orig_fullres, aspect='equal')
    ax2.imshow(im_orig, aspect='equal')
    ax3.imshow(im_reconstructed, aspect='equal')


    ax1.set_title("Original High Res")
    ax2.set_title("Original")
    ax3.set_title("Reconstructed")

    fig.colorbar(ax2.images[0], ax=ax2)
    fig.colorbar(ax3.images[0], ax=ax3)

    
    # Note: Do NOT include plt.show() in your implementation - it will be called separately for testing
    return fig, ax1, ax2, ax3

x = load_and_center_dataset("celeba_60x50.npy")
z = get_covariance(x)
eigvals, U = get_eig(z, 3)
celeb_idx = 25
x_img = x[celeb_idx]
full_res_image = np.load("celeba_218x178x3.npy")[celeb_idx]
reconstructed_img = project_and_reconstruct_image(x_img, U)
fig, ax1, ax2, ax3 = display_image(full_res_image, x_img, reconstructed_img)
plt.show()
