#!/usr/bin/env python3

from matplotlib import pyplot as plt
from persim import PersistenceImager, plot_diagrams
from ripser import ripser
from scipy import sparse
import numpy as np


def read_time_series():
    t = np.arange(0, 20, 0.1)
    y = t*np.sin(t)
    return t, y


def time_series_to_diagram(y):
    def time_series_to_matrix(y):
        N = len(y)
        # add edges between adjacent points in the time series, with the "distance" along the edge equal to the max value of the points it connects
        I = np.arange(N-1)
        J = np.arange(1, N)
        V = np.maximum(y[0:-1], y[1::])
        # add vertex birth times along the diagonal of the distance matrix
        I = np.concatenate((I, np.arange(N)))
        J = np.concatenate((J, np.arange(N)))
        V = np.concatenate((V, y))
        # create the sparse distance matrix
        D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
        return D

    def matrix_to_diagram(D, threshold=None):
        dgm0 = ripser(D, maxdim=0, distance_matrix=True)['dgms'][0]
        if threshold:
            # noise filter (along the diagram diagonal)
            dgm0 = dgm0[dgm0[:, 1]-dgm0[:, 0] > threshold, :]
        if len(dgm0) > 1:
            # Remove point at infinite
            dgm0 = dgm0[:-1]
        return dgm0

    D = time_series_to_matrix(y)
    dgm0 = matrix_to_diagram(D)
    return dgm0


def diagram_to_image(dgm):
    if len(dgm) > 0:
        pixels = 20
        lifetime = np.concatenate([dgm[:, [0]], np.diff(dgm)], axis=1)
        birth_range = (lifetime.min(), lifetime.max())
        pixel_size = np.diff(birth_range)[0] / pixels
        sigma = np.diff(dgm).std() * pixel_size / 2
        if sigma == 0:
            sigma = np.eye(2) * (np.diff(dgm).mean() * pixel_size / 2)
        pimgr = PersistenceImager(birth_range=birth_range, pers_range=birth_range, pixel_size=pixel_size, kernel_params=dict(sigma=sigma))
        img = pimgr.transform(dgm)
        img = np.rot90(img)
    else:
        img = np.zeros((20, 20))
    return img


def plot(t, y, dgm0, img):
    _, axs = plt.subplot_mosaic([['Time Series', 'Persistence Diagram', 'Lifetime Diagram', 'Persistence Image']], figsize=(16, 4))
    _ = [ax.set_title(title) for title, ax in axs.items()]
    axs['Time Series'].plot(t, y)
    plot_diagrams(dgm0, ax=axs['Persistence Diagram'])
    plot_diagrams(dgm0, ax=axs['Lifetime Diagram'], lifetime=True)
    axs['Persistence Image'].imshow(img)
    axs['Persistence Image'].axis('off')


def main():

    
    t, y = read_time_series()
    dgm0 = time_series_to_diagram(y)
    img = diagram_to_image(dgm0)
    plot(t, y, dgm0, img)
    plt.show()


if __name__ == "__main__":
    main()
