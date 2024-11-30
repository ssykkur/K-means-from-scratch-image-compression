import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

def load_data():
    X = np.load("data/x.npy")

    return X

load_data()