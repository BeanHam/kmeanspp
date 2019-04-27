import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Kmeanspp:
    """K-Means++ Clustering Algorithm"""
    
    def __init__(self, k, centers=None, cost=None,iter=None, labels=None, max_iter = 1000):
        """Initialize Parameters"""
        
        self.max_iter = max_iter
        self.k = k
        self.centers = np.empty(1)
        self.cost = []
        self.iter = 1
        self.labels = np.empty(1)

    def calc_distances(self, data, centers, weights):
        """Distance Matrix"""
        
        distance = pairwise_distances(data, centers)**2
        min_distance = np.min(distance, axis = 1)
        D = min_distance*weights
        return D
    
    def initial_centers_Kmeansapp(self, data, k, weights):
        """Initialize centers for K-Means++"""
        
        centers = []
        centers.append(random.choice(data))
        while(len(centers) < k):   
            distances = self.calc_distances(data, centers, weights)
            prob = distances/sum(distances)
            c = np.random.choice(range(data.shape[0]), 1, p=prob)
            centers.append(data[c[0]])
        return centers
    

    def fit(self, data, weights=None):
        """Clustering Process"""
        
        if weights is None: weights = np.ones(len(data))
        if type(data) == pd.DataFrame: data=data.values
        nrow = data.shape[0]
        self.centers = self.initial_centers_Kmeansapp(data, self.k, weights)
        
        while (self.iter <= self.max_iter):
            distance = pairwise_distances(data, self.centers)**2
            self.cost.append(sum(np.min(distance, axis=1)))
            self.labels = np.argmin(distance, axis=1)
            centers_new = np.array([np.mean(data[self.labels == i], axis=0) for i in np.unique(self.labels)])
            
            ## sanity check
            if(np.all(self.centers == centers_new)): break 
            self.centers = centers_new
            self.iter += 1
        
        ## convergence check
        if (sum(np.min(pairwise_distances(data, self.centers)**2, axis=1)) != self.cost[-1]):
            warnings.warn("Algorithm Did Not Converge In {} Iterations".format(self.max_iter))
        return self

