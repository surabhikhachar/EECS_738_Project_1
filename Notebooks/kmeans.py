import copy
import numpy as np
import math
import random

class K_Means:
    def __init__(self, xAxis, yAxis, classifications, tol=0.0001, max_iter=500):
        self.tol = tol
        self.max_iter = max_iter

        assert len(xAxis) == len(yAxis)
        self.k = len(set(classifications))

        data = [(xAxis[i], yAxis[i]) for i in range(len(xAxis))]
        uniqueData = list(set(data))
        random.shuffle(uniqueData)
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = uniqueData[i]

        iterations = 0
        while (iterations < self.max_iter):
            iterations += 1
            distances = []
            for i in range(len(xAxis)):
                distances.append([])
                for j in range(self.k):
                    centroid = self.centroids[j]
                    squaredDistance = (float(xAxis[i]) - float(centroid[0])) ** 2 + (float(yAxis[i]) - float(centroid[1])) ** 2
                    distances[i].append(math.sqrt(squaredDistance))

            self.classifications = {}
            for i in range(len(xAxis)):
                self.classifications[i] = distances[i].index(min(distances[i]))

            old_centroids = copy.deepcopy(self.centroids)

            for i in range(self.k):
                xcentroidPoints = []
                ycentroidPoints = []
                for x in range(len(xAxis)):
                    if(self.classifications[x] == i):
                        xcentroidPoints.append(xAxis[x])
                        ycentroidPoints.append(yAxis[x])
                self.centroids[i] = (np.mean(xcentroidPoints), np.mean(ycentroidPoints))

            if old_centroids == self.centroids:
                break

            optimized = True
            for i in range(self.k):
                original_centroid = old_centroids[i]
                current_centroid = self.centroids[i]
                if np.sum((current_centroid[0] - original_centroid[0]) / original_centroid[0]*100.0) > self.tol and np.sum((current_centroid[1] - original_centroid[1]) / original_centroid[1]*100.0) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self, x, y):
        distances = []
        for i in range(self.k):
            centroid = self.centroids[i]
            squaredDistance = (x - float(centroid[0])) ** 2 + (y - float(centroid[1])) ** 2
            distances.append(np.sqrt(squaredDistance))
        classification = distances.index(min(distances))
        return classification
