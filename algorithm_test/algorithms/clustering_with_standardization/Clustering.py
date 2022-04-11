#This class standardize a dataset before clusterizing it.

import numpy as np
import math 
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) #remover warning desnecessario do numpy no terminal.

class Clustering:
    def __init__(self,  centroids: list, dataset: list, max_radius: int, dataset_dimensions: int ):
        """Receives a dataset, the centroids, the maximum value for the centroids' cluster
           and creates an empty list to store the clusters.
        """
        scaler_data_point = StandardScaler()
        self.__dataset = scaler_data_point.fit_transform(np.array(dataset))
        self.__max_radius = max_radius
        self.__dataset_dimensions = dataset_dimensions

        self.__centroids = []
        if centroids == []:
            self.__centroids.append(self.__dataset[0])
        else:
            for index in centroids:
                self.__centroids.append(self.__dataset[index])

        self.__clusters = []
        for centroid in self.__centroids:
            self.__clusters.append([centroid])

        self.start()

    @property    
    def dataset(self):
        return self.__dataset

    @dataset.setter
    def dataset(self, dataset: list):
        self.__dataset = dataset

    @property    
    def centroids(self):
        return self.__centroids 

    @centroids.setter
    def centroids(self, centroids: list):
        self.__centroids = centroids   

    @property
    def clusters(self):
        return self.__clusters

    @clusters.setter
    def clusters(self, clusters: list):
        self.__clusters = clusters

    @property    
    def max_radius(self):
        return self.__max_radius

    @max_radius.setter
    def max_radius(self, max_radius: int):
        self.__max_radius = max_radius

    @property
    def dataset_dimensions(self):
        return self.__dataset_dimensions

    @dataset_dimensions.setter
    def dataset_dimensions(self, dataset_dimensions: int):
        self.__dataset_dimensions = dataset_dimensions

    def new_data_point(self, data_point: list):
        """Creates a new data point in the data set."""
        self.__dataset.append(data_point)

    def new_cluster(self, centroid: list):
        """Creates a new cluster and defines its centroid."""
        self.clusters.append([centroid])

    def euclidean_distance(self, data_point1: list, data_point2: list): #Static Method
        """Calculates and returns the euclidean distance between two 
        given data points
        """
        num_coordinates = self.dataset_dimensions

        distance = 0
        for i in range(num_coordinates):
            
            distance += ((data_point1[i] - data_point2[i]) ** 2)
        distance = math.sqrt(distance)

        return distance

    def data_point_in_cluster(self, data_point: list, centroid: list): #Static Method
        """Determines if a data point belongs to a cluster, given its centroid."""
        distance = self.euclidean_distance(data_point, centroid)

        if distance <= self.max_radius:
            return True
        else:
            return False

    def show_clusters(self):
        n = 1
        for cluster in self.clusters:
            print(f'cluster {n}: {cluster}')
            n += 1

    def show_number_of_clusters(self):
        number_of_clusters = len(self.clusters)
        print(number_of_clusters)

    def start(self):
        """Starts the clustering algorithm and returns the clusters."""
        dataset = self.dataset
        centroids = self.centroids

        data_in_cluster = [False]*len(dataset)

        for data_point in dataset:
            index_data_point = np.where(dataset == data_point)

            for centroid in centroids:
                index_centroid = np.where(centroids == centroid)

                if (self.data_point_in_cluster(data_point, centroid)) and (not data_in_cluster[index_data_point[0][0]]):
                    self.clusters[index_centroid[0][0]].append(data_point)
                    data_in_cluster[index_data_point[0][0]] = True

            if not data_in_cluster[index_data_point[0][0]]:
                self.new_cluster(data_point)

        return self.clusters
