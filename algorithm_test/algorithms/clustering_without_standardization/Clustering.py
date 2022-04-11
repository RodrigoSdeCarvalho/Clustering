import math 
import numpy as np

class Clustering:
    def __init__(self,  centroids: list, dataset: list, max_radius: int, min_in_cluster: int, dataset_dimensions: int ):
        """Receives a dataset, the centroids, the maximum value for the centroids' cluster
           and creates an empty list to store the clusters.
        """
        self.__dataset = dataset
        self.__max_radius = max_radius
        self.__dataset_dimensions = dataset_dimensions
        self.__min_in_cluster = min_in_cluster

        self.__centroids = []
        if centroids == []:
            self.__centroids.append(self.__dataset[0])
        else:
            for index in centroids:
                self.__centroids.append(self.__dataset[index])

        self.__clusters = []
        for centroid in self.__centroids:
            new_centroid = [centroid]
            self.__clusters.append(new_centroid)

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

    @property
    def min_in_cluster(self):
        return self.__min_in_cluster

    @min_in_cluster.setter
    def max_in_cluster(self, min_in_cluster):
        self.__min_in_cluster = min_in_cluster


    def new_data_point(self, data_point: list):
        """Creates a new data point in the data set."""
        self.__dataset.append(data_point)

    def new_cluster(self, centroid: list):
        """Creates a new cluster and defines its centroid."""
        new_cluster = [centroid]
        self.clusters.append(new_cluster)

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
        """Shows all the clusters in the terminal."""
        n = 1
        for cluster in self.clusters:
            print(f'cluster {n}: {cluster}')
            n += 1

    def show_number_of_clusters(self):
        """Returns the numbers of clusters calculated by the algorithm."""
        number_of_clusters = len(self.clusters)
        print(number_of_clusters)

        return number_of_clusters

    def check_clustering(self): #Discarted for now.
        """Checks if the clustering is right and returns its accuracy based on 
        the """
        #gets the id of the centroid and checks if the other datapoints are right.
        #Returns the accuracy in %.
        pass

    def save_clusters_in_txt(self, txt_name:str):
        """Creates a txt file and saves the clusters in it."""
        #Since the cluster is a 2d matrix, it can be turned into a dataframe (How to export a dataframe in txt?).
        txt_cluster = open(f"C:/Users/CLIENTE/documentos/Dev Files/LISHA/Clustering/algorithm_test/testes/{txt_name}.txt", "w+")
        number_of_clusters = len(self.clusters)

        txt_cluster.write(f'{txt_name}: \n'
                          '\n'
                          f'centroids: {self.centroids} \n' 
                          '\n'
                          f'radius: {self.max_radius} \n'
                          '\n'
                          f'minimum number of members in a cluster: {self.min_in_cluster} \n'
                          '\n'
                          f'number of clusters: {number_of_clusters} \n'
                          '\n')

        n = 1
        for cluster in self.clusters: 
            txt_cluster.write(f'cluster {n}: {cluster}')
            n += 1

            txt_cluster.write('\n') #Blank line.
            txt_cluster.write('\n')

        txt_cluster.close()

    def get_data_dimensions(self):
        """Gets the data points dimensions and separates them in lists.
           It returns an array with all the data dimensions."""
        dataset = self.dataset

        data_0 = []
        data_1 = []
        data_2 = []
        data_3 = []
        data_4 = []
        data_5 = []
        data_ID = []

        for data_point in dataset:
            data_0.append(data_point[0])
            data_1.append(data_point[1])
            data_2.append(data_point[2])
            data_3.append(data_point[3])
            data_4.append(data_point[4])
            data_5.append(data_point[5])
            data_ID.append(data_point[6])

        data_dimensions = [data_0, data_1, data_2, data_3, data_4, data_5,data_ID]

        return data_dimensions

    def to_dataset(self, data_dimensions):
        """Receives a data_dimensions array and converts it to dataset."""
        dataset = []

        for index_datapoint in range(len(data_dimensions[0])):

            datapoint = [data_dimensions[0][index_datapoint],
                         data_dimensions[1][index_datapoint],
                         data_dimensions[2][index_datapoint],
                         data_dimensions[3][index_datapoint],
                         data_dimensions[4][index_datapoint],
                         data_dimensions[5][index_datapoint],
                         data_dimensions[6][index_datapoint]]
            dataset.append(datapoint)

        return dataset

    def remove_outliers(self):
        """Removes outliers in the dataset. And returns the dataset without outliers"""
        data_dimensions = self.get_data_dimensions()
        data_dimensions = np.array(data_dimensions)

        indexes_to_remove = []
        for index_axis, axis in enumerate(data_dimensions[0:6]):

            data_std = np.std(data_dimensions[index_axis])
            data_mean = np.mean(data_dimensions[index_axis])

            for index_value, value in enumerate(axis):
                if not (value <= data_mean + 2*data_std):
                    indexes_to_remove.append(index_value)

        data_dimensions = data_dimensions.tolist()
        aux_data_dimensions = self.get_data_dimensions()
        indexes_to_remove = list(set(indexes_to_remove))

        for index in indexes_to_remove:
            data_dimensions[0].remove(aux_data_dimensions[0][index])
            data_dimensions[1].remove(aux_data_dimensions[1][index])
            data_dimensions[2].remove(aux_data_dimensions[2][index])
            data_dimensions[3].remove(aux_data_dimensions[3][index])
            data_dimensions[4].remove(aux_data_dimensions[4][index])
            data_dimensions[5].remove(aux_data_dimensions[5][index])
            data_dimensions[6].remove(aux_data_dimensions[6][index])
            

        dataset = self.to_dataset(data_dimensions)
        self.dataset = dataset 
        
        return self.dataset


    def start(self):
        """Starts the clustering algorithm and returns the clusters."""
        dataset = self.dataset
        centroids = self.centroids

        data_in_cluster = [False]*len(dataset)

        for index_data_point, data_point in enumerate(dataset):
            for index_centroid, centroid in enumerate(centroids):
                if (self.data_point_in_cluster(data_point, centroid)) and (not data_in_cluster[index_data_point]):
                    self.clusters[index_centroid].append(data_point)
                    data_in_cluster[index_data_point] = True

            if not data_in_cluster[index_data_point]:
                self.new_cluster(data_point)

        return self.clusters

    def start_min(self): #Think of a better name.
        """Starts the clustering algorithm and returns the clusters
        that have less or an equal number of members in comparison to
        the maximum number in a cluster."""
        self.start()

        clusters = self.clusters
        min = self.min_in_cluster
        new_clusters = []

        for cluster in clusters:
            if len(cluster) >= min:
                new_clusters.append(cluster)

        self.clusters = new_clusters

        return self.clusters

    def start_without_outliers(self):
        self.remove_outliers()
        self.start()
