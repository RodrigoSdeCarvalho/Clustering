from algorithms.clustering_without_standardization.Clustering import Clustering
import pandas as pd

#quantidade minima de membros no clusters
#salvar os centroids em um csv para comparar dps
#analisar data points não agrupados

dataset = pd.read_csv("C:/Users/CLIENTE/documentos/Dev Files/LISHA/Clustering/files_rodrigo/data_multicore.csv") #reads the csv file.
dataset = dataset.iloc[:,0:6].values.tolist() #Converts the file into an array of all the datapoints (which all have 6 dimensions).
centroids = []
max_radius = 0.10 #0.10 ou 0.15

clusters = Clustering(centroids, dataset, max_radius, 6)
clusters.show_clusters() #Exportar isto dps num arquivo de texto pra conferir os ids (Ver se estão no cluster certo).
clusters.show_number_of_clusters()
