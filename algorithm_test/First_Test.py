from algorithms.clustering_with_standardization.Clustering import Clustering

dataset = [[48.33333333,1800],[41.16666667,5600],[22.66666667,1700],
           [20,1000],[27,1200],[21,2900],[37,1850],[46,900],
           [53,950],[55,2000],[47,2100],[52,3000],[32,5900],
           [39,4100],[41,5100],[39,7000],[48,5000],[48,6500]] 
centroids = [0, 1, 2] 
max_radius = 0.50

clusters = Clustering(centroids, dataset, max_radius, 2)

clusters.show_clusters()

clusters.show_number_of_clusters()
