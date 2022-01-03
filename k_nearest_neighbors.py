# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        
        self.X_train = X
        self.Y_train = y


    def get_prediction(self, weight_type, index, dist_b, pred_list):

        target = []

        # Assignment of weight for distance and uniform

        if self.weights == 'distance':

            class_diff = {}

            for each_index in range(len(index)):

                target.append(self.Y_train[index[each_index]])
            
            
            for each_target in range(len(target)):

                if target[each_target] in class_diff.keys():

                    norm_dist = 1/dist_b[each_target]

                    # The inverse count with respect to the distance is calculated and added to the key value

                    class_diff[target[each_target]] += norm_dist

                else:
                    class_diff[target[each_target]] = 1

            # Sorted with respect to the value


            dict_order = dict(sorted(class_diff.items(), key=lambda item: item[1], reverse = True))

            pred = list(dict_order.keys())[0]

            pred_list.append(pred)



        # If the weight type is uniform      
                

        else:

            for i in index:
                target.append(self.Y_train[i])

            # Just obtain the max count class

            pred_list.append(max(set(target), key = target.count))


        return pred_list

        

    # Function for prediction

    def predict(self, X):

        pred_list = []
        

        for each in X:

            dist = []

            # Numpy is converted to list of tuples with X and X_train

            for each_train in self.X_train:

                distance_fact = self._distance(each_train,each)

                dist.append(distance_fact)

            # index in sored order of the distance


            index = [i[0] for i in sorted(enumerate(list(dist)), key=lambda x:x[1])]

            dist_list = list(dist)

            dist_list.sort()

            # Parsed with the given K

            index_parse = index[:self.n_neighbors]

            dist_list_parse = dist_list[:self.n_neighbors]


            weight_type = self.weights

            # FUnction to get the prediction

            pred_list = self.get_prediction(weight_type, index_parse, dist_list_parse, pred_list)


        return np.array(pred_list)
