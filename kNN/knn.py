#!/usr/bin/python
from collections import Counter
import math

def get_instance(csv_row, separator = ','):
    data = csv_row.strip().split(separator)
    for index, item in enumerate(data):
        try:
            data[index] = float(item)
        except ValueError:
            pass
    return data

def knn(data_set, reg_query, k = 3, center_fn = 'mean', debug = False):
    neighbor = []
    instance_len = None

    choice_functions = {
        'mean': lambda labels: sum(labels) / len(labels),
        'mode': lambda labels: Counter(labels).most_common(1)[0][0]
    }

    for index, observation in enumerate(data_set):
        if not index:
            instance_len = len(observation) - 1
        distance = euclidean_distance(observation[:-1], reg_query)
        classification = observation[instance_len]
        neighbor.append((distance, index, classification))

    sorted_neighbor = sorted(neighbor)
    k_nearest_neighbor = sorted_neighbor[:k]
    neighbor_labels = [clf for _, __, clf in k_nearest_neighbor]
    return k_nearest_neighbor, choice_functions[center_fn](neighbor_labels)

def euclidean_distance(x, y):
    sum_squared_distance = 0
    for i in range(len(x)):
        sum_squared_distance += math.pow(x[i] - y[i], 2)
    return math.sqrt(sum_squared_distance)

if __name__ == '__main__':
    raw_iris_data = []
    with open('iris.data', 'r') as file:
        for line in file.readlines():
            instance = get_instance(line)
            raw_iris_data.append(instance)
    neighbors, prediction = knn(raw_iris_data, [4.8, 3.0, 1.4, 0.1], center_fn='mode')
    print(neighbors, prediction)