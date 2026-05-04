import numpy as np
import math

def replace_farthest_neighbour(neighbour, nearest_neighbours):
    num = neighbour[1]

    # first neighbour has always the biggest distance
    if (nearest_neighbours[0][1] > num): 
        nearest_neighbours[0] = neighbour
        nearest_neighbours.sort(key=lambda neighbour: neighbour[1], reverse=True)
    
    return nearest_neighbours
    

def find_k_nearest_neighbours(k, target_coordinates : np.array, data : np.array, labels : np.array):
    # [0] - coordinates
    # [1] - distance to target
    # [2] - flower class
    nearest_neighbours = [[[0, 0, 0, 0], 999999, '-'] for i in range(k)]
    
    for i in range(data.shape[0]):
        flower_name = labels[i]
        curr_len = math.sqrt(sum((data[i] - target_coordinates) ** 2))
        curr_neighbour = [data[i], curr_len, flower_name]
        
        nearest_neighbours = replace_farthest_neighbour(curr_neighbour, nearest_neighbours)

    return nearest_neighbours


def predict(k : int, target_coordinates : list[int], data : np.array, labels : np.array):
    nearest_neighbours = find_k_nearest_neighbours(k, target_coordinates, data, labels)
    
    setosa_votes = sum([1 for neighbour in nearest_neighbours if neighbour[2] == 0])
    versicolor_votes = sum([1 for neighbour in nearest_neighbours if neighbour[2] == 1])
    virginica_votes = sum([1 for neighbour in nearest_neighbours if neighbour[2] == 2])

    votes = [setosa_votes, versicolor_votes, virginica_votes]

    return votes.index(max(votes))


def shuffle_data(data : np.array, labels: np.array):
    num_of_features = data.shape[1]
    labels = np.reshape(labels, (labels.shape[0], 1))
    
    concatenated = np.concatenate((data, labels), axis=1)
    np.random.shuffle(concatenated)
    
    shuffled_data = concatenated[:, 0:num_of_features]
    shuffled_labels = concatenated[:, -1]

    shuffled_labels = np.ravel(shuffled_labels)

    return (shuffled_data, shuffled_labels)


def intToGroup(num : int):
    if num == 0: return 'setosa'
    if num == 1: return 'versicolor'
    if num == 2: return 'virginica'
    return None

# testing ground

# features = np.array([[5.1, 3.5, 1.4, 0.2],
#  [4.9, 3.,  1.4, 0.2],
#  [4.7, 3.2, 1.3, 0.2]])
# labels =  np.array([0, 1, 2])
