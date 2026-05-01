import numpy as np
import math

def replace_farthest_neighbour(neighbour, array : list[tuple]):
    num = neighbour[1]

    if (array[0][1] > num): 
        array[0] = neighbour
        array.sort(key=lambda neighbour: neighbour[1], reverse=True)
        return array
    
    return array
    

def find_k_nearest_neighbours(k, coordinates : np.array, data : np.array, labels : np.array):
    nearest_neighbours = [((0, 999999)) for i in range(k)]
    
    for i in range(data.shape[0]):

        curr_len = math.sqrt(sum((data[i] - coordinates) ** 2))
        curr_neighbour = (i, curr_len)

        nearest_neighbours = replace_farthest_neighbour(curr_neighbour, nearest_neighbours)

    return nearest_neighbours