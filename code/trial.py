import numpy as np
def find_local_extrema(data,mini = -30,maxi = 30,dist=100):
    """
    Find indices of local minima and maxima in the given 1D numpy array.
    
    :param data: A 1D numpy array of the time series data.
    :return: A tuple of two lists, first containing indices of local minima and second containing indices of local maxima.
    """
    # Initialize lists to hold the indices of local minima and maxima
    minima = []
    maxima = []

    # Iterate over the data, skipping the first and last points
    for i in range(len(data)-1):
        # Local minimum if the current value is less than both neighbors
        if data[i - 1] > data[i] < data[i + 1]:
             if data[i] <mini:
                minima.append(i)
        # Local maximum if the current value is greater than both neighbors
        elif data[i - 1] < data[i] > data[i + 1]:
             if data[i] >maxi:
                maxima.append(i)
        
    return minima, maxima
                
def get_trial_structure(ptimes, positions,mini=-25,maxi = 25, dist = 100):
    mini, maxi = find_local_extrema(positions,mini = mini,maxi = maxi,dist=dist)
    maxi = np.append(maxi,len(positions)-1)
    trial_start = ptimes[mini]
    trial_end = ptimes[maxi]
    return trial_start,trial_end
