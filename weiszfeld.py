import numpy as np 

def geometric_median(points, tol=1e-5, max_iter=1000):

    points = np.asarray(points)
    median = np.mean(points, axis=0)

    for _ in range(max_iter):

        distances = np.linalg.norm(points - median, axis=1)

        non_zero_distances = np.where(distances == 0, 1e-10, distances)

        weights = 1 / non_zero_distances
        new_median = np.sum(points * weights[:, np.newaxis], axis=0) / np.sum(weights)

        if np.linalg.norm(new_median - median) < tol:
            return new_median 
        

    return median 



