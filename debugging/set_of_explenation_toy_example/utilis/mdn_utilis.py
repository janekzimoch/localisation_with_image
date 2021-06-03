import numpy as np
from bisect import bisect

def sample_from_categorical_many_samples(cat_dist):
    num_samples = len(cat_dist)
    random_numbers = np.random.uniform(low=0.0, high=1.0, size=num_samples)
    
    # convert categorical to cumulative distribution
    cumulative_dist = np.zeros(cat_dist.shape) 
    for i in range(cat_dist.shape[-1]):
        if i == 0:
            cumulative_dist[:,i] = cat_dist[:,i]
        else:
            cumulative_dist[:,i] = cat_dist[:,i] + cumulative_dist[:,i-1]
            
    # identify samples
    samples = np.empty(num_samples)
    for i in range(num_samples):
        samples[i] = bisect(cumulative_dist[i], random_numbers[i])
    
    return samples.astype(int)


def sample_from_categorical(cat_dist):
    random_number = np.random.uniform(low=0.0, high=1.0, size=1)
    
    # convert categorical to cumulative distribution
    cumulative_dist = np.zeros(len(cat_dist))
    for i in range(len(cat_dist)):
        if i == 0:
            cumulative_dist[i] = cat_dist[i]
        else:
            cumulative_dist[i] = cat_dist[i] + cumulative_dist[i-1]
            
    # identify sample
    samples = bisect(cumulative_dist, random_number)
    
    return samples



def softmax(e):
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    dist = e / np.sum(e)
    return dist



def sample_from_output(y_pred, num_comp):
    " y_pred must be a single prediction "
    
    gaus_comp = y_pred[:num_comp]
    means = y_pred[num_comp:2*num_comp]
    variances = y_pred[2*num_comp:]
    
    pis = softmax(gaus_comp)
    m = sample_from_categorical(pis)
    
    output_dim = 1
    means_vector = means[m:m+1]
    variances_vector = variances[m:m+1]
    scale_matrix = np.identity(output_dim) * variances_vector  # scale matrix from diag
    cov_matrix = np.matmul(scale_matrix, scale_matrix.T)  # cov is scale squared.
    sample = np.random.multivariate_normal(means_vector, cov_matrix, 1)
    return sample