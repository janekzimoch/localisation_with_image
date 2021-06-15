import numpy as np
from general_utilis import get_mixture_dist


def evaluate_GMM_log_likelihood(model, x, y):
    """
    Evaluate likelihood of model parameters p(y|x,theta) 
    where model is expressed with a GMM, where pi, mu, sigma are outputs of a neural network
    
    We use this functions to compare across models.
    """
    y_pred = model.predict(x)
    
    num_datapoints = len(x)
    output_dim = y.shape[-1]
    num_comp = int(y_pred.shape[-1] / (3*output_dim))

    mix_comp_logits = y_pred[:, :num_comp]
    mus = y_pred[:, num_comp:(1+output_dim)*num_comp]
    sigmas = y_pred[:, (1+output_dim)*num_comp:]
    
    # convert logits to categorical distribution - need to itterate through all points
    mix_comp = np.zeros((num_datapoints, num_comp))
    for i in range(num_datapoints):
        mix_comp[i,:] = get_mixture_dist(mix_comp_logits[i,:], num_comp)
    
    log_likelihood = 0
    for i in range(num_comp):
        for j in range(output_dim):
            mse = -0.5*np.sum(mix_comp[:,i]*np.square((y[:,j]-mus[:,(i*output_dim)+j])/sigmas[:,(i*output_dim)+j]))
            sigma_trace = -np.sum(mix_comp[:,i]*np.log(sigmas[:,(i*output_dim)+j]))
            log2pi = -np.sum(mix_comp[:,i]*0.5*output_dim*np.log(2*np.pi))

            log_likelihood += mse + sigma_trace + log2pi
             
    avg_log_likelihood = np.round(log_likelihood / num_datapoints, 2)
    print(f'Log likelihood: {avg_log_likelihood}')
    return avg_log_likelihood