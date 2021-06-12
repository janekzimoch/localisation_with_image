import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf

from general_utilis import get_mixture_dist


def visualise_probability_for_datapoint(model, x, y, pattern_ids, num_modes, pattern_ID=0, num_points=1, ind=None, fixed_variance=True):
    """
    This function visualises all modes ('y' values) for pattern associated with 'pattern_ID'.
    It will then plot the probability distribution outputed by the MDN. 
    """

    unique_patterns = np.unique(pattern_ids)
    p_indexes = np.where(pattern_ids == pattern_ID, 1, 0).astype(bool)
    gt_modes = np.unique(y[p_indexes])
    num_outputs = y.shape[-1]
    
    for i in range(num_points):
        plt.figure(figsize=(16,2))

        # plot probability
        y_tmp = y[p_indexes]
        x_tmp = x[p_indexes]
        if ind != None:
            random_ind = ind
        else:
            random_ind = np.random.randint(len(x_tmp))
        y_dist = model.predict(np.expand_dims(x_tmp[random_ind], axis=0))
        mix_comp, means, var = y_dist[0,:num_modes], y_dist[0,num_modes: (num_outputs+1)*num_modes], y_dist[0, (num_outputs+1)*num_modes:]
        if fixed_variance:
            var = [1]*num_modes
        
        # somehow need to plot a gaussian
        n = 1000
        y_samples = np.arange(n)
        probability = np.zeros(n)
        gaus_mix = get_mixture_dist(mix_comp, num_modes) 
        
        for j in range(num_modes):
            probability += gaus_mix[j] * norm.pdf(y_samples, loc=means[j], scale=var[j])
        plt.plot(y_samples, probability, linewidth=0.5, color='tab:blue', label='GMM\'s p(y|img)')
        
        # plot ground truth
        plt.scatter(y_tmp[random_ind], [0], color='tab:green', linewidth=10, alpha=0.5, label='sample GT')
        plt.scatter(gt_modes, [0] * len(gt_modes), linewidth=3, alpha=0.5, color='tab:red', label='GT modes')
        
        plt.ylabel('p( y | img)')
        plt.xlabel('y (distance)')
        plt.title(f'Pattern {pattern_ID}')
        plt.legend()
        plt.show()

        print(f'GT: {y_tmp[random_ind]}')
        for j in range(num_modes):
            pi = np.around(gaus_mix[j],2)
            mu = np.around(means[j],2)
            sig = np.around(var[j],2)
            print(f'Component {j+1}: pi=', pi, ', mu=', mu, ', sig=', sig)

            
def visualise_model_evolution(mix_evolution, y_label, log=False):
    """
    This function plots evolution of average max_mixture component across epochs.
    Because the dataset is balanced, each mode should receive approximately 0.33 component probability - so that should be the max
    However, in practice we are more likely to see max mixture with pi>0.33 because sometimesmodes overlap.
    
    It would be cool to see what happens when we use MDN with more mixtures then there are modes in the data.
    Maybe then this metric will be lower.
    """
    plt.figure(figsize=(16,2))
    plt.plot(range(len(mix_evolution)), mix_evolution)
    plt.xlabel('epochs')
    plt.ylabel(y_label)
    if log:
        plt.yscale('log')
    plt.show()            


def visualise_modes(model, x, y, pattern_ids, modes, num_modes, num_outputs):
    """
    This function shows where the predicted mode means lie with respect to the ground truth modes.
    later i may add a plot to visualise entire learned probability distribution.
    I use an arbitrary way to compute locations of modes (i.e. mean across all point sharing that mode.)
    This might be not the best way to do this. 
    It might be in fact best to evaluate this problem at a point-by-point basis. - see visualise_probability_for_datapoint() function
    """
    plt.figure(figsize=(16,2))
    unique_patterns = np.unique(pattern_ids)
    
    for i, p_id in enumerate(unique_patterns):
        p_indexes = np.where(pattern_ids == p_id, 1, 0)
        
        for m_id in range(num_modes):
            m_indexes = np.where(modes == m_id, 1, 0)
            mp_indexes = p_indexes * m_indexes
            mp_indexes = mp_indexes.astype(bool)

            x_tmp = x[mp_indexes]
            y_tmp = y[mp_indexes]
            y_pred = model.predict(x_tmp)
            
            main_comp_ind = np.argmax( np.mean(y_pred[:,:num_modes], axis=0) )
            means = np.mean(y_pred[:, num_modes:(1+num_outputs)*num_modes], axis=0 )
            plt.scatter(i, np.mean(y_tmp), color='tab:green')  # true
            plt.scatter(i, means[main_comp_ind], color='tab:orange')  # pred

    plt.show()