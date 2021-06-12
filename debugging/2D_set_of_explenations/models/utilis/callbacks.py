import tensorflow as tf

from general_utilis import get_mixture_dist


# note metrics can only output scalar. you may want to use a callback or something so that you get more flexibility
def GMM_Metrics(num_comp, output_shape):
    
    def means(y_true, y_pred):
        means = y_pred[:,num_comp:(output_shape+1)*num_comp]
        return tf.reduce_mean(means)
        
    def mix_comp(y_true, y_pred):
        mix_comp = y_pred[:,0:num_comp], 
        return tf.reduce_mean(mix_comp)
        
    def var(y_true, y_pred):
        variances = y_pred[:,(output_shape+1)*num_comp:]
        return tf.reduce_mean(variances)
        
    return [mix_comp, means, var]   


def trace_max_component(num_comp, batch_size):
    """ This metric allows to track whether mixture components stay at 0.33 probability 
    - as they should for this problem, because there are consistent three modes
    or to show how max_mixture component changes.
    """
    def max_mixture(y_true, y_pred):
        mix_comp = y_pred[:,0:num_comp]
        cat_dist_max = 0 
        for i in range(batch_size):
            cat_dist = get_mixture_dist(mix_comp[i], num_comp, numpy=False)
            cat_dist_max += tf.reduce_max(cat_dist)
        return cat_dist_max / batch_size
        
    return max_mixture