from tensorflow import keras
import numpy as np
import tensorflow as tf
import keras.backend as K

from losses.mdn_loss import get_mdn_loss


def get_MDN_RWTA_loss(num_comp, eps, output_shape, fixed_variance=False):
    mdn_loss_function = get_mdn_loss(num_comp, eps, output_shape=output_shape, fixed_variance=fixed_variance)
    nll_rwta_loss_function = get_RWTA_loss(num_comp, eps, output_shape=output_shape, fixed_variance=fixed_variance)
    
    def get_combined_loss(y_true, y_pred):
        
        loss_mdn = mdn_loss_function(y_true, y_pred)
        loss_rwta = nll_rwta_loss_function(y_true, y_pred)
        
        loss = loss_mdn + loss_rwta
        return loss
    
    return get_combined_loss



def multivariate_NLL(y_true, y_pred):
    """ negative log likelihood loss function for multivariate gaussian with DIAGONAL covariance
    Parameters: 
    y_true [batch_size, output_dim]
    y_pred [batch_size, output_dim*2] i.e. means followed by logvariance (both predicted by a NN)
    """
    n_dims = int(int(y_pred.shape[1])/2)
    mu = y_pred[:, 0:n_dims]
    sigma = y_pred[:, n_dims:]
    
    mse = -0.5*K.sum(K.square((y_true-mu)/sigma),axis=1)
    sigma_trace = -K.sum(K.log(sigma), axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = mse+sigma_trace+log2pi

    return -log_likelihood


def get_RWTA_loss(M, eps, output_shape, fixed_variance=False):
    """
    M - number of mixture components; 
    eps - (epsilon) the relaxation weight - usually something very small i.e. 0.001
    output_dim - how many variables does the gaussian have? usually 1 (for 1D - x coord) or 3 (for 3D - x,y,z)
    """
    output_dim = output_shape

    
    def RWTA_loss_for_mdn(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        y_pred = tf.reshape(y_pred, [-1, (2 * M * output_dim) + M], name='reshape_ypreds')
        y_true = tf.reshape(y_true, [-1, output_dim], name='reshape_ytrue')
        
        # Split the inputs into paramaters
        out_pi, out_mu, out_sigma = tf.split(y_pred, num_or_size_splits=[M,
                                                                         M * output_dim,
                                                                         M * output_dim],
                                             axis=-1, name='mdn_coef_split')
        if fixed_variance:
#             batch_size = out_sigma.get_shape()
#             print(batch_size)
            out_sigma = tf.fill(tf.shape(out_sigma), 1.0) #np.ones((batch_size[0], M*output_dim))

        # get NLL for each mixture
        mixtures_nll = []
        for m_ind in range(M):
            mixture_y_pred = tf.concat([out_mu[:, m_ind*output_dim:(m_ind+1)*output_dim], out_sigma[:, m_ind*output_dim:(m_ind+1)*output_dim]], -1)
            nll = multivariate_NLL(y_true, mixture_y_pred)
            nll = tf.expand_dims(nll, 1)
            mixtures_nll.append(nll)
        
        mixtures_combined = tf.concat(mixtures_nll, 1)
        # RWTA
        loss = (1-eps) * (M-1/M) * tf.math.reduce_min(mixtures_combined, axis=1) + (eps/M) * tf.math.reduce_sum(mixtures_combined, axis=1)
        return tf.reduce_mean(loss)
    
    return RWTA_loss_for_mdn