import tensorflow as tf

from losses.mdn_rwta import multivariate_NLL

def get_CE_RWTA_loss(M, eps, output_dim, fixed_variance=False):
    """
    Get X-entropy loss to train 'pi's - the mixture component weighting
    And RWTA loss to train each of gaussian mixtures
    """
    
    def CE_RWTA_loss(y_true, y_pred):
        
        # split y_pred - into GMM components
        out_pi, out_mu, out_sigma = tf.split(y_pred, num_or_size_splits=[M,
                                                                         M * output_dim,
                                                                         M * output_dim],
                                             axis=-1, name='mdn_coef_split')
        if fixed_variance:
            out_sigma = tf.fill(tf.shape(out_sigma), 1.0) #np.ones((batch_size[0], M*output_dim))
        
        # compute
        mixtures_nll = []
        for m_ind in range(M):
            mixture_y_pred = tf.concat([out_mu[:, m_ind*output_dim:(m_ind+1)*output_dim], out_sigma[:, m_ind*output_dim:(m_ind+1)*output_dim]], -1)
            nll = multivariate_NLL(y_true, mixture_y_pred)
            nll = tf.expand_dims(nll, -1)
            mixtures_nll.append(nll)
        
        # one hot encode mixtures_nll, where min(NLL) = 1, else =0
        mixtures_combined = tf.concat(mixtures_nll, 1)
        # i needed this hacky dimension expansion to use tf.equal
        mixtures_comb_min = tf.concat([tf.expand_dims( tf.reduce_min(mixtures_combined, axis=1), -1)] * M, 1)
        NLL_one_hot = tf.equal(mixtures_comb_min, mixtures_combined)
        
    
        
        # FINALLY GET LOSSES
        # X-entropy
        CE_loss = tf.nn.softmax_cross_entropy_with_logits(NLL_one_hot, out_pi, axis=-1)
        
        # RWTA
        RWTA_loss = (1-eps) * (M-1/M) * tf.math.reduce_min(mixtures_combined, axis=1) + (eps/M) * tf.math.reduce_sum(mixtures_combined, axis=1)
        
        loss = CE_loss + RWTA_loss
        
        return tf.reduce_mean(loss)
    
    return CE_RWTA_loss