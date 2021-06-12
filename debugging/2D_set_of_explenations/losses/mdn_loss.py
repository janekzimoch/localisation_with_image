import tensorflow as tf
from tensorflow_probability import distributions as tfd


def get_mdn_loss(num_comp, output_shape=3, fixed_variance=True):
    """ 
    As implemented here: https://github.com/cpmpercussion/keras-mdn-layer/blob/master/mdn/__init__.py
    get MDN loss function defined by a GMM which consists of 'num_components' gaussian mixtures 
    and provides distribution over 'output_dim' variables 
    """
    num_mixes = num_comp
    output_dim = output_shape
    
    
    def mdn_loss(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + num_mixes], name='reshape_ypreds')
        y_true = tf.reshape(y_true, [-1, output_dim], name='reshape_ytrue')
        # Split the inputs into paramaters
        out_pi, out_mu, out_sigma = tf.split(y_pred, num_or_size_splits=[num_mixes,
                                                                         num_mixes * output_dim,
                                                                         num_mixes * output_dim],
                                             axis=-1, name='mdn_coef_split')
        # Construct the mixture models
        cat = tfd.Categorical(logits=out_pi)

        component_splits = [output_dim] * num_mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        if fixed_variance:
            sigs = [[1]] * num_comp
        else:
            sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)

        coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                in zip(mus, sigs)]
        mixture = tfd.Mixture(cat=cat, components=coll)

        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
        return loss
    
    return mdn_loss


# THIS IS MY IMPLEMENTATION OF MDN    -   IT ALSO WORKS

# def mdn_loss_mine(num_comp, output_dim=1):
#     " my implementation "
    
#     def get_mdn_loss(y_true, y_pred):
        
#         mix_comp = y_pred[:,0:num_comp], 
#         means = y_pred[:,num_comp:(output_shape+1)*num_comp]
#         variances = y_pred[:,(output_shape+1)*num_comp:]
#         variances_fake = [[1]]
       
#         # get categorical distribution over GMM mixtures
#         cat = tfd.Categorical(logits=mix_comp[0])
        
#         component_splits = [output_shape] * num_comp
#         means = tf.split(means, num_or_size_splits=component_splits, axis=1)
#         variances = tf.split(variances, num_or_size_splits=component_splits, axis=1)
        
#         # build multivariate gaussians
#         coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale in zip(means, variances_fake)]
        
#         # combine 'cat' categorical distribution and multivariate gaussians 'coll' ina GMM model
#         GMM = tfd.Mixture(cat=cat, components=coll)
            
#         # calculate log probability
#         loss = GMM.log_prob(y_true)
#         loss = tf.negative(loss)
#         loss = tf.reduce_mean(loss)
 
#         return loss
    
#     return get_mdn_loss