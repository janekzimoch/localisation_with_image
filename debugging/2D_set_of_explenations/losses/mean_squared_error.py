import tensorflow as tf

def multiple_heads_MSE(num_heads):
    weight = 1.0 / num_heads
    
    def custom_loss(y_true, y_pred):
        " This loss is the same as standard MSE, but MSE is applied to each head. "

        loss = 0
        for i in range(num_heads):
            y_pred_tmp = y_pred[:,3*i:3*(i+1)]

            loss_tmp = tf.keras.losses.mean_squared_error(y_true, y_pred_tmp)
            loss += weight * tf.reduce_mean(loss_tmp)

        return loss
        
    return custom_loss



def multiple_heads_MSE_WTA(num_heads):
    weight = 1.0 / num_heads
    
    def custom_loss(y_true, y_pred):
        " This loss is the same as standard MSE, but MSE is applied to each head. "

        losses = []
        for i in range(num_heads):
            y_pred_tmp = y_pred[:,3*i:3*(i+1)]

            loss_tmp = tf.keras.losses.mean_squared_error(y_true, y_pred_tmp)
            losses.append(weight * tf.reduce_mean(loss_tmp))
        
        # WTA approach - choose loss which did the best and compute gradients wrt only this one.
        loss = tf.math.reduce_min(losses)

        return loss
        
    return custom_loss



def get_RWTA_for_MSE_loss(M, eps, output_dim, fixed_variance=True):
    """ 
    Note: fixed_variance is not used in this loss, 
    it is just passed to conveniently loop through all the losses 
    """
    weight = 1.0 / M
    
    def RWTA_for_MSE_loss(y_true, y_pred):
        " This loss is the same as standard MSE, but MSE is applied to each head. "

        heads_losses = []
        for i in range(M):
            y_pred_tmp = y_pred[:,output_dim*i:output_dim*(i+1)]

            loss_tmp = tf.keras.losses.mean_squared_error(y_true, y_pred_tmp)
            loss_tmp = tf.expand_dims(loss_tmp, -1)
            heads_losses.append(loss_tmp)
        
        losses_combined = tf.concat(heads_losses, 1)
        
        # WTA approach - choose loss which did the best and compute gradients wrt only this one.
        loss = (1-eps) * (M-1/M) * tf.reduce_min(losses_combined, axis=1) + (eps/M) * tf.reduce_sum(losses_combined, axis=1)
        
        return tf.reduce_mean(loss)
        
    return RWTA_for_MSE_loss