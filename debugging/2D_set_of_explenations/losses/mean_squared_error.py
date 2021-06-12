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



def multiple_heads_MSE_Relaxed_WTA(M, eps):
    weight = 1.0 / M
    
    def custom_loss(y_true, y_pred):
        " This loss is the same as standard MSE, but MSE is applied to each head. "

        losses = []
        for i in range(M):
            y_pred_tmp = y_pred[:,3*i:3*(i+1)]

            loss_tmp = tf.keras.losses.mean_squared_error(y_true, y_pred_tmp)
            losses.append(weight * tf.reduce_mean(loss_tmp))
        
        # WTA approach - choose loss which did the best and compute gradients wrt only this one.
        loss = (1-eps) * (M-1/M) * tf.math.reduce_min(losses) + (eps/M) * tf.math.reduce_sum(losses)
        
        return loss
        
    return custom_loss