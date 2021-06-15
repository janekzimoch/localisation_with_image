"""
This callbacks are mainly created to make for metrics
such that I can compare models with different loss functions to each other, while they are being trained.
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np

from general_utilis import get_mixture_dist

class model_log_likelihood_callback(keras.callbacks.Callback):
    def __init__(self, model_name, x, y, num_comp): 
        super(model_log_likelihood_callback, self).__init__()
        
        # store data to print at the end
        self.log_likelihood = []
        
        # store data
        self.model_name = model_name
        self.x = x
        self.y = y
        self.num_comp = num_comp
        self.num_datapoints = len(x)
        self.output_dim = y.shape[-1]

        # tensorboard writer
        self.writer = tf.summary.create_file_writer(exp_dir + '/logs/' + model_name + "_log_likelihood")
        self.step_number = 0
        

    def get_metric(self):
        return self.log_likelihood
        
    def load_data(self, data_filename):
        data = np.load(data_filename)
        return data
        
    def on_epoch_begin(self, epoch, logs=None):
        
        # get predictions
        y_pred = self.model.predict(self.x)

        # split into GMM parameters
        mix_comp_logits = y_pred[:, :self.num_comp]
        mus = y_pred[:, self.num_comp:(1+self.output_dim)*self.num_comp]
        sigmas = y_pred[:, (1+self.output_dim)*self.num_comp:]

        # convert logits to categorical distribution - need to itterate through all points
        mix_comp = np.zeros((self.num_datapoints, self.num_comp))
        for i in range(self.num_datapoints):
            mix_comp[i,:] = get_mixture_dist(mix_comp_logits[i,:], self.num_comp)
        
        # get log likelihood
        log_likelihood = 0
        for i in range(self.num_comp):
            for j in range(self.output_dim):
                mse = -0.5*np.sum(mix_comp[:,i]*np.square((self.y[:,j]-mus[:,(i*self.output_dim)+j])/(sigmas[:,(i*self.output_dim)+j]+1e-9)))
                sigma_trace = -np.sum((mix_comp[:,i]+1e-9)*np.log(sigmas[:,(i*self.output_dim)+j]+1e-7))
                log2pi = -np.sum(mix_comp[:,i]*0.5*self.output_dim*np.log(2*np.pi))
                log_likelihood += mse + sigma_trace + log2pi

        avg_log_likelihood = np.round(log_likelihood / self.num_datapoints, 2)
        self.log_likelihood.append(avg_log_likelihood)
        
        print(f'{self.model_name} avg. Log likelihood: {avg_log_likelihood}')
        
        
        
        # save to tensorboard
        with self.writer.as_default():
            tf.summary.scalar('avg_log_likelihood', avg_log_likelihood, step=self.step_number, description=None)
        self.step_number += 1