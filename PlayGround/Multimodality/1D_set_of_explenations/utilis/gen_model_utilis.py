import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

import numpy as np
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



####################
###   NN MODEL   ###
####################

def simple_model(input_shape=1):
    """ returns a simple NN to model p(x|y) """

    input_ = layers.Input(shape=(input_shape))
    x = layers.Dense(100, activation='relu')(input_)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dense(50, activation='relu')(x)
    y = layers.Dense(1)(x)
    model = Model(inputs = input_, outputs = y)

    # compile
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='mean_squared_error')
    return model

#####################################
###   PROBABILITY DISTRIBUTIONS   ###
#####################################

def get_probability(arguments, dataset_type="sin_or_lin_dataset"):
    """ Arguments are all passed to get_unnormalised_posterior__custom_dataset() function
    
    x_query: 
        if scalar: this function outputs conditional probabilty p(y|x)  -  probability
        if vector: this function outputs a joint probability p(y,x)  -  heat_map
    model: Neural Network approximating p(x|y,c) or p(x|c)
    num_clusters: number of mixtures/components - variable needed when marginalising
    y_range: range of y values for which distribution should be computed 
    N: number of y_points to space evenly in y_range 
    c_mins, c_maxs: vectors which define prior for each class p(c_i) = Uniform(c_mins[i], c_maxs[i])  """
    
    # compute unnormalised posterior
    if dataset_type == "sin_or_lin_dataset":
        # arguments: [x_query, model, num_clusters, y_range, N, y_shifts]
        unnormalised_probability = get_unnormalised_probability__sin_or_lin_dataset(*arguments) 
    
    elif dataset_type == "custom_dataset":
        # arguments: [x_sample, model, num_components, y_range, N_y, c_mins, c_maxs]
        unnormalised_probability = get_unnormalised_probability__custom_dataset(*arguments)
    

    # normalise posterior (applied to both conditional probability and joint probability)
    probability = unnormalised_probability / np.sum(unnormalised_probability)
    return probability


##################################################################################
###   FUNCTIONS TO COMPUTE Cond. PROBABILITY   -  SINUSOIDAL/LINEAR datasets   ###
##################################################################################
 
def get_unnormalised_probability__sin_or_lin_dataset(x_query, model, num_components, y_range, N, y_shifts):
    
    min_y, max_y = y_range
    freq = (max_y - min_y) / N
    y_points = np.arange(min_y, max_y, freq)  # choose y points to evaluate p(y|x=x_query) for
    D = len(y_points)
    M = len(x_query)
    
    # get unnormalised probability p(y|x)
    unnormalised_probability__c_marginalised = np.zeros((D,M))
    for c in range(num_components):  # sum over all mixture components to marginalise c out
        if num_components > 1:
            cluster_one_hot = np.zeros((len(y_points), num_components))
            cluster_one_hot[:,c] = 1
            model_input = np.concatenate([y_points.reshape(-1, 1), cluster_one_hot], axis=-1)
        else:
            model_input = y_points
        
        x_mean = model.predict(model_input).flatten()  # assume p(x|y) is gaussian N(f(y), 0.1) => predict means


        # compute nominator
        unnormalised_probability = np.zeros((D,M))  # 
        for i in range(D):  # for each y_point evaluate p(y|x=x_query)
            p_y_c = norm.pdf(y_points[i], loc=y_shifts[c], scale=1)  # probability of y point given that c has generated y
            p_x_y_c = norm.pdf(x_query, loc=x_mean[i], scale=0.5)  # p(y|c) * p(x|y,c) = p(x,y|c)   (we multiply by p(c) later)
            unnormalised_probability[i,:] = p_y_c * p_x_y_c  # p(y,x=x_query|c) = p(y|x=x_query,c) 

        unnormalised_probability__c_marginalised += (1 / num_components) * unnormalised_probability  # marginalise c out p(c) = 1/num_sinusoids
    
    return unnormalised_probability__c_marginalised




###################################################################
###   FUNCTIONS TO COMPUTE Cond. PROBABILITY  -  CUSTOM dataset ###
###################################################################

def get_unnormalised_probability__custom_dataset(x_query, model, num_clusters, y_range, N, c_mins, c_maxs):
    " x_query can be a scalar for likelihood plot or a vector for a heatmap "

    # get y_points for which we will evaluate p(y|x=x_query)
    min_y, max_y = y_range
    freq = (max_y - min_y) / N
    y_points = np.arange(min_y, max_y, freq)
    D = len(y_points)
    M = len(x_query)
    
    # compute unnormalised likelihood: SUM^c [ p(y,c|x=x_query) ] = p(y|x=x_query) 
    unnorm_probability__c_marginalized = np.zeros((D,M))
    for c in range(num_clusters):  
        # itterate over all clusters and sum their probabilities (marginalise c out) 
        
        cluster_one_hot = np.zeros((len(y_points), num_clusters))
        cluster_one_hot[:,c] = 1        
        x_mean = model.predict(cluster_one_hot).flatten()  # assume p(x|c) is gaussian N(x; f(c), 0.5) => predict means
        
        unnormalised_probability = np.zeros((D,M))
        
        for i in range(D):
            p_c_y = np.where(c_mins[c] < y_points[i] < c_maxs[c], 1, 0)
            p_x_c = norm.pdf(x_query, loc=x_mean[i], scale=0.1)
            unnormalised_probability[i] = p_c_y * p_x_c

        unnorm_probability__c_marginalized += (1 / num_clusters) * unnormalised_probability  # marginalise c out p(c) = 1/num_sinusoids
    
    return unnorm_probability__c_marginalized  # return posterior with category marginalised out



##################
###  PLOTING   ###
##################

def plot_heat_map(heat_map, x_range, y_range, N_x, N_y, tick_freq=100):
    """
    tick_freq - this will determine how many ticks you get in your plot.
    so if you have high N_x and N_y then set tick_freq proportionally high
    """

    fig, ax = plt.subplots(figsize=(12,6))
    # normalise to pop-colors
    # heat_map = (heat_map - heat_map.min()) / (heat_map.max() - heat_map.min())
    im = ax.imshow(heat_map)
    
    # x ticks
    num_x_ticks = int(N_x / tick_freq)
    x_ticks = np.arange(0,N_x,tick_freq)
    x_labels = np.round(np.arange(x_range[0], x_range[1], (x_range[1] - x_range[0]) / num_x_ticks),1)
    plt.xticks(x_ticks, x_labels)

    # y ticks
    num_y_ticks = int(N_y / tick_freq)
    y_ticks = np.arange(0,N_y,tick_freq)
    y_labels = np.round(np.arange(y_range[0], y_range[1], (y_range[1] - y_range[0]) / num_y_ticks),1)
    plt.yticks(y_ticks, y_labels)
    plt.gca().invert_yaxis()
    plt.title('joint pdf p(y,x)')
    plt.ylabel('y')
    plt.xlabel('x')
    
    # add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    plt.show()



def plot_probability(p_y_given_x, y_range, N_y, y_data, x_data, x_sample, cluster_assignment):
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax2 = ax1.twinx() 
    points = np.arange(y_range[0], y_range[1], (y_range[1] - y_range[0]) / N_y)
    num_points = len(points)

    num_categories = cluster_assignment.max() + 1
    for i in range(num_categories):
        indexes = np.where(cluster_assignment == i)[0]
        ax2.scatter(y_data[indexes], x_data[indexes])

    ax1.plot(points, p_y_given_x, label='p(y|x)')
    ax2.plot(points, [x_sample]*num_points, '--', alpha=0.5, color='red', label='x sample')
    ax1.set_ylabel('p(y|x)')
    ax2.set_ylabel('x axis')
    ax1.set_xlabel('y axis')

    ax1.legend(loc=2)
    ax2.legend(loc=1)
    plt.show()