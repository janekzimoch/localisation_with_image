import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_probability import distributions as tfd



def generate_dataset(dataset_size, num_modes, output_dim, num_patterns, noise):
    # output matrices - these get modified during dataset generation process
    images = np.random.uniform(100-noise,100+noise,size=(1000, 224,224))  # mean=100
    labels = np.zeros((dataset_size, output_dim))
    
    # matrices used to create dataset
    patterns = np.random.uniform(170-noise,170+noise, size=(1000, 80,80)) # mean=170
    label_bank = np.random.uniform(0,1000, size=(num_modes,num_patterns))  # used to be [0,1000]
    pattern_ids = np.random.choice(np.arange(10), size=dataset_size)
    modes = np.random.choice(np.arange(num_modes), size=dataset_size)
    
    # establish pattern locations
    pattern_H = 80
    pattern_W = 80
    pattern_loc_h = np.random.randint(0,224-pattern_H, size=num_patterns, dtype=int)
    pattern_loc_w = np.random.randint(0,224-pattern_W, size=num_patterns, dtype=int)
    
    for i, p_id in enumerate(pattern_ids):
        h = pattern_loc_h[p_id]
        w = pattern_loc_w[p_id]
        images[i, h:h+pattern_H, w:w+pattern_W] = patterns[i]
        
        labels[i,0] = label_bank[modes[i],p_id]
        
    
    return images, labels, modes, pattern_ids


def visualise_data_points(images, labels, modes, pattern_ids):
    """
    This function visualises several (8) datapoints:
    (1): of the same pattern - to check whether they have different modes 'y'
    (2): of the same pattern and mode - to see that images are exactly the same
    """
    a = np.where(modes == 0, 1, 0)
    b = np.where(pattern_ids == 0, 1, 0)
    c= a*b
    
  
    print("same pattern, different modes")
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(16,8))
    axes = (ax1, ax2, ax3, ax4, ax5, ax6)
    for img, lbl, ax in zip(images[b.astype(bool)][:6], labels[b.astype(bool)][:6], axes):
        ax.imshow(img)
        ax.set_title(f'{np.round(lbl[0],2)}')
    plt.show()
    
    
    print("same pattern, same modes")
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(16,8))
    axes = (ax1, ax2, ax3, ax4, ax5, ax6)
    for img, lbl, ax in zip(images[c.astype(bool)][:6], labels[c.astype(bool)][:6], axes):
        ax.imshow(img)
        ax.set_title(f'{np.round(lbl[0],2)}')
    plt.show()


def visualise_mode_overlap(model, x_test, y_test, pattern_ids_test):
    " This function is only compatible with 3 heads - i haven't generalised it yet. "
    plt.figure(figsize=(18,4))

    # PREDICT
    y_pred = model.predict(x_test)

    # PLOT WALL
    N = len(pattern_ids_test)
    start, end = 0, 0
    uniques, counts = np.unique(pattern_ids_test, return_counts=True)
    for i, count in zip(uniques, counts):
        end += count
        indexes = np.where(pattern_ids_test == i)[0]
        plt.scatter(range(start, end), y_test[indexes, 0])

        y_pred_sorted = np.partition(y_pred[indexes], (8,7,6), axis=-1)

        for i in range(3):
            plt.scatter(range(start, end), y_pred_sorted[:,-i-1], color='tab:red', alpha=0.5)      
        start = end
    plt.ylabel('position of wall')
    plt.xlabel('data points')

    # PLOT PREDICTED MODES
    plt.show()



def get_mixture_dist(logits, num_components, numpy=True):
    """
    This function converts logits into a categorical probability distribution
    there are 'num_components' elements in logits array. 
    Note: logits has to be 1D vector (i.e. you have to get categorical distribution for each datapoint seperately)  
    """
    dist = tfd.Categorical(logits=logits)
    n = 1e4
    empirical_prob = tf.cast(
        tf.histogram_fixed_width(
          dist.sample(int(n)),
          [0, num_components-1],
          nbins=num_components),
        dtype=tf.float32) / n
    if numpy:
        return empirical_prob.numpy()
    else:
        return empirical_prob