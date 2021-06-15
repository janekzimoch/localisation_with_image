# multimodel_output_models
 
In this repository i explore different methods of modeling problems where for a given input there are many potential solutions. You can also think of it as one-to-many mapping or input having multimodal output. Standard, discriminative regression machine learning models or classification models with a softmax head are unable to output a set of potential solutions and will either fail at learning the task or will consistently output only single output. 

To model problems with multimodal output we need to output a probability distribution p(y|x) where rather than a scalar our model outputs adsitribution. 
### In this repository we explore following models:
* Mixture Density Networks (MDN) - you can read more about them in C. Bishop's book ["Machine Learning and PAttern Recognition" Chapter 5.6 "Mixture Density networks pages 272-277](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)
* Generative models - constructed using Bayesian Network and Monte Carlo integration (to marginalise out mixture variables) (see notes and derivations

### To fully explore different Multimodal Output Models I create 3 toy-datasets:
* 1. Mixture of Linear Functions
* 2. Mixture of inverted Sinusoidal functions - where each inverted sinusoid may hav multiple y outputs for some x values.
* 3. A custom dataset which is meant to be an extreme simplification of camera relocalisation task which i do for my disertation


### This repository has following notebooks:
* **explore_dataset.ipynb** - this short notebook visualises each dataset - fucntions to generate datasets are imported from generate_dataset.py file
* **generative_model.ipynb** - this notebook: (1) gets dataset, draws out generative framework, (2) approximates conditional probability over x with a neural network, (3) imports functions for evaluating generative model from utilis.gen_model_utilis.py file (3) plots conditional probability p(y|x) and (4) plots joint probability p(x,y). Generative models are very cool because they allow you to evaluate and compare probabilities for different combinations of variables. They also allow you to obtain any conditional distribution. In order to construct a generative model we use a Bayesian Network graph to write down factoristion of joint distribution over all variables. Then by analysing how the dataset is generated, we write down the expressions for each probability distribution and we approximate with NN distributions any distribution of which we don't know the form but have access to training data. Then we multiply probabilities by densly sampling points from each distribution and by using spipy nroamla and unifrom pdf.

**To be continued...**
