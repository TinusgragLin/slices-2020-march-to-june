# LIN's Machine Learning Note

## Basic Concepts and Terms

#### 1. Dataset

   A dataset is a matrix which consists of $m$ vectors. 

   $ D=[\vec{x_1}, \vec{x_2},\cdots,\vec{x_m}] $

   Each vector has an dimension of $n$, and is called a  **example** or an **instance**.

   $$ \vec{x_k}= [a_{(k)(1)}, a_{(k)(2)},\cdots,a_{(k)(n)}] $$

   Each dimension of such vector describes certain attribute of the instance  $\vec{x_k}$. 

   And for supervised learning method, there is another vector named **label vector**  
   which gives each instance an label to indicate a certain quality of each instance:

   $$ \vec{Y}=[y_{1},y_2,\cdots,y_n] $$

#### 2. Hypothesis Function $H(\vec{x})$ 

   A hypothetical(trial) function which describes how a certain value changes with one or many variables.  

   In machine learning, we often want a function which can correctly label an instance based on a series of attributes.

   Note that the hypothesis function is by definition a function of $n$ attributes $a_1, a_2, \cdots,a_n$, but since we are now interested in finding a function that somehow well fits our database, we may at first come up with a guess(trail) function that may not fits the database very well, say we come up with a guess function like this:  

   $H(a_1,a_2,\cdots,a_n)=k_1a_1+k_2a_2+\cdots +k_na_n+b$

   which is an linear function of $a_1,a_2,\cdots,a_n$ and sometimes, for simplicity, we may rewrite this as:

   $H(a_1,a_2,\cdots,a_n)=\vec{w}\cdot \vec{x}+b $ 

   in which case (linear model) vector $\vec{w}$ is called **weight vector**, which is not so difficult to understand, as $\vec{w}$ stores the contribution(weight) to the function output of each attribute of an instance $\vec{x}$.

   Or we may came up with a more complicated function like this:

   $H(a_1,a_2,\cdots,a_n)=e^{k_1a_1+k_2a_2+\cdots +k_na_n + b}=e^{\vec{w}\cdot \vec{x}+b}$   

   One way or another, we find that our hypothesis function $H$ is also a function of a series of **parameters** and only with a certain set of parameters can we determine our hypothesis function and use it to predict a certain quality of an instance.

   In this way, our hypothesis function is also a function of a series of parameters:  

   $$ H(\vec{x},\vec{k})\,\,or\,\, H(\vec{x},\vec{w}) $$

#### 3. Loss Function $L$ or  Cost Function $C$

   A function which evaluate how well a hypothesis function fits the dataset.  

   Usually we first use the hypothesis function $H$ to label every instance in our dataset then, we evaluate the difference between the real label of and the label predicted by the hypothesis function for each instance and somehow sum the difference to get a general deviation between the hypothesis and the reality(or a perfect approximation of "the function" somehow determines a quality of an instance based on a series of attributes).  

   

   Here are some loss functions that are commonly used:  

   - Mean Squared Error(MSE)  

     $$ L=\frac{1}{n}\cdot \sum_{k=1}^{n}(H(\vec{x})-Y_k)^2 $$

   - Cross-Entropy Loss(or Log Loss)

     $$ L= $$

#### 4. Bias-Variance Balance  

In machine learning, a model with high **bias** has such a strong bias for a certain assumption about the dataset that it can not see the "truth" about the dataset. For example, in logistic regression, if your dataset is actually not linear separable, but your hypothesis is a linear function about primal features, then your decision boundary will end up always be a line so it seems that the algorithm has such a strong bias for the dataset being linear separable that it can not see the fact that it is not linear separable.  

And a model with high variance is a model that fits another dataset that is similarly distributed to the training set(could just be different parts of a large dataset or datasets about the same objects collected by different people) badly so the performance varies greatly. 

In other words, a model with high bias does not fit the training set well, and a model with high variance is not a generalized model. 

In machine learning, on one hand we want the model to fit the training set well, on the other hand we want it to be a generalized model so that it could have good performance in all other datasets. 

But if your model fits the training set too well, it then has a preference for the training set so some random noises in the training set might be considered too significant, thus it might not perform well in other dataset.  

#### 5. Norm Regularization  

It turns out that in the case of high variance model, the fitting curve or decision boundary become very wiggly and far away from smoothing because it wants to fit every noises. That could result in derivatives of the curve at most points being very large, which indicates that the coefficients i.e. the parameters are large while in the case of low variance model, the fitting curve or decision boundary is relatively smoothing so for the same reason the parameters tend to have small values. 

So, one way to lower the variance of a model is trying to lower the norm of the parameter vector. But of course, you want to fit the training set well, so you also want to lower the loss function or lift up the probability of model. We can express these two goals in minimizing: 

$$Loss+\lambda||\vec{w}||$$

or maximizing

$$P_{\vec{w}}(y|\vec{x})-\lambda||\vec{w}||$$

where $\lambda$ is the balance factor, depending on whether you want lower variance or lower bias, you can make $\lambda$ large or small. Here $||\vec{w}||$ is the norm of vector $\vec{w}$, normally we would choose $\mathcal{l}_{1}$ norm: $(\sum_{i=1}^{d}|x_i|^{1})^{1}$ or $\mathcal{l}_{2}$ norm: $(\sum_{i=1}^{d}|x_i|^{2})^{\frac12}$.  

*Derivation of Regularization Term: It turns out that using Maximum A Posteriori Estimation and assume the prior probability of parameters to be Gaussian, you can derive the regularization term for logistic regression*

#### 6. Cross Validation

You can tell how well the model has fit the training set by the computing predicting error on the training set. 

But what we want is to get a universal model through capturing some of  the major features of the training set that indicates some general facts about what the dataset is revealing. 

So, we have to know the generalization capacity of a model, so that we can choose better hyper parameters or a more generalized model.   

To do this, you need to compute the predicting errors on some other different datasets that have similar distributions but not identical to the training set.  

- Simple Hold-out Cross Validation\
  Say, you have a dataset $D$, one way we do this validation is to randomly split the whole dataset into three pieces $D_{train}$, $D_{dev}$ and $D_{test}$ (or simply $D_{train}$ and $D_{test}$).
  
  Very often people divides in 60/20/20 but the principle is that $D_{dev}$ and $D_{test}$ should be large enough for being able to tell the difference of a good model and bad model.
  
  Basically what you are going to do is just to choose a certain set of parameters and a model at first, train your model on the $D_{train}$, and see what the predicting error is on $D_{dev}$(or $D_{test}$), you can then adjust the parameters or the model accordingly until the error is low enough. Note that you should keep the errors on the both $D_{train}$ and $D_{dev}$ reasonably low, or you could easily over-fit one of the two dataset. Finally you could run your final test on the $D_{test}$ and see the outcome.


- $k$-fold Cross Validation\
  If you have a relatively small dataset, using the method above might end up having a unacceptably small training set. There is another method called $k$-fold cross validation.

  The basic idea is that first randomly divide your dataset into $k$(usually $10$) pieces, hold out one of the $k$ sub dataset, train you model on the rest and compute the predicting error on the held-out sub dataset and again pick out another sub dataset, train on the rest and compute the error on the picked-out dataset. Do this $k$ times in total and compute the average error.

  Then you can again adjust the model and hyper parameters until the average error is reasonably low. 


#### 7. Feature Selection

Sometimes among the features you have on your dataset, some of them are redundant or actually irrelevant to the quality you want to predict. 

The basic idea of feature selection is to see a feature does or does not have a significant influence in predicting a certain quality, and leave or drop that feature accordingly. 

There are two ways to do this, one is called forward search which add one feature to the feature list if it matters, another one is called backward search which drop one feature from the feature list if it does not matters. 

So basically, for forward search, you start with a empty list of feature and try adding one of the feature outside the feature list, train your model with the current feature list and see the change in the predicting error until you find the one which decreased the error the most, and do this until you feel you have enough features in the feature list or no other feature seems significant.
