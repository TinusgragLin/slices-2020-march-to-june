# LIN's Machine Learning Note

## Basic Concepts and Terms

1. Dataset

   A dataset is a matrix which consists of $m$ vectors. 

   $ D=[\vec{x_1}, \vec{x_2},\cdots,\vec{x_m}] $

   Each vector has an dimension of $n$, and is called a  **example** or an **instance**.

   $$ \vec{x_k}= [a_{(k)(1)}, a_{(k)(2)},\cdots,a_{(k)(n)}] $$

   Each dimension of such vector describes certain attribute of the instance  $\vec{x_k}$. 

   And for supervised learning method, there is another vector named **label vector**  
   which gives each instance an label to indicate a certain quality of each instance:

   $$ \vec{Y}=[y_{1},y_2,\cdots,y_n] $$

2. Hypothesis Function $H(\vec{x})$ 

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

3. Loss Function $L$ or  Cost Function $C$

   A function which evaluate how well a hypothesis function fits the dataset.  

   Usually we first use the hypothesis function $H$ to label every instance in our dataset then, we evaluate the difference between the real label of and the label predicted by the hypothesis function for each instance and somehow sum the difference to get a general deviation between the hypothesis and the reality(or a perfect approximation of "the function" somehow determines a quality of an instance based on a series of attributes).  

   

   Here are some loss functions that are commonly used:  

   - Mean Squared Error(MSE)  

     $$ L=\frac{1}{n}\cdot \sum_{k=1}^{n}(H(\vec{x})-Y_k)^2 $$

   - Cross-Entropy Loss(or Log Loss)

     $$ L= $$
   
4. Bias-Variance Blance  
