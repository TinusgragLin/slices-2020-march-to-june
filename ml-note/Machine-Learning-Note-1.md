# Machine Learning Note

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

2. Hypothesis Function $H(a_1,a_2,\cdots,a_n)$ 

   A hypothetical(trial) function which describes how a certain value changes with one or many variables.  

   In machine learning, we often want a function which can correctly label an instance based on a series of attributes.

3. Loss Function $L(a_1,a_2,\cdots, a_n)$ 

   A function which evaluate how well a hypothesis function fits the dataset.  
   
   Usually we first use the hypothesis function $H$ to label every instance in our dataset then, we evaluate the difference between the real label of one instance and the label predicted by the hypothesis function and somehow sum the difference to get a general deviation between the hypothesis and the reality(or a perfect approximation of "the function" somehow determines a quality of an instance based on a series of attributes).