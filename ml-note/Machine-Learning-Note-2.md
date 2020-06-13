# LIN's Machine Learning Note

### Supervised Learning - Linear Regression

> Linear regression is a **linear model**, e.g. a model that  assumes a **linear relationship between the input variables (x) and the  single output variable (y).** More specifically, that y can be calculated  from a linear combination of the input variables (x).

1. Hypothesis Function

   $$ H=k_1a_1+k_2a_2+\cdots+k_na_n +b=\vec{w}\cdot \vec{x}+b $$

2. Define Our Loss Function

   To estimate our parameters $k_1,k_2,\cdots,k_n$, 

   according to **Ordinary Least Squares Estimate**, we need  to minimize:  

   $$ \sum_{i=1}^{m}(y_i-H(\vec{w},b,\vec{x_i}))^2 $$  

   according to **Maximum Likelihood Estimate**(Gaussian Distribution), we need to maximize:  
  
   $$ F_{likelihood}=-\frac{1}{2\sigma^2}\sum_{i=1}^{m} (y_i-H(\vec{w},b,\vec{x_i}))^2-n\ln \sigma\sqrt{2\pi}$$  
  
   Since $\sigma$ and $n$ is constant, to maximize $F_{likelihood}$ means to minimize: 
  
   $$ \sum_{i=1}^{m} (y_i - H(\vec{w},b,\vec{x_i}))^2 $$
  
   Therefore, we can reasonably define our loss function to be:  
  
   $$ L=\sum_{i=1}^{m}(y_i-H(\vec{w},b,\vec{x_i}))^2 $$
  
3. Minimize Our Loss Function  

   Since our loss function is a function with $n+1$ variables($k_1,k_2,\cdots,k_n,b$), to minimize this function is to find a local or global minimum of $Z=L(\vec{w},b)$. Now we can use our mathematical knowledge.  

   Basically, to do this, we need to take the partial derivative of $Z$ to each variable and try to make every equation equals to zero. So we will have $n+1$ equations with $$ n+1 $$ unknowns and by using some linear algebra knowledge, we can then determine all the parameters in which case will make our loss function goes down to a minimum. And this whole process is called **Ordinary Least Square Estimate**. More detail at the supplement down below. 

   Or we can adopt another way called **Gradient Descent** which has something to do with the **gradient vector** of a scalar function. In short, the gradient vector of a scalar function at a certain point is a **direction vector** pointing at the direction in which the value of the function **increases** most rapidly.  

   That means, if we updates our variables in such a way that the direction of our "moving" is against the direction of the gradient vector at the current "point", then what we are doing is like "going down a hill", therefore, after we continue doing this for a while, we may land at the local or global lowest "point" e.g. the local or global minimum.  

   So basically, that is what we are going to do:  

   $$ (k_{1}^{\prime},k_{2}^{\prime},\cdots,k_{n}^{\prime},b^{\prime})=(k_1,k_2,\cdots,k_n,b)+\alpha[-\vec{\nabla}\cdot H] $$

   where $\vec{\nabla}=(\frac{\part}{\part k_1},\frac{\part}{\part k_2},\cdots,\frac{\part}{\part k_n},\frac{\part}{\part b})$, and $\alpha$ is the so called the "learning rate". If we have only three variables $k_1,k_2,b$, it is the length of the xOy projection of the trajectory we "walked" in the plane.
   
   As you may have noticed, as we approach a local or the global minimum, the $-\nabla\cdot H$ term approaches $(0,0,\cdots,0)$, in other word, the step we are going to make is becoming smaller and smaller as we approach the minimum. So it will be reasonable if we set a minimum step distance, the whole process comes to a halt when the step we make is smaller than the minimum step distance.  
   

**NOTE & SUPPLEMENT:**  

1. When we want write our hypothesis function as $ H=k_1a_1+k_1a_2+k_2a_1^2+k_2a_2^2+b $ we are actually doing this:  $a_3=a_1^2,a_4=a_2^2$.  

2. Another way we can accelerate the gradient descent process is trying to make the value ranges of all the attributes as uniform as possible by "scaling" some attributes. We call this trick "feature-scaling".

3. Be careful when you are trying to implement gradient descent algorithm, the $H$ in $-\nabla\cdot H$ must be the **old** $H$ where all the **old** parameters($k_1,k_2,\cdots,k_n,b $) remain the same.

4. In practice, for simplicity, we may consider $b$ as $k_0$, and add an extra dimension $a_0$ which equals to 1 to our $\vec{x}$ vector. And define: 

   $\vec{W}=[k_0,k_1,k_2,\cdots,k_n]$  and  $\vec{X}=\begin{bmatrix}a_0\\a_1\\a_2\\\vdots\\a_n\end{bmatrix}$. So now our hypothesis function can be simplified as :  

   $$ H=\vec{W}\cdot \vec{X}=[k_0,k_1,k_2,\cdots,k_n]\cdot\begin{bmatrix}a_0\\a_1\\a_2\\\vdots\\a_n\end{bmatrix}$$  
   
   Accordingly, our dataset $D$ is now: 
   
   $$ D=[\vec{X_1},\vec{X_2},\cdots,\vec{X_m}] $$

5. Calculate Gradient in Matrix Notation and Use Matrix Opration

   In practice, we may rewrite everything in matrix notation: 

   $$ H_M=\vec{W}\cdot D $$

   $$ L=(H_M-Y)(H_M-Y)^{T} $$  

   where $D_{(n+1)\times m}=[\vec{X_1},\vec{X_2},\cdots,\vec{X_m}] $ is our modified dataset(or designed matrix), $\vec{W}_{1\times(n+1)}=[k_0,k_1,k_2,\cdots,k_n]$, $\vec{Y}_{1\times m}$ is our label matrix. We can then get the gradient of $L$ : 

   $$(\nabla\cdot L)_{(n+1)\times 1}=\frac{\part L}{\part \vec{W}}=2D\cdot (\vec{W}\cdot D-Y)^{T} $$

   If you prefer "closed-form solution" or "normal equation", then you make this 0 and solve $\vec{W}$: 
   
   $$\nabla\cdot L=2D\cdot(D^T\cdot\vec{W}^T-Y^T)=0$$  
   
   $$ \vec{W}^T=(D\cdot D^T)^{-1}D\cdot Y^T $$
   
   If you prefer gradient descent, then you update all your parameters at the same time using matrix operation: 
   
   $$ \vec{W}_{new}=\vec{W}_{old}-\alpha(\nabla\cdot L)^T=\vec{W}_{old}-2\alpha(\vec{W}_{old}\cdot D-Y)\cdot D^T $$

6. **Stochastic Gradient Descent(SGD)  & Mini-batch Gradient Descent**  

   There are alternatives to batch gradient descent. While batch gradient descent need to take the derivative of the overall loss, stochastic gradient descent uses the gradient of the loss for just one instance in dataset while mini-batch gradient descent, which is often used in ANN, uses the gradient of the loss of just a part of training instances to update the parameters. 

   So in **Stochastic Gradient Descent**, instead of taking the derivative of $ L=\sum_{i=1}^{m}(y_i-H(\vec{W},\vec{x_i}))^2 $, we just take the derivative of $L_i=(y_i-H(\vec{W},\vec{x_i}))^2$, which is: 

   $$ \frac{\part L_i}{\part k_j}=2a_k^{(i)}(H(W,\vec{x_i})-y_i) $$

   or more generally in matrix notation: 

   $$(\nabla L_i)_{(n+1)\times1}=2\vec{X_i}(\vec{W}\cdot \vec{X_i}-y_i)$$

   So you select the $i$-th instance in your dataset and update your parameters by: 

   $$ \vec{W}_{new}=\vec{W}_{old}-2\alpha \vec{X_i}^T(\vec{W}_{old}\cdot \vec{X_i}-y_i)$$  

   *(But somehow, when people are talking about SGD in ANN, they are actually talking about mini-batch gradient descent.)*  

   Basically, in **Mini-batch Gradient Descent**, you first divides your dataset to many *batches* so that each batch have the same number(called "batch size", usually denoted as $b$) of training examples that are **randomly** picked out from the dataset except for the last batch, which might have a smaller number of training examples.  

   In practice, this step might be done by first shuffle the whole dataset and then divide the shuffled dataset to $m|b$ batches, and give the rest $(m-m|b)$ training examples(if any) to the last batch. 

   For each iteration, you update your parameters $W$ to minimize the loss function for just one batch instead of the whole dataset. That is, using the gradient of the loss function for just one batch  to update our parameters $W$. 

   Every time this process scans through all the batches, we call it one epoch. In other words, the algorithm scans through the whole dataset once in one epoch. 

   Now, for each iteration, your loss function becomes: 

   $$ L_{batch[k]}=\sum_{i=1+(k-1)b}^{kb}(H(\vec{W},\vec{X}_{i})-y_{i})^2 $$

   Or you can store all the $X$ in $k$-th batch to $(B_k)_{(n+1)\times b}$ and all the $y$ to $(Y_{B_k})_{1\times b}$. Then the equation above can be expressed in matrix notation: 

   $$L_{batch[k]}=(W\cdot B_k-Y_{B_k})(W\cdot B_k-Y_{B_k})^T$$

   So the updating equation would be: 

   $$ \vec{W}_{new}=\vec{W}_{old}-\alpha(\nabla\cdot L_{batch[k]})^T=\vec{W}_{old}-2\alpha(\vec{W}_{old}\cdot B_k-Y_{B_k})\cdot {B_k}^T $$  

8. Details about Using **Maximum Likelihood Estimate** to Derive Loss Function(which I still cannot fully comprehend. :cry: ​)  

   First assume that the function the really determined a certain quality of one instance is : 

   $$ y_i=\vec{W}\cdot\vec{X}_i+\gamma_i $$

   where $\gamma_i$ is the error term for one specific instance that takes **un-modeled effects and random nosies** into account.  

   Then, according to **central limit theorem**, it turns out that the error term is very likely distributed as standard normal distribution : 

   $$ \gamma \sim \mathcal{N}(0,\sigma^2)$$

   that would means the probability density function of $\gamma$ is: 

   $P(\gamma)=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(\gamma-0)^2}{2\sigma^2})$  

   Then we make a assumption that all these error terms are **independently and identically distributed** which means these errors are  identically distributed and are independent from each other. (this would probably indicate that all the random variable $y_i$ are also IID.)  

   And all these above implies that (here $P((y_i|\vec{X}_i)_{W})$ means the distribution of $y_i$ given $\vec{X}_i$ with parameters $W$): 

   $$P((y_i|\vec{X}_i)_{W})=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y_i-W\cdot \vec{X}_i)^2}{2\sigma^2})$$
   
   i.e.
   
   $$ (y_i|\vec{X}_i)_{W} \sim \mathcal{N}(W\cdot\vec{X},\sigma^2) $$
   
   which means that the random variable $y_i$ is distributed as Gaussian distribution with the mean being $W\cdot \vec{X}$ and the variance being $\sigma^2$. 
   
   Now that we know that for all $y_i$, $y_i \sim \mathcal{N}(W\cdot\vec{X},\sigma^2)$ and the random variables $y_i$ are IID, it is when we should use **Maximum Likelihood Estimation** to estimate the parameters $W\cdot \vec{X}$ so that the random variables $y_i$ are distributed exactly the way they are.  
   
   So, as the first step of MLE, the joint probability for all $y_i$ or likelihood function of $y$ would be:  
   
   $$\begin{align}L(W)&=\prod_{i=1}^{m}P((y_i|\vec{X}_i)_{W})\\&=\prod_{i=1}^{m} \frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(y_i-W\cdot \vec{X}_i)^2}{2\sigma^2})\end{align}$$
   
   Then, the log likelihood function would be: 
   
   $$\begin{align}LL(W)&=log[L(W)]\\&=\sum_{i=1}^{m}log\frac{1}{\sqrt{2\pi}\sigma} \exp(-\frac{(y_i-W\cdot  \vec{X}_i)^2}{2\sigma^2})\\&=\sum_{i=1}^{m}[-log (\sqrt{2\pi}\sigma)-\frac{(y_i-W\cdot  \vec{X}_i)^2}{2\sigma^2}]\end{align}$$
   
   And the last step is just maximizing the log likelihood, which is the same as minimizing the $\sum_{i=1}^{m}(y_i-W\cdot\vec{X})^2$ term. 
   
7. Local Weighted Regression  

   While the linear regression tries to find a function to fit the entire training set, there is  another regression algorithm called Locally Weighted Regression which focus **mainly** the part of the dataset that is relatively **"close"** to the query instance. 

   In other words, this algorithm does not try to find a set of parameters that minimizes the loss function so that you can predict a certain quality for a new instance with a parameterized function. Instead, every time a new query instance comes,  it just look around and mainly focus on the "vicinity" of the query instance in the dataset, and fit a model that is good enough in the vicinity of the query instance, then make a prediction according to the model. 

   So basically what we are going to do is trying to use a function to down-weight the instances that are "far  away" from the query instance, so the function must decrease when the "distance" between two instance increases. 

   The way you define "distance" varies from one to another, we mostly often uses Euclidean Distance. 

   One weight function commonly used is Gaussian function: 

   $$  K_{Gauss}(\vec{a},\vec{b})=e^{-\frac{||\vec{a}-\vec{b}||^2}{2\sigma^2}}  $$  

   So instead of minimizing $ L=\sum_{i=1}^{n}(y_i-H(\vec{W},\vec{x_i}))^2 $, we now want to minimize: 

   $$  L=\sum_{i=1}^{n}w(\vec{x}_q,\vec{x}_i)\cdot(y_i-H(\vec{W},\vec{x_i}))^2  $$  

   where $w(\vec{a},\vec{b})$ is our weight function. 

9. Some **Optimal Methods** when Using Gradient Descent  

   - Momentum  (to address the ravine oscillation issue)

     - Original Momentum  

     - Nesterov's Momentum(Nesterov's Accelerated Gradient, GAD)
     
   - **Adaptive Learning Rate (for different parameters)**
      - Adaptive Gradient(adagrad) , Adadelta , RMSprop ......

   - Composite(combining momentum trick and adaptive learning rate trick)  

     - Adaptive Moment Estimation(Adam)     

   more at [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/index.html)