# LIN's Machine Learning Note

### Supervised Learning - Linear Regression

> Linear regression is a **linear model**, e.g. a model that  assumes a **linear relationship between the input variables (x) and the  single output variable (y).** More specifically, that y can be calculated  from a linear combination of the input variables (x).

1. Hypothesis Function

   $$ H=k_1a_1+k_2a_2+\cdots+k_na_n +b=\vec{w}\cdot \vec{x}+b $$

2. Define Our Loss Function

   To estimate our parameters $k_1,k_2,\cdots,k_n$, 

   according to **Ordinary Least Squares Estimate**, we need  to minimize:  

   $$ \sum_{i=1}^{n}(y_i-H(\vec{w},b,\vec{x_i}))^2 $$  

   according to **Maximum Likelihood Estimate**(Gaussian Distribution), we need to maximize:  
  
   $$ F_{likelihood}=-\frac{1}{2\sigma^2}\sum_{i=1}^{n} (y_i-H(\vec{w},b,\vec{x_i}))^2-n\ln \sigma\sqrt{2\pi}$$  
  
   Since $\sigma$ and $n$ is constant, to maximize $F_{likelihood}$ means to minimize: 
  
   $$ \sum_{i=1}^{n} (y_i - H(\vec{w},b,\vec{x_i}))^2 $$
  
   Therefore, we can reasonably define our loss function to be:  
  
   $$ L=\sum_{i=1}^{n}(y_i-H(\vec{w},b,\vec{x_i}))^2 $$
  
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

5. Calculate Gradient in Matrix Notation and Use It

   In practice, we may rewrite everything in matrix notation: 

   $$ H_M=\vec{W}\cdot D $$

   $$ L=(H_M-Y)(H_M-Y)^{T} $$  

   where $D_{(n+1)\times m}=[\vec{X_1},\vec{X_2},\cdots,\vec{X_m}] $ is our modified dataset(or designed matrix), $\vec{W}_{1\times(n+1)}=[k_0,k_1,k_2,\cdots,k_n]$, $\vec{Y}_{1\times m}$ is our label matrix. We can then get the gradient of $L$: 

   $$ (\nabla\cdot L)_{(n+1)\times 1}=2D\cdot (\vec{W}\cdot D-Y)^{T} $$

   If you prefer "closed-form solution", then you make this 0 and solve $\vec{W}$: 
   
   $$\nabla\cdot L=2D\cdot(D^T\cdot\vec{W}^T-Y^T)=0$$  
   
   $$ \vec{W}^T=(D\cdot D^T)^{-1}D\cdot Y^T $$
   
   If you prefer gradient descent, then you update all your parameters at the same time using matrix operation: 
   
   $$ \vec{W}_{new}=\vec{W}_{old}-\alpha(\nabla\cdot L)^T=\vec{W}_{old}-2\alpha(\vec{W}_{old}\cdot D-Y)\cdot D^T $$

6. Stochastic Gradient Descent(SGD)  & Mini-batch Gradient Descent

     

7. Local Weighted Regression  