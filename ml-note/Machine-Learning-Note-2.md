# LIN's Machine Learning Note

### Supervised Learning - Linear Regression

> Linear regression is a **linear model**, e.g. a model that  assumes a **linear relationship between the input variables (x) and the  single output variable (y).** More specifically, that y can be calculated  from a linear combination of the input variables (x).

- Hypothesis Function

  $$ H=k_1a_1+k_2a_2+\cdots+k_na_n +b=\vec{w}\cdot \vec{x}+b $$

- Define Our Loss Function

  To estimate our parameters $k_1,k_2,\cdots,k_n$, 

  according to **Ordinary Least Squares Estimate**, we need  to minimize:  

  $$ \sum_{i=1}^{n}(y_i-H(\vec{w},b,\vec{x_i}))^2 $$  

  according to **Maximum Likelihood Estimate**(Gaussian Distribution), we need to maximize:  
  
  $$ F_{likelihood}=-\frac{1}{2\sigma^2}\sum_{i=1}^{n} (y_i-H(\vec{w},b,\vec{x_i}))^2-n\ln \sigma\sqrt{2\pi}$$  
  
  Since $\sigma$ and $n$ is constant, to maximize $F_{likelihood}$ means to minimize: 
  
  $$ \sum_{i=1}^{n} (y_i - H(\vec{w},b,\vec{x_i}))^2 $$
  
  Therefore, we can reasonably define our loss function to be:  
  
  $$ L=\sum_{i=1}^{n}(y_i-H(\vec{w},b,\vec{x_i}))^2 $$
  

- Minimize Our Loss Function  

  Since our loss function is a function with $n+1$ variables($k_1,k_2,\cdots,k_n,b$), to minimize this function is to find a local or global minimum of $Z=L(\vec{w},b)$. Now we can use our mathematical knowledge.  

  Basically, to do this, we need to take the partial derivative of $Z$ to each variable and try to make every equation equals to zero. So we will have $n+1$ equations with $$ n+1 $$ unknowns and by using some linear algebra knowledge, we can then determine all the parameters in which case will make our loss function goes down to a minimum. And this whole process is called **Ordinary Least Square Estimate**.  

    

  Or we can adopt another way called **Gradient Descent** which has something to do with the **gradient vector** of a scalar function. In short, the gradient vector of a scalar function at a certain point is a **direction vector** pointing at the direction in which the value of the function **increases** most rapidly.  

  That means, if we updates our variables in such a way that the direction of our "moving" is against the direction of the gradient vector at the current "point", then what we are doing is like "going down a hill", therefore, after we continue doing this for a while, we may land at the local or global lowest "point" e.g. the local or global minimum.  

  So basically, that is what we are going to do:  

  $$ (k_{1}^{\prime},k_{2}^{\prime},\cdots,k_{n}^{\prime},b^{\prime})=(k_1,k_2,\cdots,k_n,b)+[-\vec{\nabla}\cdot H] $$

  where $\vec{\nabla}=(\frac{\part}{\part k_1},\frac{\part}{\part k_2},\cdots,\frac{\part}{\part k_n},\frac{\part}{\part b})$  

  