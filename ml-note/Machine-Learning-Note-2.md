# LIN's Machine Learning Note

### Supervised Learning - Linear Regression

> Linear regression is a **linear model**, e.g. a model that  assumes a **linear relationship between the input variables (x) and the  single output variable (y).** More specifically, that y can be calculated  from a linear combination of the input variables (x).

- Hypothesis Function

  $$ H=k_1a_1+k_2a_2+\cdots+k_na_n +b=\vec{w}\cdot \vec{x}+b $$

- Our Loss Function

  Using **Ordinary Least Squares Estimate**:  

  $$ L=\sum_{i=1}^{n}(y_i-H(\vec{x}))^2 $$  

   Using **Maximum Likelihood Estimate**:  

  $$ L=\sum_{i=1}^{n} (y_i - H(\vec{x}))^2 $$