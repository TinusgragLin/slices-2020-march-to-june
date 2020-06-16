# LIN's Machine Learning Note

## Support Vector Machine

#### Optimal Margin Classifier

Before we start discussing optimal margin classifier, let us first look back at logistic regression. 

Say each instance in your dataset has only two attributes and your dataset looks like this:  

![logistic-regression](/home/lin/GitRepo/summary/ml-note/logistic-regression.png)

where $x_1,x_2$ axises each indicates one attribute of an instance and here a square stands for one positive instance while a triangle is one negative instance.  

And what LR is doing is simply trying to find a line (your $z=\vec{w}\cdot\vec{x}+b$) that can separate positive and negative examples. 

So logistic regression algorithm may come up with a decision boundary like the green line below which indeed perfectly separates the two classes: 

![logistic-regression-decision-boundary](/home/lin/GitRepo/summary/ml-note/logistic-regression-decision-boundary.png)

But the red line, which also separates the two classes but has a larger distance to all the points than the green line, seems to have done a better job while the green decision boundary is so close to some of the points that with a little rotation, it could just mis-classify these points.  

Or, looking back to logistic regression, when a point is very close to the decision boundary, $z$ is very close to $0$, which means $g(z)$ is very close to 0.5 and since $g(z)$ is the probability of $y=1$, this means that the algorithm is not so sure about $y=1$ or $y=0$. In contrast, when a point is quite far away from the decision boundary, $g(z)$ is very close to 1 or 0, indicating that the algorithm is quite sure about $y$ being 1 or 0. So we certainly want the distance between the decision boundary and all the points as larger as possible. 

So, how can we get such a decision boundary?  

Well, since we are going to make the distance(or "margin") from the decision boundary to each example point larger,  we have to measure the distance. 

To get the distance, let us first expand our two dimensional problem to three dimensional in which case $\vec{w}\cdot\vec{x}+b=0$ would be a plane.

And we know that a unit normal vector to a plane $F(\vec{x})=0$ would be: 

$$\hat{n}=\frac{\nabla_{\vec{x}} F(\vec{x})}{||\nabla_{\vec{x}} F(\vec{x})||}=\frac{(F_{x_1},F_{x_2},F_{x_3})}{\sqrt{F_{x_1}^2+F_{x_2}^2+F_{x_3}^2}}$$ 

Now $F(\vec{x})=\vec{w}\cdot\vec{x}+b$, so $\hat{n}=\frac{\vec{w}^T}{||\vec{w}||}$

And for any point $\vec{p}$ that is outside the plane, take any point $\vec{q}$ in the plane which satisfies $F(\vec{q})=0$, the distance between the plane and point $\vec{p}$ would be

$$d=|\hat{n}\cdot (\vec{p}-\vec{q})|=\frac{1}{||\vec{w}||}|\vec{w}^T\cdot (\vec{p}-\vec{q})|=\frac{1}{||\vec{w}||}|\vec{w}^T\cdot \vec{p}-\vec{w}^T\cdot\vec{q}|=\frac{1}{||\vec{w}||}|\vec{w}^T\cdot \vec{p}+b|=\frac{1}{||\vec{w}||}(\vec{w}\cdot\vec{p}+b)$$

*(note that in the equation above, the first 4 dot($\cdot$) indicate the scalar product of two vectors and the last one  indicates matrix multiplication.)*





![logistic-regression-decision-boundary-margin](/home/lin/GitRepo/summary/ml-note/logistic-regression-decision-boundary-margin.png)

For the convenience of deriving the algorithm, we define our negative label to be $-1$ and positive label to be $1$. 