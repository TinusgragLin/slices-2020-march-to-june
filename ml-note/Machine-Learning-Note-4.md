# LIN's Machine Learning Note

### K-Nearest Neighbors

#### Introduction

K-Nearest Neighbors algorithm explain itself through its name: this algorithm find the K nearest "neighbors" of one input instance by measuring **a certain kind of distance** between the instance and all the instances in our entire dataset and output some value based on the qualities of its K nearest "neighbors". 

KNN could be used in classification as well as regression.  When used in regression problem, KNN outputs **the mean or median value** of the properties of the instance's k nearest neighbors. When used in classification problem, KNN outputs the majority vote outcome.

Depending on how you define the "distance" between two instance, you can implement your KNN in various way. Most often, we would choose so-called "Euclidean distance" which is define as the square root of the sum of all the squared distance in every *dimension*: 

$$ D_{Euclid}=\sqrt{(d_1^{\prime}-d_1)^2+(d_2^{\prime}-d_2)^2+\cdots+(d_n^{\prime}-d_n)^2} $$

Other popular distance in $n$-dimensional space include: 

- **Manhattan Distance**: the sum of the coordinate difference in each dimension.  

  $$ D_{Manhattan}=|d_1^{\prime}-d_1|+|d_2^{\prime}-d_2|+\cdots+|d_n^{\prime}-d_n| $$

- **Mahalanobis Distance**: see [wikipedia](https://en.wikipedia.org/wiki/Mahalanobis_distance).  

- **Cosine Distance**:  

  $$ D_{similarity}=\cos \theta=\frac{\vec{A}\cdot \vec{B}}{||\vec{A}||\cdot||\vec{B}||} $$

#### Problems within the Algorithm

1. Huge Expanse for Real-time Calculation
   Since we have to real-timely calculate the distance between one input instance and every instance in the entire dataset. The space and time the calculation takes would increase tremendously as we increase the dimension of our attribute vector for a instance or expand our dataset.  
   
   One way to tackle this is to store the information of all the "points" in our $n$-dimension space using a structure called K-dimension Tree.

2. The Number Takes Over Instead of Distance  

   Suppose we have a dataset like this and we now have an input instance marked as "x" and we take its 3 nearest neighbors.  

    ![knn-problems-number-takes-over](knn-problems-number-takes-over.png)

Well, you can easily tell that the instance is more likely to be an circle than a square. But the KNN algorithm thinks otherwise because it has taken the instance's 3 nearest neighbors and found 2 of them are square! So the algorithm would think x as an square even though the instance is much closer to the only circle than to the two squares. 

One way to avoid this is to use Weighted-KNN