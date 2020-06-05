# LIN's Machine Learning Note

### K-Nearest Neighbors

K-Nearest Neighbors algorithm explain itself through its name: this algorithm find the K nearest "neighbors" of one input instance by measuring **a certain kind of distance** between the instance and all the instances in our entire dataset and output some value based on the qualities of its K nearest "neighbors". 

Depending on how you define the "distance" between two instance, you can implement your KNN in various way. Most often, we would choose so-called "Euclidean distance" which is define as the square root of the sum of all the squared distance in every *dimension*: 

$$ D_{Euclid}=\sqrt{(d_1^{\prime}-d_1)^2+(d_2^{\prime}-d_2)^2+\cdots+(d_n^{\prime}-d_n)^2} $$

Other popular distance in $n$-dimensional space include: 



KNN could be used in classification as well as regression.  