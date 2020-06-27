# Summary

## w6.14w6.21 Summary

week6.14主要跟着cs229和一些blog理清了[SVM的细节](../ml-note/Machine-Learning-Note-5.md)，搞清楚了很多问题，诸如KKT条件的几何意义，何以本来是w的优化问题最后变成了关于拉格朗日乘子的优化问题，为什么smo算法每次更新2个乘子，为什么引入软间隔，为什么kernek trick是trick，但仍然还是有一些疑惑，比如|wx+b|=y_i(wx+b)，这个等式成立的条件显然是wx+b=0这个超平面已经分隔开positive和negative examples，这么说SMO算法是在优化决策边界，那么何以它会从randomly initialized的w和b开始，找到最佳的w和b呢？

week6.21同样跟着cs229进入了[决策树](../ml-note/Machine-Learning-Note-6.md)的世界，它的基本思想很简单，各种算法最核心的就是partitioning criteria，ID3的information gain，C4.5的information gain rate，CART的gini inpurity，借概率论与数理统计、信息学（仍然有概率论与数理统计的基础）的概念对纯度、混乱度做了量化，这一部分真是头大，不得不提前了解了一部分概率论与数理统计的概念，不过我想对今后的学习会大有帮助。

除此之外，还是跟着cs229重新认识了bias和variance的概念，原来它们是基于同一算法过程在不同数据集（但同一分布，比如从一个大数据集上分出来的train、dev、test集）上的表现，高bias的算法给出的w(可以看作随机变量)的mean会偏离最优解，高variance的算法给出的w的variance很高，很发散，所以即使算法有低bias（以最优解为中心分布），但是由于非常发散，在另一个同分布数据集上，得出的解很可能偏离最优解很远。以此为前提，提出了Norm Regularization，它可以减小variance的原因可以从代数（导数）几何理解（函数很不平滑时，定义域内大部分点的导数会很大，feature scaling之后自变量的范围变化不会很大，所以wx+b中的系数w_i会很大），CS229给出的解释比较抽象，但更一般，涉及误差分解的内容。

还有cross validation和feature selection的内容，这一部分在[ml-note-1](../ml-note/Machine-Learning-Note-1.md).

另外，花了一些时间考察了自己比较感兴趣的ML（主要是DL）在音乐领域的应用，最初是想要找找音色转移（保持节奏、音高不变，将一件乐器的声音变成另一件乐器的声音）的内容，结果发现还有音乐生成、音乐风格（流派）转移这些，但是DL在这个领域上好像没有什么里程碑式地发展，几篇论文都是尝试性质的，但自己还是很想尝试实现一个，就是算力的问题，可能要花点钱了。



