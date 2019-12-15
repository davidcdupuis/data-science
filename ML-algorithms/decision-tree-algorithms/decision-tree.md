# Decision Tree

http://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/

<a id="summary"></a>
## Summary

 1. [Definition](#definition)
 2. [Explanation](#explanation)
 3. [Pros](#pros)
 4. [Cons](#cons)
 5. [Implementation](#implementation)

<a id="definition"></a>
## Definition

Decision tree is a type of supervised learning algorithm (having a pre-defined target variable) that is mostly used in classification problems. It works for both categorical and continuous input and output variables. In this technique, we split the population or sample into two or more homogeneous sets (or sub-populations) based on most significant splitter / differentiator in input variables.

<a id="explanation"></a>
## Explanation

### Terms

* **Root node**: It represents entire population or sample and this further gets divided into two or more homogeneous sets.
* **Splitting**: It is a process of dividing a node into two or more sub-nodes.
* **Decision Node**: When a sub-node splits into further sub-nodes, then it is called decision node.
* **Leaf/Terminal Node**: Nodes do not split is called Leaf or Terminal node.
* **Pruning**: When we remove sub-nodes of a decision node, this process is called pruning. You can say opposite process of splitting.
* **Branch/Sub-Tree**: A sub section of entire tree is called branch or sub-tree.
* **Parent and Child Node**: A node, which is divided into sub-nodes is called parent node of sub-nodes where as sub-nodes are the child of parent node.

### Regression vs Classification

### Splitting

#### Gini Index

* It works with categorical target variable “Success” or “Failure”.
* It performs only Binary splits
* Higher the value of Gini higher the homogeneity.
* CART (Classification and Regression Tree) uses Gini method to create binary splits.

##### Steps
1. Calculate Gini for sub-nodes, using formula sum of square of probability for success and failure (p^2+q^2).
2. Calculate Gini for split using weighted Gini score of each node of that split

#### Chi-Square

* It works with categorical target variable “Success” or “Failure”.
* It can perform two or more splits.
* Higher the value of Chi-Square higher the statistical significance of differences between sub-node and Parent node.
* Chi-Square of each node is calculated using formula,
* Chi-square = ((Actual – Expected)^2 / Expected)^1/2
* It generates tree called CHAID (Chi-square Automatic Interaction Detector)

##### Steps
1. Calculate Chi-square for individual node by calculating the deviation for Success and Failure both
2. Calculated Chi-square of Split using Sum of all Chi-square of success and Failure of each node of the split

#### Entropy

##### Steps
1. Calculate entropy of parent node
2. Calculate entropy of each individual node of split and calculate weighted average of all sub-nodes available in split.

#### Reduction in Variance
Used for continuous target variables
[formula]

##### Steps

1. Calculate variance for each node.
2. Calculate variance for each split as weighted average of each node variance.

### Key Parameters

1. #### Minimum samples for a node Split
    * Defines the minimum number of samples (or observations) which are required in a node to be considered for splitting.
    * Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
    * Too high values can lead to under-fitting hence, it should be tuned using CV.

2. #### Minimum samples for a leaf
    * Defines the minimum samples (or observations) required in a terminal node or leaf.
    * Used to control over-fitting similar to min_samples_split.
    * Generally lower values should be chosen for imbalanced class problems because the regions in which the minority class will be in majority will be very small.

3. #### Maximum depth of the tree
    * The maximum depth of a tree.
    * Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
    * Should be tuned using CV.

4. #### Maximum number of terminal nodes
    * The maximum number of terminal nodes or leaves in a tree.
    * Can be defined in place of max_depth. Since binary trees are created, a depth of ‘n’ would produce a maximum of 2^n leaves.

5. #### Maximum features to consider for split
    * The number of features to consider while searching for a best split. These will be randomly selected.
    * As a thumb-rule, square root of the total number of features works great but we should check upto 30-40% of the total number of features.
    * Higher values can lead to over-fitting but depends on case to case.

<a id="pros"></a>
## Pros

* Easy to Understand
* Useful in data exploration
* Less data cleaning required
* Data type is not a constraint
* Non-parametric Method

<a id="cons"></a>
## Cons

* Overfitting
* Not fit for continuous variables

<a id="implementation"></a>
## Implementation
```python
#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn import tree
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create tree object
model = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
# model = tree.DecisionTreeRegressor() for regression
# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)
#Predict Output
predicted= model.predict(x_test)
```
