# Machine Learning Concepts

## Types

**Regression:** A supervised problem, the outputs are continuous rather than discrete.

**Classification:** Inputs are divided into two or more classes, and the learner must produce a model that assigns unseen inputs to one or more (multi-label classification) of these classes. This is typically tackled in a supervised way.

**Clustering:** A set of inputs is to be divided into groups. Unlike in classification, the groups are not known beforehand, making this typically an unsupervised task.

**Density Estimation:** Finds the distribution of inputs in some space.

**Dimensionality Reduction:** Simplifies inputs by mapping them into a lower-dimensional space.

## Kind

**Parametric:** Step 1: Making an assumption about the functional form or shape of our function (f), i.e. f is linear thus we will select a linear model. Step 2: Selecting a procedure to fit or traing our model. This means estimating the Beta parameters in the linear function. A common approach is the (ordinary) least squares, amongst others.

**Non-parametric:** When we do not make assumptions about the form of our function (f). However, since these methods do not reduce the problem of estimating f to a small number of parameters, a large number of observations is required in order to obtain an accurate estimate for f. An example would be the thin-plate spline model.

## Categories

**Supervised:** The computer is presented with example inputs and their desired outputs, given by a "teacher", and the goal is to learn a general rule that maps inputs to outputs.

**Unsupervised:** No labels are given to the learning algorithm, leaving it on its own to find structure in its input. Unsupervised learning can be a goal in itself (discovering hidden patterns in data) or a means towards an end (feature learning).

**Reinforcement Learning:** A computer program interacts with a dynamic environment in which it must complete a certain goal (such as driving a vehicle or playing a game against an opponent). The program is provided feedback in terms of rewards and punishments as it navigates its problem space.


## Motivation

**Prediction:** When we are interested mainly in the predicted variable as a result of the inputs, but not on the way each of the inputs affects the prediction. In a real estate example, Prediction would answer the question of: Is my house over or under valued? Non-linear models are very good at these sort of predictions, but not great for inference because the models are much less interpretable.

**Inference:** When we are interested in the way each one of the inputs affects the prediction. In a real-estate example, Inference would answer the question of: How much would my house cost if it had a view on the sea? Linear models are more suited for inference because the models themselves are easier to understand than their non-linear counterparts.

## Performance analysis

**Confusion Matrix:**

**Accuracy**

**F1 Score:**

**ROC Curve:**

**Bias-Variance Tradeoff:**

**Goodness of Fit = R²:**

**Mean Squared Error (MSE):** The mean square error (MSE) or mean squared deviation (MSD) of an estimator (of a procedure for estimating an unobserved quantity) measures the average of the squares of the errors or deviations - that is, the difference between the estimator and what is estimated.

**Error Rate:** The proportion of mistakes made if we apply out estimate model function to the training observation in a classification setting.

## Tuning

### Cross-validation

**Methods:**
* Leave-p-out cross-validation
* Leave-one-out cross-validation
* k-fold cross-validation
* Holdout method
* Repeated random sub-sampling validation


### Hyperparameters

**Grid Search:** The traditional way of performing hyperparameter optimization has been grid search, or a parameter sweep, which is simply an exhaustive searching through a manually specified subset of the hyperparameter space of a learning algorithm. A grid search algorithm must be guided by some performance metric, typically measured by cross-validation on the training set or evaluation on a held-out validation set.

**Random Search:**

**Gradient-based Optimization:**

**Early Stopping (Regularization):** Early stopping rules provide guidance as to how many iterations can be run before the learner begins to over-fit, and stop the algorithm then.

**Overfitting:** When a given method yields a small training MSE (or cost), but a large test MSE (or cost), we are said to be overfitting the data. This happens because our statistical learning procedure is trying too hard to find patterns in the data, that might be due to random chance, rather than a property of our function. In other words, the algorithms may be learning the training data too well. If the model overfits, try removing some features, decreasing degrees of freedom, or adding more data.

**Underfitting:** Opposite of overfitting, underfitting occurs when a model cannot capture the underlying trend of the data. It occurs when the model does not fit the data enough. Underfitting occurs if the model shows low variance but high bias (to contrast the opposite, overfitting has high variance and low bias). It is often a result of an excessively simple model.

**Bootstrap:** Test that applies Random Sampling with Replacement of the available data, and assigns measures of accuracy (bias, variance, etc.) to sample estimates.

**Bagging:** An approach to ensemble learning that is based on boostrapping. Shortly, given a training set, we produce multiple different training sets (called bootstrap samples), by sampling with replacement from the original dataset. Then, for each bootstrap sample, we build a model. This results in an ensemble of models, where each model votes with equal weight. Typically, the goal of this procedure is to reduce the variance of the model of interest (e.g. decision trees).
