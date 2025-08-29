# Storing notes on model training progress

## Next Steps and Recommendations (from last assignment)
Here are some recommendations for future research:
* More analysis with SVM - my initial run took 10 hours only afterwhich I realized my parameter choices were suboptimal. I suspect more trial and error with poly and other features would yield a more competitive model. From a runtime perspective, for the capstone project I can look at targeted usage of larger EC2 instances.
* Try KNeighborsRegressor
* Try Logistic Regression with polynomials and regularization
* Analyze model accuracy to look at using models on subsets of data rather than the whole dataset
* Consider using some of the data from the previous campaigns
* Use Stochastic Gradient Descent - didn't get a chance to use this
* More visualizations and better libraries for decision tree (ran into issue with anaconda)
* Now that I'm looking at the confusion matrix for decision tree, it's 

## TODO
- [x] Import initial models
- [x] Move to git hub!!
- [ ] SVM - param fitting with kernel=poly
- [ ] Spike : Research KNeighborsRegressor
- [ ] Spike : Logistic Regression with polynomials and regularization
- [ ] Spike : Stochastic Gradient Descent - didn't get a chance to use this
- [ ] Spike : Split out different game scenarios. 4th down, 2min drill, etc ??
- [ ] Spike : Team by team analysis

# Training Log
## 8/16
```
Imported initial models, initial param fitting, results below
```
| Model | Features | Train Time | Train Accuracy | Test Accuracy |
| ----- | -------- | ---------- | -------------  | -----------   |
| Dummy | 0 | 0.0121 secs | 0.599545 | 0.603615 |
| LogisticRegression | 11 | 0.0655 secs | 0.724623 | 0.736600 |
| KNN with 2 neighbors | 11 | 0.2155 secs | 0.778582 | 0.601311 |
| DecisionTreeClassifier with max_depth=5 | 11 | 0.0491 secs | 0.726897 | 0.738283 |
| SVM (default params) | 11 | 2.1202 secs | 0.599545 | 0.603615 |

## 8/29
```
Added framework from class work, seeing a drop in test accuracy. Training percent dropped, tweaked formations, added hyper parameters. 

Next:
 - data is pretty clean, accuracy is topping out at 71%
 - look at binary classification (results are better, exclude kneel downs and spikes - put qb sneak in with runs and change formation to be JUMBO), will get roc curve working again
 - look at pruning nodes from decision tree
 - look into 
```
| Model | Features | Train Time | Train Accuracy | Test Accuracy |
| ----- | -------- | ---------- | -------------  | -----------   |
|  NON-BINARY CLASSIFICATION |
| Dummy | 0 | 0.0149 secs | 0.602564 | 0.602364 |
| LogisticRegression (fit_intercept = False, penalty = l2, C = 10000.0, class_weight = None, solver = liblinear) | 11 | 0.0442 secs | 0.730356 | 0.728659 |
| K-Nearest Neighbors (algorithm = kd_tree, n_neighbors = 10, weights = None) | 11 | 0.0650 secs | 0.655914 | 0.586458 |
| DecisionTreeClassifier with (criterion=log_loss, max_depth=6, max_features=log2) | 11 | 0.0222 secs | 0.729529 | 0.724062 |
| SVM (default params) | 11 | 1.2117 secs | 0.602564 | 0.602364 |
|  |  |  |  |  |
|  BINARY CLASSIFICATION |
| Dummy | 0 | 0.0183 secs | 0.605809 | 0.603790 |
| LogisticRegression (fit_intercept = True, penalty = None, C = 1.0, class_weight = None, solver = lbfgs) | 11 | 0.0358 secs | 0.788382 | 0.734181 |
| K-Nearest Neighbors (algorithm = auto, n_neighbors = 1, weights = None) | 11 | 0.0580 secs | 1.000000 | 0.525719 |
| DecisionTreeClassifier with (criterion=gini, max_depth=1, max_features=None) | 11 | 0.0212 secs | 0.726141 | 0.684002 |
| SVM (default params) | 11 | 0.1329 secs | 0.605809 | 0.603790 |
