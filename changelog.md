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
- [x] Switch over to binary classification; get images working again
- [ ] WHAT IS THE SCORE OF THE GAME!!! Determine home team and add in win-probability
- [ ] Spike : Split out different game scenarios. 4th down, 2min drill, etc ??
              Create obvious running vs passing scenarios
- [ ] Spike : Team by team analysis - add hot encoding? 
              Also, research team by team accuracy to see which teams had highest accuracy
- [ ] SVM - param fitting with kernel=poly
- [ ] Spike : Research KNeighborsRegressor
- [ ] Spike : Logistic Regression with polynomials and regularization
- [ ] Spike : Stochastic Gradient Descent - didn't get a chance to use this

# Training Log
## 10/05
*Best Run is 0.743689 from LogisticRegression (fit_intercept = True, penalty = l2, C = 1.0, class_weight = None, solver = newton-cholesky)*

| Model | Features | Train Time | Train Accuracy | Test Accuracy |
| ----- | -------- | ---------- | -------------  | -----------   |
| Dummy | 0 | 0.0087 secs | 0.603805 | 0.603823 |
| LogisticRegression (fit_intercept = True, penalty = l2, C = 1.0, class_weight = None, solver = newton-cholesky) | 29 | 0.0142 secs | 0.741935 | 0.743689 |
| K-Nearest Neighbors (algorithm = auto, n_neighbors = 10, weights = None) | 29 | 0.1761 secs | 0.740281 | 0.740479 |
| DecisionTreeClassifier with (criterion=entropy, max_depth=6, max_features=None) | 29 | 0.0123 secs | 0.746071 | 0.742959 |
| Dummy | 0 | 0.0153 secs | 0.603805 | 0.603823 |
| LogisticRegression (fit_intercept = True, penalty = l2, C = 1.0, class_weight = None, solver = newton-cholesky) | 29 | 0.0163 secs | 0.741935 | 0.743689 |
| K-Nearest Neighbors (algorithm = auto, n_neighbors = 10, weights = None) | 29 | 0.1756 secs | 0.740281 | 0.740479 |
| DecisionTreeClassifier with (criterion=entropy, max_depth=6, max_features=None) | 29 | 0.0123 secs | 0.746071 | 0.742959 |
| SVM (default params) | 29 | 0.0296 secs | 0.767577 | 0.730045 |
| RandomForestClassifier with (n_estimators=3, max_features=6) | 29 | 0.0371 secs | 0.937138 | 0.687436 |
| GradientBoostingClassifier with (n_estimators=32) | 29 | 0.0924 secs | 0.753929 | 0.743397 |

## 09/13
```added random forests and gradient boost classifier```

*Best Run is 0.746680 from GradientBoostingClassifier with (n_estimators=32)*

| Model | Features | Train Time | Train Accuracy | Test Accuracy |
| ----- | -------- | ---------- | -------------  | -----------   |
| Dummy | 0 | 0.0218 secs | 0.603805 | 0.603823 |
| LogisticRegression (fit_intercept = False, penalty = l2, C = 1.0, class_weight = balanced, solver = liblinear) | 26 | 0.0187 secs | 0.734078 | 0.737852 |
| K-Nearest Neighbors (algorithm = auto, n_neighbors = 10, weights = None) | 26 | 0.0979 secs | 0.739454 | 0.741500 |
| DecisionTreeClassifier with (criterion=entropy, max_depth=3, max_features=None) | 26 | 0.0142 secs | 0.739454 | 0.741500 |
| SVM (default params) | 26 | 0.0143 secs | 0.739454 | 0.741500 |
| Ridge with StandardScaler | 26 | 0.0230 secs | 0.936725 | 0.676784 |
| RandomForestClassifier with (n_estimators=3, max_features=7) | 26 | 0.0209 secs | 0.937965 | 0.690646 |
| GradientBoostingClassifier with (n_estimators=32) | 26 | 0.0742 secs | 0.752275 | 0.746680 |

## 9/09
```added sequential feature selection```

*Best Run is 0.741500 from K-Nearest Neighbors (algorithm = auto, n_neighbors = 10, weights = None)*


| Model | Features | Train Time | Train Accuracy | Test Accuracy |
| ----- | -------- | ---------- | -------------  | -----------   |
| Dummy | 0 | 0.0240 secs | 0.603805 | 0.603823 |
| LogisticRegression (fit_intercept = False, penalty = l2, C = 1.0, class_weight = balanced, solver = liblinear) | 26 | 0.0198 secs | 0.734078 | 0.737852 |
| K-Nearest Neighbors (algorithm = auto, n_neighbors = 10, weights = None) | 26 | 0.0964 secs | 0.739454 | 0.741500 |
| DecisionTreeClassifier with (criterion=entropy, max_depth=3, max_features=None) | 26 | 0.0146 secs | 0.739454 | 0.741500 |
| SVM (default params) | 26 | 0.0144 secs | 0.739454 | 0.741500 |

## 8/30
```Using 15/85 split for both operations. Adding score or win percentage yields the same EXACT result. ```

TODO
[ ] Git rid of noise - drop formation, is_overtime and just use secs_remaining
[ ] Add in something akin to "pass liklihood", based on win percentage and time remaining
[ ] Consider using sequential feature selection

*Best Run is 0.741354 from DecisionTreeClassifier with (criterion=entropy, max_depth=3, max_features=None)*

| Model | Features | Train Time | Train Accuracy | Test Accuracy |
| ----- | -------- | ---------- | -------------  | -----------   |
| Dummy | 0 | 0.0083 secs | 0.603805 | 0.603823 |
| LogisticRegression (fit_intercept = False, penalty = l2, C = 1.0, class_weight = balanced, solver = liblinear) | 14 | 0.0258 secs | 0.748966 | 0.737633 |
| K-Nearest Neighbors (algorithm = auto, n_neighbors = 10, weights = None) | 14 | 0.6249 secs | 0.683623 | 0.605720 |
| DecisionTreeClassifier with (criterion=entropy, max_depth=3, max_features=None) | 14 | 0.0233 secs | 0.741108 | 0.741354 |
| SVM (default params) | 14 | 0.5421 secs | 0.603805 | 0.603823 |

```Adding in score and win probability and it dropped! :/ ```

*Best Run is 0.733037 from LogisticRegression (fit_intercept = True, penalty = None, C = 1.0, class_weight = None, solver = lbfgs)*

| Model | Features | Train Time | Train Accuracy | Test Accuracy |
| ----- | -------- | ---------- | -------------  | -----------   |
| Dummy | 0 | 0.0079 secs | 0.603805 | 0.603823 |
| LogisticRegression (fit_intercept = True, penalty = None, C = 1.0, class_weight = None, solver = lbfgs) | 14 | 0.0366 secs | 0.726220 | 0.733037 |
| K-Nearest Neighbors (algorithm = auto, n_neighbors = 1, weights = None) | 14 | 0.4931 secs | 1.000000 | 0.569896 |
| DecisionTreeClassifier with (criterion=gini, max_depth=1, max_features=None) | 14 | 0.0223 secs | 0.660877 | 0.664308 |
| SVM (default params) | 14 | 0.5445 secs | 0.603805 | 0.603823 |

```Switching over to full binary classification; improving automation. Training with 1.5% of data.```

*Best Run is 0.736466 from LogisticRegression (fit_intercept = True, penalty = None, C = 1.0, class_weight = None, solver = lbfgs)*

| Model | Features | Train Time | Train Accuracy | Test Accuracy |
| ----- | -------- | ---------- | -------------  | -----------   |
| Dummy | 0 | 0.0081 secs | 0.603805 | 0.603823 |
| LogisticRegression (fit_intercept = True, penalty = None, C = 1.0, class_weight = None, solver = lbfgs) | 11 | 0.0352 secs | 0.732423 | 0.736466 |
| K-Nearest Neighbors (algorithm = auto, n_neighbors = 1, weights = None) | 11 | 0.0590 secs | 0.990074 | 0.559463 |
| DecisionTreeClassifier with (criterion=gini, max_depth=1, max_features=None) | 11 | 0.0230 secs | 0.660877 | 0.664308 |
| SVM (default params) | 11 | 1.1560 secs | 0.603805 | 0.603823 |


## 8/29
```
Using non-binary classification.
Added framework from class work, seeing a drop in test accuracy. Training percent dropped, tweaked formations, added hyper parameters. Training with 1.5% of data.

Next:
 - data is pretty clean, accuracy is topping out at 71%
 - look at binary classification (results are better, exclude kneel downs and spikes - put qb sneak in with runs and change formation to be JUMBO), will get roc curve working again
 - look at pruning nodes from decision tree
 - look into 
```
*Best Run is 0.733183 from DecisionTreeClassifier with (criterion=log_loss, max_depth=3, max_features=log2)*

| Model | Features | Train Time | Train Accuracy | Test Accuracy |
| ----- | -------- | ---------- | -------------  | -----------   |
| Dummy | 0 | 0.0149 secs | 0.602564 | 0.602364 |
| LogisticRegression (fit_intercept = False, penalty = l2, C = 10000.0, class_weight = None, solver = liblinear) | 11 | 0.0442 secs | 0.730356 | 0.728659 |
| K-Nearest Neighbors (algorithm = kd_tree, n_neighbors = 10, weights = None) | 11 | 0.0650 secs | 0.655914 | 0.586458 |
| DecisionTreeClassifier with (criterion=log_loss, max_depth=3, max_features=log2) | 11 | 0.0353 secs | 0.732423 | 0.733183 |
| SVM (default params) | 11 | 1.2117 secs | 0.602564 | 0.602364 |


## 8/16
```
Using binary classification. Imported initial models, initial param fitting, results below. Training with 30% of data.
```

*Best Run is 0.738283 from DecisionTreeClassifier with max_depth=5*

| Model | Features | Train Time | Train Accuracy | Test Accuracy |
| ----- | -------- | ---------- | -------------  | -----------   |
| Dummy | 0 | 0.0121 secs | 0.599545 | 0.603615 |
| LogisticRegression | 11 | 0.0655 secs | 0.724623 | 0.736600 |
| KNN with 2 neighbors | 11 | 0.2155 secs | 0.778582 | 0.601311 |
| DecisionTreeClassifier with max_depth=5 | 11 | 0.0491 secs | 0.726897 | 0.738283 |
| SVM (default params) | 11 | 2.1202 secs | 0.599545 | 0.603615 |
