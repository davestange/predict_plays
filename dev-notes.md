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
- [ ] Move to git hub!!
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