### Predicting Plays within NFL Data Bowl, 2025

David Stange

#### Executive summary

Using data from the first 9 weeks of the NFL season in 2022, this model will predict the outcome of the scrimmage play (run or pass). 

#### Rationale
In football games, there are two general types of plays that an offense will run from scrimmage - runs and passes. Defenses attack runs and passes in different ways, with different resources, formations, and personel. Being able to predict what the opposing team will do provides an enormous tactical advantage.  

#### Research Question
What the outcome of the play will be; a run or a pass. 

#### Data Sources
I'll be using the NFL Big Data Bowl 2025 dataset (https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data). This data set consists of play by play data (this is what I'll be using to predict), game data (helps to better categorize play by play data), and player tracking (frame by frame player movement data). 

The play by play data contains game conditions (down, distance, score), pre-snap information (formation, personel), and the result of the play (type of play, outcome, etc). Only the game conditions and pre-snap information will be used for model training. 

#### Methodology
This is a binary classification problem. To solve this, I will be using scaling, iterative hyperparameter tuning, a number of classification models (logistic regression, K-Nearest Neighbors, Decision Tree, Support Vector Machines) as well as ensemble techniques (random forest and gradient boosting). 

#### Results
The baseline model achieved a baseline accuracy score of 60%. The current best model (using GradientBoostingClassifier with 32 classifiers) is able to achieve an accuracy score of 74.7% (an increase of 24.5%).

#### Next steps
The upper limit, given the training features that I'm currently using, seems to be around 75%. I originally used dummy fields for the offensive team (1 of 32 values) but ran into compute limits. Further exploration should include the offensive team, defensive team and other game conditions (weather, etc). 

#### Outline of project

- [Link to notebook 1]()
- [Link to notebook 2]()
- [Link to notebook 3]()


##### Contact and Further Information