# Capstone Project
# Predicting Plays within NFL Data Bowl 2025, Final Report

David Stange
![AWS NFL Big Data Bowl](resources/aws_nfl_big_data_bowl.jpg "AWS NFL Big Data Bowl")


# Executive Summary
American football in the 21st century, is an era defined by prolific offensive play callers. These coaches (both head and assistant) have been able to use data, defensive tendencies, and advanced game play theory to gain advantages over their defensive counterparts. They use position groupings, down and distance, and pre-snap motion to force defenses to tip their hand for what they have planned on each play. Likewise, defensive coaches, are looking for any advantage they can gain to determine what type of play the offense has called - run or pass. 

## Problem Statement
This project attempts to predict, using game and pre-snap information, whether the offense will run or pass the ball on a given play. This is the primary objective for any defensive playcaller and knowing this information will provide valuable information when determining what groupings and formation of players the defense should be in. As each non-special teams play has only two outcomes, this naturally lends itself to binary classification. As runs generally involve less risks than passes and lead to less game clock stoppages, they are the natural choice for any team that wants to end the half or game as quickly and efficiently as possible. Likewise, as passes provide a higher probability of scoring and provide more opportunities for game clock stoppage, they are the preferred choice for teams that are losing and need to score as quickly as possible. 

## Challenges
The challenges with attempting to predict in-game behavior are:
* player differences - each team has natural strengths and weaknesses depending on the players on the team, overall roster construction, and player availability due to injury. Some teams feature players that are more skilled at runs (offensive line and running backs) while others may have players more suited to passes (quarterback and wide receivers). Each player is not created equal - replacing a dominant player with a lesser skilled player due to injury or suspension will alter what plays and formations the team will call.
* coaching differences - each coaching staff also has strengths and weaknesses depending on the systems of plays that they run, how well they mesh with the players on the team, and how effective they are at calling those plays. A coach that prefers to run the ball will call more plays than average, depending on game conditions. Also, a less skilled coaching staff might be more predictable with their play calls than a more skilled or data aware coaching staff.
* chess matches - as defenses have gotten better at scouting offensive tendencies, offenses have worked harder to go against those tendencies. Using personnel that favor runs and passes equally and calling the opposite type of a play out of a formation (i.e. running out of a passing formation and vice versa) are two ways that offenses can counter defensive research. Also, calls later in the game are totally dependent on every play that happened previously. Throwing a pass in a situation where a team had previously run the ball are all attempts at keeping the defense in the dark as to their tendencies.
* offensive and defensive matchups - the players and strengths of the defense will factor into what types of plays the offense will call. When facing a defense that is weaker against the run, for instance, offenses will tend to run more running plays. 
* game conditions - field makeup (grass versus turf) and game time weather conditions also factor in. Grass is naturally slower than turf fields and naturally degrades during the game and the season. A fast team playing on fast conditions might prefer to pass more than run to take advantage of their speed differences. Game time weather conditions may include extreme heat, cold, rain, fog, or snow which may force offenses to favor runs over passes. In 2008 a game with extreme winds caused one team to run 86% of the time due to the inherit difficulty in throwing accurately. 

Each of the above challenges were not accounted for in the dataset used in this project. 

## Summary of American Football and the National Football League
>American football, referred to simply as football in the United States and Canada and also known as gridiron football, is a team sport played by two teams of eleven players on a rectangular field with goalposts at each end. The offense, the team with possession of the oval-shaped football, attempts to advance down the field by running with the ball or throwing it, while the defense, the team without possession of the ball, aims to stop the offense's advance and to take control of the ball for themselves. The offense must advance the ball at least ten yards in four downs or plays; if they fail, they turn over the football to the defense, but if they succeed, they are given a new set of four downs to continue the drive. Points are scored primarily by advancing the ball into the opposing team's end zone for a touchdown or kicking the ball through the opponent's goalposts for a field goal. The team with the most points at the end of the game wins. In the case of a tie after four quarters, the game enters overtime. 

>A football game is played between two teams of 11 players each. Playing with more on the field is punishable by a penalty. Teams may substitute any number of their players between downs. The role of the offensive unit is to advance the football down the field with the ultimate goal of scoring a touchdown. The offensive team must line up in a legal formation before they can snap the ball - the formations are named and consist of groupings of players by position, which are groupings of players based on their ability to run, catch, tackle, throw, and move other players. 
per [Wikipedia](https://en.wikipedia.org/wiki/American_football)

The National Football League consists of 32 teams based in American cities; during the season each team will play at most one game against another team with one team playing in their home stadium and the other team designated as away. While there is a rotating bye week throughout the season, most weeks during the season consist of 16 games. 

# DataSet Details
The data comes from the National Football League and their cloud partner, AWS. The data comes from the [NFL Big Data Bowl 2025](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025) dataset for every NFL game during the first nine weeks of the 2020 NFL season. The following datasets are available:

## Game Data
This data provides summary information for each game in the dataset. It includes the following fields
| Field | Type | Description | 
| ----- | ---- | ----------- | 
| gameId | numeric | Game identifier, unique | 
| season | numeric | Season of game | 
| week | numeric | Week of game | 
| homeTeamAbbr | text | Home team three-letter code | 
| visitorTeamAbbr | text | Visiting team three-letter code | 

## Play Data
This data provides summary information on each play, for every game in the dataset. It includes 50 features, for brevity, I'm only including the fields that were used during analysis.
| Field | Type | Description | 
| ----- | ---- | ----------- | 
| gameId | numeric | Game identifier, unique |
| playId | numeric | Play identifier, not unique across games |
| quarter | numeric | Game quarter |
| down | numeric | Down |
| yardsToGo | numeric | Distance needed for a first down |
| possessionTeam | text | Team abbr of team on offense with possession of ball |
| gameClock | text | Time on clock of play (MM:SS) |
| preSnapHomeScore | numeric | Home score prior to the play |
| preSnapVisitorScore | numeric | Visiting team score prior to the play |
| preSnapHomeTeamWinProbability | numeric | The win probability of the home team before the play |
| preSnapVisitorTeamWinProbability | numeric | The win probability of the visiting team before the play | 
| offenseFormation | text | Formation used by possession team |
| receiverAlignment | text | Enumerated as 0x0, 1x0, 1x1, 2x0, 2x1, 2x2, 3x0, 3x1, 3x2 |
| qbSpike | boolean | Boolean indicating whether the play was a QB Spike |
| qbKneel | numeric | Whether or not the play was a QB Kneel |
| qbSneak | numeric| Whether or not the play was a QB Sneak |
| isDropback | boolean | Boolean indicating whether the QB dropped back, meaning the play resulted in a pass, sack, or scramble |

## Understanding the Data
The data provided can be grouped into three categories:
* game conditions - this includes when and where the game is played.
* play setting - what information can be gleaned immediately from the end of the previous play. This includes how much time is remaining in the quarter/game, where the ball is on the field, what down and how many yards to go, and the score of the game. This also includes advanced metrics such as the win probability for both offense and defense - this metric is based on how often other teams have fared, historically, when faced with similar conditions. Both teams start with a 50% win percentage while a team with a large lead towards the end of the game will have a win percentage approaching 1. 
* play formation - what information can be gleaned from the offense prior to the snap. This includes what formation is the offense in and what the player groupings are. Offenses must have _at least_ five linemen and the remaining six players are a mixture of a quarterback (QB), running backs (RB), wide receivers (WR), tight ends (TE), and lineman (OL). Additional TEs and OL typically indicates a higher likelihood of a run play while additional WRs will typically indicate the opposite. Offensive formations include `EMPTY` (no players in the backfield), `SHOTGUN` (QB is aligned 5 yards behind the line), `PISTOL` (QB is aligned 2.5 yards behind the line), `SINGLEBACK` (one player in the backfield), `I_FORM` (two players in the backfield), `JUMBO` (additional OL), `WILDCAT` (player other than QB receiving the snap). Receiver alignment indicates how many of each skill position (RB, WR, and TE) are in each formation. 

## Running the Notebook
Prior to running the notebook, the following dataset will need to be downloaded from Kaggle: [Big Data Bowl 2025 dataset](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025/data) and extracted to the following folder structure
```
├─data
│ └┬──big_data_bowl_2025
│  └──────<data-files>
└─predict_plays
  └───predict.ipynb
```

The notebook has two modes, which can be set in the topmost section (Hyperparameters):
* Hyperparameter Tuning - this allows you to tune parameters. Set `IS_TRAINING_MODE = False`. The final cell (in Summary) prints out the parameters determined during this run. 
* Model Training - this allows you to run models based on tuning parameters. Set `IS_TRAINING_MODE = True`. This will print out summary information which can be copied and pasted into [the change log](changelog.md).  

# Exploratory Data Analysis

## Data Preparation

To ensure that the data was free of missing values, I queried each of the features for invalid numeric, boolean, or textual fields. For any fields which had NaN or unknown values, I interrogated those fields for any data patterns which I could overcome. While most of the data did not require filtering, there were some changes that needed to be made:
- *Formations*: Formation data was consistent, with the exception of QB spikes, kneels, and sneeks. In these plays, the formation is of secondary concern (for instance, any valid formation for a QB spike is ok). In cases where the formation could not be determined `UNKNOWN` was substituted. For `JUMBO` formations, the number of skill lineman could not be determined so I simply set a boolean to indicate this (`extra_ol`). 

The following features needed to be created:
- Position Features - These indicate the number of each position based on the `offenseFormation` and `receiverAlignment`, including `wr_count`, `te_count`, `rb_count`, and `extra_ol`.
- Time Features - This indicate the number of seconds remaining in the game (`secs_remaining`), is it inside of the last two minutes of the half or overtime (`is_inside_two_mins`), and whether the game is in overtime (`is_overtime`).
- Play Result Feature - This expanded on `isDropback` which didn't account for QB sneaks, kneels, and spikes. The feature `play_result` was added with these accounting as runs, runs, and passes respectively. 
- Offensive Features - The original dataset referenced everything as home or away team; getting at the score differential (`score_offset`) and win probability (`win_probability`) required joining against the games data file.

The following encodings were needed:
- *Formation* - one hot encoding was used for each formation.
- *Quarter* - while the quarter is numerical, it needed to be treated categoricall, not a numerical one (the fourth quarter is not double the second quarter). One hot encoding was used here as well as a boolean to indicate overtime. 
- *Down* - similar to quarter, down needed to be treated as a categorical feature and one-hot encoding was used. 
- *Offensive Team* - one hot encoding was used here as well, however, this did not prove to be a useful field and was discarded to keep model training performance levels acceptable. 

## Observations

When I started this project, there were several expectations that I expected the data to easily support. 

## Expectation #1: Teams pass substantially more in the last two minutes of each half
These plots show how the odds of a pass is affected by the last two minutes of each half.

*What I expected to see* - That the rate of passes will spike in the last three minutes of each half consistently across both halves.

*What I actually observed* - The rate of passes spikes in the 29th (72.6%) and 30th minute (84.2%). The variance in the data in the second half was higher but the percentage of passes in the 59th and 60th minutes weren't substantially elevated (56.5% and 64.5%). I would attribute this to teams evenly trying to get one last score before the end of the first half; in the second half the focus is more on winning the game (rather than just scoring points) so the choices here are more inline with the overall average.

<img src="resources/pass_percent_by_min_first_half.png" alt="drawing" width="500"/><br/>
<img src="resources/pass_percent_by_min_second_half.png" alt="drawing" width="500"/><br/>

## Expectation #2: Teams pass more based on how much they are winning or losing
These plots show how the odds of a pass or a run is affected by the score offset and the quarter. 

*What I expected to see* - The more that a team is losing by will have a direct correlation to how much the pass; similarly with teams holding a lead and running the ball.  

*What I actually observed* - This becomes more and more pronounced as the game goes on. In the first quarter, when teams were losing by 2 scores (14+ points) they ran at a much _higher_ rate than the overall average (60%). Over time, however, the trend line gradually decreased. 

<img src="resources/score_vs_result_Q1.png" alt="drawing" width="500"/><br/>
<img src="resources/score_vs_result_Q2.png" alt="drawing" width="500"/><br/>
<img src="resources/score_vs_result_Q3.png" alt="drawing" width="500"/><br/>
<img src="resources/score_vs_result_Q4.png" alt="drawing" width="500"/>

## Measuring Accuracy
For baseline I chose `DummyClassifier`, which uses the mean for the dataset (60.38%), and chose `accuracy` to guage to effectiveness of my models. 

With a false positive, defenses will be expecting a pass when it's actually a run. This means that they will generally have smaller, faster players on the field (defensive backs and coverage linebackers). This will give the offense a greater chance for a successful play (one that nets them a first down) without substantially increasing the risk of an explosive play (one that is 20+ yards). 

With a false negative, defenses will be expecting a run when it's actually a pass. This means that they will generally have larger, slower players on the field (more defensive lineman and linebackers in place of defensive backs). This will give the offense a greater chance for a successful play AND increase the risk of an explosive play. 

For defensive coaches, who are generally risk adverse, false negatives come at a much higher risk than false positives. Letting your opponent slowly move the ball down the field is _greatly preferred_ to letting them get explosive plays and long scoring plays.

## Training Data
For training data, I choose a 15/85 split (15% training data, 85% test data). The data was shuffled and stratified based on the training features with a constant random state chosen. 
 
# Outcome

## Overall Goal
The goal of this project was to predict the outcome of an individual play. Seven different supervised models were evaluated against their accuracy. The mean of the dataset is 60% passes, the initial expectation was that our model could improve upon the baseline by at least 120% (70% accuracy being our goal). 

## Findings
The algorithm with the best accuracy LogisticRegression (0.7437) followed by GradientBoostingClassifier, DecisionTreeClassifier, and K-Nearest Neighbors all being differentiated by no more than 1%. The algorithm with the best precision is GradientBoostingClassifier (0.6923) with LogisticRegression, K-Nearest Neighbors, and DecisionTreeClassifier coming in within 1.2%. 

While we were able to achieve a 23% increase over our baseline of 60.38%, we'll need a higher accuracy than 74% before it will be useful.  

| Model | Features | Train Time | Train Accuracy | Test Accuracy | Precision |
| ----- | -------- | ---------- | -------------  | -----------   | -----------   |
| LogisticRegression (fit_intercept = True, penalty = l2, C = 1.0, class_weight = None, solver = newton-cholesky) | 29 | 0.0163 secs | 0.741935 | 0.743689 | 0.6870 |
| K-Nearest Neighbors (algorithm = auto, n_neighbors = 10, weights = None) | 29 | 0.1756 secs | 0.740281 | 0.740479 | 0.6853 |
| DecisionTreeClassifier with (criterion=entropy, max_depth=6, max_features=None) | 29 | 0.0123 secs | 0.746071 | 0.742959 | 0.6843 |
| SVM (default params) | 29 | 0.0296 secs | 0.767577 | 0.730045 | 0.6523 |
| RandomForestClassifier with (n_estimators=3, max_features=6) | 29 | 0.0371 secs | 0.937138 | 0.687436 | 0.5997 |
| GradientBoostingClassifier with (n_estimators=32) | 29 | 0.0924 secs | 0.753929 | 0.743397 | 0.6923 |


| Model | EMPTY | I_FORM | JUMBO | MUDDLE | PISTOL | SHOTGUN | SINGLEBACK | VICTORY | WILDCAT |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| lr | 0.9665 | 0.7138 | 0.8426 | 1.0000 | 0.6552 | 0.7464 | 0.6658 | 1.0000 | 0.8378 | 
| kn | 0.9665 | 0.7149 | 0.8426 | 0.0476 | 0.6438 | 0.7456 | 0.6616 | 1.0000 | 0.8514 | 
| dt | 0.9656 | 0.7103 | 0.8426 | 1.0000 | 0.6610 | 0.7468 | 0.6655 | 1.0000 | 0.6892 | 
| sv | 0.9550 | 0.7034 | 0.8148 | 0.4762 | 0.6800 | 0.7294 | 0.6553 | 0.9728 | 0.8649 | 
| rf | 0.9259 | 0.6431 | 0.6667 | 0.7143 | 0.6190 | 0.7053 | 0.5918 | 1.0000 | 0.6622 | 
| gb | 0.9665 | 0.7080 | 0.7963 | 0.0000 | 0.7029 | 0.7509 | 0.6586 | 1.0000 | 0.7297 | 

## Logistic Regression
With hyperparameters, fit_intercept = True, penalty = l2, C = 1.0, class_weight = None, solver = newton-cholesky, it achieved an accuracy of 0.743689 and a precision of 0.6870 in 0.0142 secs.

<img src="resources/model_logistic_regression.png" width="600"/>

## K-Nearest Neighbors
With hyperparameters, algorithm = auto, n_neighbors = 10, weights = None, it achieved an accuracy of 0.740479 and a precision of 0.6853 in 0.1756 secs.

<img src="resources/model_k_neighbors.png" width="600"/>

## Decision Tree
With hyperparameters, criterion=entropy, max_depth=6, max_features=None, it achieved an accuracy of 0.742959 and a precision of 0.6843 in 0.0123 secs.

<img src="resources/model_decision_tree.png" width="600"/>

## Support Vector Machines
With default parameters, it achieved an accuracy of 0.730045 and a precision of 0.6523 in 0.0296 secs.

<img src="resources/model_support_vector.png" width="600"/>

## Random Forest
With hyperparameters, n_estimators=3, max_features=6, it achieved an accuracy of 0.687436 and a precision of 0.5997 in 0.0371 secs.

<img src="resources/model_random_forest.png" width="600"/>

## Gradient Boosting Ensemble
With hyperparameters, n_estimators=32, it achieved an accuracy of 0.743397 and a precision of 0.6923 in 0.0924 secs.

<img src="resources/model_gradient_boosting.png" width="600"/>

## Extreme Gradient Boosting





 - Identify the type of learning (classification or regression) and specify the expected output of your selected model.
 - Determine whether supervised or unsupervised learning algorithms will be used.
 - What types of models did
you consider for your problem (classification, regression, unsupervised)? 
Articulate the evaluation metrics you used and how you determined which model
was most optimal for your problem.
 - list which model did the best and how well it performed
 - list each model, along with ROC curve, confusion matrix, and how it performed in each category
 - can we come up with an approach that is the 

# notebook cleanup
 - try XgBoost
 - cleanup ALL ERRORS
 - disable ALL WARNINGS
 - do final runthrough

# Next Steps areas for future analysis
 - what areas would I consider in the future
 - what are different datasets that I would incorporate
 - how would I handle team differences

# Exploratory Data Analysis
Talk about
Initial EDA has been done over the past several months. During this time, I've cleaned the data (removing plays which can't realistically be attributed to run vs pass classification), added a number of additional columns (score offset, dummies, etc), and joined data from other sources (which team is home versus away). I've also done some initial analysis - this plot shows how the odds of pass or run is affected by the quarter and the score offset. 
![score_vs_result_by_quarter](resources/score_vs_result_by_quarter.png "score_vs_result_by_quarter")

## Methodology
This is a binary classification problem. To solve this, I will be using scaling, iterative hyperparameter tuning, a number of classification models (logistic regression, K-Nearest Neighbors, Decision Tree, Support Vector Machines) as well as ensemble techniques (random forest and gradient boosting). 

## Results
The baseline model achieved a baseline accuracy score of 60%. The current best model (using GradientBoostingClassifier with 32 classifiers) is able to achieve an accuracy score of 74.7% (an increase of 24.5%).

Training results can be seen in the [Change Log](changelog.md)

#### Next steps
The upper limit, given the training features that I'm currently using, seems to be around 75%. I originally used dummy fields for the offensive team (1 of 32 values) but ran into compute limits. Further exploration should include the offensive team, defensive team and other game conditions (weather, etc). 

#### Outline of project

The following notebooks are in use
- [predict.ipynb](predict.ipynb)


##### Contact and Further Information

# Model training changes
⚠️ ExiBoost - use it
⚠️ Kneeldowns - Consider victory formation
⚠️ For spikes - use the previous formation?
⚠️ Create "behind the sticks" and "short yardage" features
⚠️ decimal points
