# Capstone: Forecasting Batting Performance using Machine Learning

## Problem Statement

The world of professional sports is synonymous with competition: whether it is on the field of play, preparing for games, training talent, or finding talent for your team; players, coaches, and executives are always looking for an edge. Baseball is no exception and has drawn the attention of data scientists so much that there is an entire field devoted to baseball data and statistics: Sabermetrics.

With more and more data being collected on the events on the field, critical decisions can be better informed using sabermetric tools. One of these tools is a metric called Wins Above Average (WAA) which takes all of a player's outcomes on the field and normalizes them against the competition for that season, ultimately coming up with a value the player performed better than an average player, and in a metric that is all important: Wins. Players all have career arcs, and free agency and salaries will take into account past performance with other factors to pick which player should be on the team. These decisions make or break championship aspirations.

Armed with WAA and other metrics available from Baseball-Reference.com, utilize machine learning to provide a forecast of a player's next season's WAA per game. Find that edge, get to the World Series!

## Project Directory
```
capstone
|__ code
|   |__ 01_Data_Cleaning.ipynb
|   |__ 02_Feature_Extraction_and_Cleaning.ipynb
|   |__ 03_Full_Approach_Preprocessing.ipynb
|   |__ 04_Full_Approach_Modelling.ipynb
|   |__ 05_RDF_YDF_Extraction.ipynb
|   |__ 06_RDF_YDF_Preprocessing.ipynb
|   |__ 07_RDF_YDF_Modelling.ipynb
|__ data
|   |__ war_daily_bat.txt
|   |__ majors_appearances.csv
|   |__ baseballdatabank-master\baseballdatabank-master\core\Teams.csv
|   |__ batpos_eda.csv
|   |__ batpos_feature_extracted.csv
|   |__ batpos_full_feature_extracted.csv
|   |__ X_test_full_for_baseline.csv
|   |__ batpos_full_preprocessed.csv
|   |__ rdf_ydf.csv
|   |__ rdf_ydf_feature_extracted.csv
|   |__ rdf_ydf_preprocessed.csv
|__ README.md
```

## Executive Summary
Information was obtained from Baseball-Reference.com, either directly via download, or through a data request to the company. From there the data was cleaned to include only batters who played from 1962-2019 and played a significant amount of the season. The goal was to predict Wins Above Average (WAA) per game from past performance, and players were trimmed to keep only players with four consecutive seasons of play allowing for three years of lookback and a target year.

Two approaches were used when preprocessing the data: (1) Create time-series lookbacks for each feature, and (2) Model Player Performance and League Environment separately, then include that with past performance of WAA.

Approach (1) produced model score results of 39% through a Linear Regression model, while (2) produced model score results of 37% through Random Forest Regression. Both were above the baseline score of 29%--calcualted by taking the weighted average of the past three years WAA--indicating that there is a opportunity to forecast batting performance with a competitive edge.

## Data Collection
Data for this project was collected three ways:
-  Baseball-Reference.com hosts a downloads [page](https://www.baseball-reference.com/data/) where data on players' value performance stats can be easily downloaded.
-  A data request was made to an employee at Baseball-Reference for a set of data unavailable for download
-  [Sean Lahman's baseball database](https://www.seanlahman.com/baseball-archive/statistics/) which also provides various statistics and information

From these three sources, I was able to create a database of baseball hitters statistics from the early 1900s to present day. The target variable I was interested in was initially Wins Above Replacement (WAR) as it is the more commonly cited value statistic in baseball circles; however, due to data incompleteness and some gaps in the logic I reverse engineered from Baseball-Reference, I decided to change to a target of Wins Above Average (WAA). This metric is also a measure of batter value and had more data completeness than WAR.

The database that was created had a level-of-detail of player-season-stint. Each row was specific to one player, specific to one season of that player, and specific to the stint on a team that player had during that season. In other words, if a player was traded from the Yankees to the Dodgers during the season, he would have 2 rows for that season whereas a player who was not traded would only have the one row.

## Data Cleaning
Baseball fans, reporters, historians, and statisticians tend to agree that there are different eras for baseball. They have names like "dead-ball era", "live-ball era", "post-war", "steroid era", etc. The lively debate online is not whether they exist but the lines of demarcation between the eras.

<p align="center"><img src=assets/Era_Chart.png  alt="drawing" width="800" /></p>

[image source](https://aluby.domains.swarthmore.edu/sdv/posts/2021-02-26-mlb-home-runs/)

For this analysis, we will examine the league starting in the "expansion era" which is commonly described as starting in 1962 due to the increase in the number of teams in the league. This will have a few benefits:

1.  The number of teams will be at least 20
2.  It excludes years with sparse game information, allowing for better data completeness
3.  The schedule was standardized to 162 games by this point

On the other side, the data will end in 2019 due to a shortened season in 2020 because of the COVID-19 pandemic.

As mentioned during data collection, the dataset does not currently have a level-of-detail of one player's season per row. In order to accomplish this, I needed to aggregate multiple stint seasons. Cumulative metrics were handled with simple addition, while rate metrics were recalculated as a weighted average of the stint rows by the amount of games. Doing this left us with the correct level-of-detail to proceed.

Our target is a cumulative metric, but we want to be able to control for seasons where a player may have played fewer games than others. Just because a player was injured, doesn't mean they weren't performing well without the injury. For this reason, all cumulative stats were turned into rates of per-game. Pre-existing rate stats were unchanged.

The last bit that needed to be done was to create positional variables for the percent of games played at a position within the season. The data from Baseball-Reference was originally in the format of games at each position, so those were recalculated to produce the percentage share of games played at each position.

## Feature Extraction
At this point, our dataset consisted of the WAA_pg (our rate target variable) and the underlying features from the player-season. But, we aren't interested in calculating WAA_pg using known season statistics, we want to be able to forecast WAA_pg *before* the season starts! To accomplish this, the dataset was analyzed to see how much of our data would be culled if we only kept players who had rows for a consistent number of years. A few lookback windows were reviewed:

-  2 Year Lookback: Drops 43% of rows for a dataset of 1,388 unique players
-  3 Year Lookback: Drops 55% of rows for a dataset of 1,159 unique players
-  5 Year Lookback: Drops 73% of rows for a dataset of 776 unique players
-  10 Year Lookback: Drops 94% of rows for a dataset of 248 unique players

A 3-year lookback was selected meaning that only players with four (three lookback plus the target) consecutive seasons were kept. This reduced the dataset to 45% of its original size.

Next, a dataframe was created to include the lookback values of each feature which in effect quadrupled the amount of features. For example: a row now has runs_bat_pg_1yr, _2yr, and _3yr in addition to runs_bat_pg.


## Preprocessing RDF and YDF
A problem I foresaw with trying to model WAA_pg was that it relied on both a player's performance as well as the league environment. Trying to model both simultaneously seemed to be asking too much and could be simplified by trying to model the player's performance and the league environment separately. For example: Run variables are correlated with WAA_pg while the league environment variable opponent runs per game isn't.

<p align="center"><img src=assets/runs.png alt="drawing" width="800" class="center"/></p>
<p align="center"><img src=assets/opprpg.png alt="drawing" width="400" class="center"/></p>

Note how there are bands of values for opprpg. Those values are constant for a season and will span the WAA_pg of every player who played in that season. While uncorrelated, the opprpg is used in the calculation of WAA to translate runs into wins.

With runs and opprpg being critical parrts of the WAA equation, I decided to preprocess them individually in preparation for modelling them individually. The runs and opprpg datasets were both standard scaled, and runs was reduced from 62 features to 20 using Recursive Feature Elimination (RFE). The noise from those 42 eliminated features was then added back in using Principal Component Analysis, and 5 PCA features were added back to the runs dataset for a total of 25 features. These 5 PCA features captured over half of the variance explained.

## Modelling RDF and YDF
Both the runs dataframe (RDF) and the opprpg (year dataframe aka YDF) were modelled using a Linear Regression with and without regularization, Random Forest Regression, and Support Vector Regression. Hyperparameters for RDF were tuned using Grid Search for both Random Forest and SVR. The findings were RDF performed best using SVR with a score of 35% against a standard regression baseline of 0, while YDF performed best with Random Forest with a score of 95% against a standard regression baseline of 0.

The predicted values on the test set were sent over to be used in modelling where they were combined with the rest of the features not included in RDF and YDF. The same regression model types were used and Random Forest Regression performed the best with a score of 37% against the 29% baseline.

## Preprocessing Full Approach
The same process was used with the full dataset that was used for RDF for preprocessing. Standard Scaled, reduced from 68 columns down to 11, and had 6 PCA features added back to capture about 50% of the noise from those 57 removed features.

<p align="center"><img src=assets/pca.png alt="drawing" width="400" class="center"/></p>

## Modelling Full Approach
The same models were used to model the full approach with Linear Regression without regularization taking the top performance at a score of 39%

## Future Developments
So many things can be added to this model to try and capture additional noise. The most surprising thing was the poor performance of the RDF models. It would seem that the previous 3 years of a player's career would be a good indicator for future performance, but it just wasn't the case. Looking deeper into the nuts and bolts of the normalizing that Baseball-Reference does could also help tighten the model--there may be more at play with the league environmental variables beyond just the average amount of runs being scored per game.
