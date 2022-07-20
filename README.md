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
|   |__ Teams.csv
|   |__ batpos_eda.csv
|   |__ batpos_feature_extracted.csv
|   |__ batpos_full_feature_extracted.csv
|   |__ X_test_full_for_baseline.csv
|   |__ batpos_full_preprocessed.csv
|   |__ rdf_ydf.csv
|   |__ rdf_ydf_feature_extracted.csv
|   |__ rdf_ydf_preprocessed.csv
|__ Data_Dictionary.ipynb
|__ requirements.txt
|__ README.md
```

## Executive Summary
Information was obtained from Baseball-Reference.com, either directly via download, or through a data request to the company. From there the data was cleaned to include only batters who played from 1962-2019 and played a significant amount of the season. The goal was to predict Wins Above Average (WAA) per game from past performance, and players were trimmed to keep only players with four consecutive seasons of play allowing for three years of lookback and a target year.

Two approaches were used when preprocessing the data: (1) Create time-series lookbacks for each feature, and (2) Model Player Performance and League Environment separately, then include that with past performance of WAA.

Approach (1) produced model score results of 39% through a Linear Regression model, while (2) produced model score results of 37% through Random Forest Regression. Both were above the baseline score of 29%-31%--calculated by taking the weighted average of the past three years WAA per game--indicating that there is an opportunity to forecast batting performance with a competitive edge.

## Data Dictionary
Data Dictionary for this project is available in the repository at Data_Dictionary.ipynb

## Data Collection
Data for this project was collected three ways:
-  Baseball-Reference.com hosts a downloads [page](https://www.baseball-reference.com/data/) where data on players' value performance stats can be easily downloaded.
-  A data request was made to an employee at Baseball-Reference for a set of data unavailable for download
-  [Sean Lahman's baseball database](https://www.seanlahman.com/baseball-archive/statistics/) which also provides various statistics and information

From these three sources, I was able to create a database of baseball hitters statistics from the early 1871 to mid-2022. The target variable I was interested in was initially Wins Above Replacement (WAR) as it is the more commonly cited value statistic in baseball circles; however, due to data incompleteness and some gaps in the logic I reverse engineered from Baseball-Reference, I decided to change to a target of Wins Above Average (WAA). This metric is also a measure of batter value and had more data completeness than WAR.

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

Our target is a cumulative metric, but we want to be able to control for seasons where a player may have played fewer games than others. For this reason, all cumulative stats were turned into rates of per-game. Pre-existing rate stats were unchanged. To control for players that may have had 10 great games in a season and then got injured, or were sent down to the minors, etc.; a minimum PA requirement of 3.1 PA per team game played was instituted. This requirement is inline with what Major League Baseball uses when determining the league leaders for rate statistics like batting average.

The last bit that needed to be done was to create positional variables for the percent of games played at a position within the season. The data from Baseball-Reference was originally in the format of games at each position, so those were recalculated to produce the percentage share of games played at each position.

## Feature Extraction
At this point, our dataset consisted of the WAA_pg (our rate target variable) and the underlying features from the player-season. But, we aren't interested in calculating WAA_pg using known season statistics, we want to be able to forecast WAA_pg *before* the season starts! To accomplish this, the dataset was analyzed to see how much of our data would be culled if we only kept players who had rows for a consistent number of years. A few lookback windows were reviewed:

-  2 Year Lookback: Drops 53% of rows for a dataset of 832 unique players
-  3 Year Lookback: Drops 66% of rows for a dataset of 639 unique players
-  5 Year Lookback: Drops 82% of rows for a dataset of 376 unique players
-  10 Year Lookback: Drops 97% of rows for a dataset of 78 unique players

A 3-year lookback was selected meaning that only players with four (three lookback plus the target) consecutive seasons were kept. This reduced the dataset to 34% of its original size.

Next, a dataframe was created to include the lookback values of each feature which in effect quadrupled the amount of features. For example: a row now has runs_bat_pg_1yr, _2yr, and _3yr in addition to runs_bat_pg.

After this step, two approaches were performed for preprocessing and modeling:
1.  Full Approach: doing one model against target waa_pg
2.  RDF_YDF Approach: doing one model for runs against target total_runs_pg, one model for target opprpg, and combining those findings with remanining features to model waa_pg

## Preprocessing Full Approach
During preprocessing, the features were standard scaled, and then reduced using Recursive Feature Elimination. The minimum features required during this step was 20 and the estimator was BayesianRidge() due to it being the default estimator for SelectKBest(). This reduced the number of features from 71 to 39. Those 32 dropped features were then used in PCA to capture signal across the top 5 PC features by variance explained. All told, 56.84 of the cumulative variance explained by the 32 dropped features was added back to the model.

<p align="center"><img src=assets/pca.png alt="drawing" width="400" class="center"/></p>

## Modelling Full Approach
Three regression models were trained. Random Forest Regression hyperparameters were optimized using Halving Grid Search and resulted in a score of 35.9% against the 29.6% baseline. SVM Regression hyperparameters were optimized using Grid Search and result in a score of 34.0%. Linear regression without regularization scored 39.0% and lasso feature selection got it up to 39.1%. This became the production model.

## RDF and YDF Approach
A problem I foresaw with trying to model WAA_pg was that it relied on both a player's performance as well as the league environment. Trying to model both simultaneously seemed to be asking too much and could be simplified by trying to model the player's performance and the league environment separately. For example: Run variables are correlated with WAA_pg while the league environment variable opponent runs per game isn't.

<p align="center"><img src=assets/runs.png alt="drawing" width="800" class="center"/></p>
<p align="center"><img src=assets/opprpg.png alt="drawing" width="400" class="center"/></p>

Note how there are bands of values for opprpg. Those values are constant for a season and will span the WAA_pg of every player who played in that season. While uncorrelated, the opprpg is used in the calculation of WAA to translate runs into wins.

With runs and opprpg being critical parrts of the WAA equation, I decided to preprocess them individually in preparation for modelling them individually. The runs and opprpg datasets were both standard scaled. The remaining features not in RDF or YDF were also standard scaled with the number of features reduced from 53 to 20 using Recursive Feature Elimination (RFE). The noise from those 33 eliminated features was then added back in using Principal Component Analysis, and 5 PCA features were added back to this non-rdf/ydf dataset for a total of 25 features. These 5 PCA features captured over half of the variance explained.

## Modelling RDF and YDF
Both the runs dataframe (RDF) and the opprpg (year dataframe aka YDF) were modelled using a Linear Regression with and without regularization, Random Forest Regression, and Support Vector Regression. Hyperparameters for Random Forest for RDF and YDF were tuned using Halving Grid Search, while SVM Regression used Grid Search. The findings were RDF performed best Linear Regression with Lasso feature selection for as score of 37.1 against a standard regression baseline of 0, while YDF performed best with Random Forest with a score of 96% against a standard regression baseline of 0.

The predicted values on the test set were sent over to be used in modeling where they were combined with the rest of the features not included in RDF and YDF. The same regression model types were used and Random Forest Regression performed the best with a score of 37.3% against the 31% baseline (the baseline was actually higher during this approach).

Ultimately, separating RDF and YDF did not yield better results.

## Results and Conclusions
The production model outperformed the baseline using the Full Approach by 10%. The linear model that achieved the score showed the most important features to be the previous 3 years of waa_pg where waa_pg_1yr was 1.2 times more impactful than waa_pg_2yr and 1.5 times more than waa_pg_3yr.

Age was the leading negative contributor with each year of age negatively impacting predicted waa_pg the same as 0.18 standard deviations of waa_pg_1yr--the leading positive contributor.

<p align="center"><img src=assets/model.png alt="drawing" width="800" class="center"/></p>

The graph of residuals is still very blobby and spread out, but given its R2 score is 39.1%, that is expected. The thing that is important is it beat the baseline ... a baseline calculated as an intuitive metric that could feasibly be getting used currently to trend players today.

## Future Developments
So many things can be added to this model to try and capture additional noise. The most surprising thing was the poor performance of the RDF models. It would seem that the previous 3 years of a player's career would be a good indicator for future performance, but it just wasn't the case. Looking deeper into the nuts and bolts of the normalizing that Baseball-Reference does could also help tighten the model--there may be more at play with the league environmental variables beyond just the average amount of runs being scored per game. Applying more statsitics could also help; there may be some signal captured by including more traditional metrics like RBIs and HRs.

Future work on this would include adding pitchers as well as adding the ability to forecast beyond one year. This would allow for evaluation of players over a more traditionally-lengthed contract like 2-3 years.
