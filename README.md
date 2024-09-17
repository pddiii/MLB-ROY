# MLB-ROY
Let's talk about the MLB Rookie of the Year Race

## Data

- 1974 to 2024 rookie data
  - Excluded 1994 and 2020 since they were shortened seasons
  - 1974-2023 were used for Training/Testing Data
  - 2024 was used to make predictions
- Starters: minimum 100 Innings Pitched (IP)
- Relievers: minimum 40 Innings Pitched (IP)
- Batters: minimum 300 Plate Appearances (PA)

## Data Cleaning

[Data Cleaning Notebook](/scripts_and_notebooks/roy_cleaning.ipynb)

### Helper Functions

I utilized several functions repeatedly throughout the modeling process.

In order to reduce the lines of code during the modelling I created a python script (.py file) which contains these various helper functions.

[Helper Functions Script](/scripts_and_notebooks/helper_functions.py)

### Source

There were several sources for the data utilized in this project.

The data in the [Awards Folder](data/awards/) is sourced from either `Lahman` database or `baseballr`.

The data in the [Fielding Folder](data/fielding/) is sourced from the `Lahman` database

The data in the [MLB Folder](data/mlb/) and the data in the [Rookies Folder](data/rookies/) were sourced from FanGraphs utilizing their custom reports feature.

The [Cleaned Player IDs](data/cleaned_player_ids.csv) were sourced from a previous project of mine which combined player ids from the Lahman Database and the more frequently updated [PlayerIDMap](https://docs.google.com/spreadsheets/d/1JgczhD5VDQ1EiXqVG-blttZcVwbZd5_Ne_mefUGwJnk/pubhtml?gid=0&single=true).

## Models

- For both models I fit the model to three different data sets:
  - One for relievers
  - One for starters
  - One for batters

### Vote Recipients

- Output: Probability between 0 and 1 for receiving a Rookie of the Year vote (`vote_getter`)
- Took the top 8 for 2024 from both the AL and NL
- Utilize these predictions in the predictions for the 2024 Rookie of the Year model
  - Rounded the top 8 vote getters to a 1, and the rest to a 0 for proper interpretation in the Rookie of the Year model predictions

**Predictions**

[Vote Getter Predictions](/data/predictions/vote_preds.csv)

#### Relievers Variables

- The `_int` variabels were created to study the interactions between those numeric variables where the interaction is formed by multiplying the numeric variables together.

| Feature                | Feature            | Feature                | Feature                |
|------------------------|--------------------|------------------------|------------------------|
| IP_int_K%+             | K%+_int_ERA-       | ERA-_int_FIP-          | K-BB%_int_FIP-         |
| SD                     | IP_int_K/BB+       | WAR                    | K-BB%                  |
| SD_int_MD              | K/BB+              | SD_pct                 | K-BB%_int_ERA-         |
| WPA                    | WHIP+_int_ERA-     | ERA-                   | HR/9+                  |
| K%+                    | IP                 | K-BB%_int_FIP-         | ERA-_int_FIP-_int_WPA  |
| BB%+                   | ERA-_int_FIP-      | WPA/LI                 | decade                 |
| IP_int_K%+             | WAR/IP             | WHIP+                  | WHIP+_int_FIP-         |
| K%+                    | IP_int_ERA-        | ERA-_int_FIP-_int_WAR  |                        |

#### Starters Variables

| Feature                | Feature            | Feature            | Feature            | Feature            |
|------------------------|--------------------|--------------------|--------------------|--------------------|
| WPA/LI                 | WPA/IP             | HR/9+              | K%+                | ERA-_int_FIP-_int_WPA |
| IP                     | IP_int_K%+         | IP_int_K/BB+       | K-BB%_int_ERA-     |                    |
| ERA-                   | WPA                | IP_int_FIP-        | WAR                |                    |

#### Batters Variables

| Feature                | Feature            | Feature            | Feature            | Feature            |
|------------------------|--------------------|--------------------|--------------------|--------------------|
| PA_int_WAR             | wOBA               | Def                | OBP+_int_SLG+      | Def_int_Off_int_WAR |
| PA                     | wOBA_int_BB/K      | K%+                | decade             | wRC+               |
| OBP+                   | WPA/LI             | Off                | BB/K               |                    |
| BB%+                   | SLG+               | Def_int_Off_int_WPA | Def_int_Off        |                    |
| WAR/G                  | BB/K               | Def_int_Off        |                    |                    |


### Rookie of the Year

- Output: Probability between 0 and 1 for winning Rookie of the Year (`rookie_of_the_year`)
- Utilize these predictions to discuss the possible Rookie of the Year candidates

**Predictions**

[Rookie of the Year Predictions](/data/predictions/vote_roy_preds.csv)

#### Relievers Variables

| Feature                | Feature            | Feature            | Feature            | Feature            |
|------------------------|--------------------|--------------------|--------------------|--------------------|
| WAR                    | ERA-_int_FIP-_int_WPA | vote_getter       | K%+                | BB%+               |
| SD_int_MD              | IP_int_K/BB+       | HR/9+              | K%+_int_BB%+       |                    |

#### Starters Variables

| Feature                | Feature            | Feature                | Feature            | Feature            |
|------------------------|--------------------|------------------------|--------------------|--------------------|
| ERA-_int_FIP-_int_WAR  | WPA/IP             | vote_getter            | ERA-_int_FIP-_int_WPA | AVG+               |
| HR/9+                  | HR/9+_int_BB%+     | IP_int_K%+             | K-BB%_int_FIP-      | BB%+               |

#### Batters Variables

| Feature            | Feature        | Feature      |
|--------------------|----------------|--------------|
| wRC+_int_WAR       | PA             | vote_getter  |