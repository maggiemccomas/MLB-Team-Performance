# MLB-Team-Performance
This project focuses on analyzing Major League Baseball (MLB) team performance by employing data analysis and machine learning techniques. The dataset used for this analysis is sourced from the Lahman Baseball Database, spanning from 1871 to 2022. The analysis covers various aspects such as distribution of wins, runs per game, decades of performance, and clustering using KMeans. Additionally, predictive models utilizing Linear Regression and Ridge Regression are built to forecast team wins based on selected attributes.

### Libraries 
The project utilizes several Python libraries, including:
1. SQLite for database management
2. NumPy and Pandas for data handling and manipulation
3. Matplotlib for visualization
4. Scikit-learn for machine learning tasks

### Data Collection and Processing 
The data is extracted from the Lahman Baseball Database, specifically focusing on teams that have played a significant number of games (150 or more). The dataset is processed to extract relevant columns and perform necessary data cleaning.

### Exploratory Data Analysis
Exploratory data analysis includes visualizations of the distribution of wins, runs per game, MLB yearly runs per game, and the relationship between runs, wins, and runs allowed per game. Additionally, the dataset is categorized into different eras and decades for a comprehensive historical perspective.

### KMeans Clustering
The project employs KMeans clustering to categorize MLB teams based on various performance attributes. The optimal number of clusters is determined using silhouette scores, and the resulting clusters are visualized for further analysis.

### Model Predictions
Two predictive models, Linear Regression and Ridge Regression, are implemented to forecast the number of wins for MLB teams. The models are trained, evaluated, and their predictions are visualized to assess their performance.
