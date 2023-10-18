
# libraries
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# get data

databaseConnect = sqlite3.connect('Data/lahman_1871-2022.sqlite')
query = '''select * from Teams inner join TeamsFranchises on Teams.franchID == TeamsFranchises.franchID
where Teams.G >= 150 and TeamsFranchises.active == 'Y'; '''
teams = databaseConnect.execute(query).fetchall()
teamsDF = pd.DataFrame(teams)

# process data

coloumns = ['year', 'league', 'team', 'franchise', 'division', 'rank', 'games_played', 'home_games', 'wins', 'loses', 
            'division_winner', 'world_cup', 'league_champion', 'world_series_winner', 'runs_scored', 'at_bats', 'hits', 
            '2B', '3B', 'HR', 'walks', 'SO', 'stolen_base', 'caught_stealing', 'hit_by_pitch', 'sacrifice_flies', 
            'opponents_runs_scored', 'earned_runs_allowed', 'earned_runs_average', 'completed_games', 'shutouts', 'saves', 
            'outs_pitched', 'hits_allowed', 'homeruns_allowed', 'walks_allowed', 'strikeouts_by_pitcher', 'errors',
            'double_plays', 'fielding_percentage', 'team_name', 'park', 'attendance','bat_performance_factor',
            'pitching_park_factor', 'team_bats_right', 'teamIDlahman45', 'team_retro', 'franchise', 'franchise_name', 
            'active', 'NAassoc']
teamsDF.columns = coloumns
drop_coloumns = ['league', 'franchise', 'division', 'rank', 'home_games', 'division_winner', 'world_cup', 'league_champion', 
                 'world_series_winner', 'caught_stealing', 'hit_by_pitch', 'sacrifice_flies', 'team_name', 'park', 'attendance','bat_performance_factor',
                 'pitching_park_factor', 'team_bats_right', 'teamIDlahman45', 'team_retro', 'franchise', 'franchise_name', 'NAassoc']
dataDF = teamsDF.drop(drop_coloumns, axis = 1)
dataDF['SO'] = dataDF['SO'].fillna(dataDF['SO'].median())

# Distribution of wins visualization

plt.hist(dataDF['wins'], linewidth = 1, edgecolor = 'black')
plt.xlabel('Wins')
plt.ylabel('Frequency')
plt.title('Distribution of Wins')
print('Mean Number of Wins: ', dataDF['wins'].mean())
plt.savefig('distribution_of_wins.png', dpi = 300)
plt.close()

# Wins

def assign_win_bin(wins):
    if wins < 50:
        return 1
    elif wins >= 50 and wins < 70:
        return 2
    elif wins >= 70 and wins < 90:
        return 3
    elif wins >= 90 and wins < 110:
        return 4
    else:
        return 5
    
dataDF['win_bin'] = dataDF['wins'].apply(assign_win_bin)
dataDF = dataDF[dataDF['year'] > 1900]

plt.scatter(dataDF['year'], dataDF['wins'], c = dataDF['win_bin'], edgecolor = 'black')
plt.title('Wins Scatter Plot')
plt.xlabel('Year')
plt.ylabel('Wins')
plt.savefig('wins_scatter_plot.png', dpi = 300)
plt.close()

# Runs

runs_per_year = {}
games_per_year = {}

for i, row in dataDF.iterrows():
    year = row['year']
    runs = row['runs_scored']
    games = row['games_played']
    if year in runs_per_year:
        runs_per_year[year] = runs_per_year[year] + runs
        games_per_year[year] = games_per_year[year] + games
    else:
        runs_per_year[year] = runs
        games_per_year[year] = games

runs_per_game_per_year = {}

for y, g in games_per_year.items():
    year = y
    games = g
    runs = runs_per_year[year]
    runs_per_game_per_year[year] = runs / games

runs_list = sorted(runs_per_game_per_year.items())
x, y = zip(*runs_list)

plt.plot(x, y)
plt.title('MLB Yearly Runs per Game')
plt.xlabel('Year')
plt.ylabel('MLB Runs per Game')
plt.savefig('mlb_yearly_runs_per_game.png', dpi = 300)
plt.close()

# Years

def assign_label(year):
    if year < 1920:
        return 1
    elif year >= 1920 and year < 1942:
        return 2
    elif year >= 1942 and year < 1946:
        return 3
    elif year >= 1946 and year < 1963:
        return 4
    elif year >= 1963 and year < 1977:
        return 5
    elif year >= 1977 and year < 1993:
        return 6
    elif year >= 1993 and year < 2010:
        return 7
    else:
        return 8

dataDF['year_label'] = dataDF['year'].apply(assign_label)
dummyDF = pd.get_dummies(dataDF['year_label'], prefix = 'era')
dataDF = pd.concat([dataDF, dummyDF], axis = 1)

def assign_mlb_rpg(year):
    return runs_per_game_per_year[year]

dataDF['mlb_rpg'] = dataDF['year'].apply(assign_mlb_rpg)

# Decades

def assign_decade(year):
    if year < 1920:
        return 1910
    elif year >= 1920 and year < 1930:
        return 1920
    elif year >= 1930 and year < 1940:
        return 1930
    elif year >= 1940 and year < 1950:
        return 1940
    elif year >= 1950 and year < 1960:
        return 1950
    elif year >= 1960 and year < 1970:
        return 1960
    elif year >= 1970 and year < 1980:
        return 1970
    elif year >= 1980 and year < 1990:
        return 1980
    elif year >= 1990 and year < 2000:
        return 1990
    elif year >= 2000 and year < 2010:
        return 2000
    elif year >= 2010 and year < 2020:
        return 2010
    else:
        return 2020

dataDF['decade'] = dataDF['year'].apply(assign_decade)
decadeDF = pd.get_dummies(dataDF['decade'], prefix = 'decade')
dataDF = pd.concat([dataDF, decadeDF], axis = 1)
dataDF = dataDF.drop(['win_bin', 'year_label', 'decade'], axis = 1)

dataDF['runs_per_game'] = dataDF['runs_scored'] / dataDF['games_played']
dataDF['runs_allowed_per_game'] = dataDF['opponents_runs_scored'] / dataDF['games_played']

fig = plt.figure(figsize = (12,6))
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(dataDF['runs_per_game'], dataDF['wins'], c = 'lightblue', edgecolor = 'black')
ax1.set_title('Runs per Game vs. Wins')
ax1.set_ylabel('Wins')
ax1.set_xlabel('Runs per Game')
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(dataDF['runs_allowed_per_game'], dataDF['wins'], c = 'tomato', edgecolor = 'black')
ax2.set_title('Runs Allowed per Game vs. Wins')
ax2.set_xlabel('Runs Allowed per Game')
plt.savefig('runs_per_game_vs_wins.png', dpi = 300)
plt.close()

numericDF = dataDF.drop(['year', 'loses', 'team', 'active'], axis = 1)
print(numericDF.corr()['wins'])

# K Means

attributes = ['games_played','runs_scored','at_bats','hits','2B','3B','HR','walks','SO','stolen_base','opponents_runs_scored','earned_runs_allowed','earned_runs_average','completed_games',
'shutouts','saves','outs_pitched','hits_allowed','homeruns_allowed','walks_allowed','strikeouts_by_pitcher','errors','double_plays','fielding_percentage','era_1','era_2','era_3','era_4','era_5','era_6','era_7','era_8',
'decade_1910','decade_1920','decade_1930','decade_1940','decade_1950','decade_1960','decade_1970','decade_1980',
'decade_1990','decade_2000','decade_2010','runs_per_game','runs_allowed_per_game','mlb_rpg']
data_attributes = dataDF[attributes]

s_score_dict = {}

for i in range(2,11):
    km = KMeans(n_clusters = i, random_state = 1, n_init = 10)
    labels = km.fit_predict(data_attributes)
    s_s = metrics.silhouette_score(data_attributes, labels)
    s_score_dict[i] = [s_s]

kmeans_model = KMeans(n_clusters = 6, random_state = 1)
distances = kmeans_model.fit_transform(data_attributes)

labels = kmeans_model.labels_
plt.scatter(distances[:,0], distances[:,1], c = labels, edgecolor = 'black')
plt.title('Kmeans Clusters')
plt.savefig('kmeans_clusters.png', dpi = 300)
plt.close()

# Model Predictions

dataDF['labels'] = labels
numericDF['labels'] = labels
attributes.append('labels')

trainData = numericDF.sample(frac = 0.75, random_state = 1)
x_train = trainData[attributes]
y_train = trainData['wins']
testData = numericDF.loc[~numericDF.index.isin(trainData.index)]
x_test = testData[attributes]
y_test = testData['wins']

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Linear Regression

modelLR = LinearRegression()
modelLR.fit(x_train_scaled, y_train)
predictionsLR = modelLR.predict(x_test_scaled)
print("Linear Regression Predictions:")
lr_predictions_df = pd.DataFrame({'Actual Wins': y_test, 'Predicted Wins (LR)': predictionsLR})
lr_predictions_df = pd.concat([x_test.reset_index(drop=True), lr_predictions_df], axis=1)
lr_predictions_df = lr_predictions_df.dropna(subset=['Actual Wins', 'Predicted Wins (LR)'])
print(lr_predictions_df.head())

maeLR = mean_absolute_error(y_test, predictionsLR)
print("Mean Absolute Error (Linear Regression):", maeLR)

plt.scatter(y_test, predictionsLR, color = '#88c999', edgecolors='#36503D')
plt.plot(y_test, y_test, color = 'black', linestyle = '--')
plt.xlabel('Actual Wins')
plt.ylabel('Predicted Wins')
plt.title('Actual vs. Predicted Wins')
plt.savefig('linear_regression.png', dpi = 300)
plt.close()

# Ridge Regression

alphas = {'alpha': [0.01, 0.05, 0.1, 0.15, 0.5, 1.0, 10.0]}

ridge = Ridge()
grid_search = GridSearchCV(ridge, alphas, cv=5)
grid_search.fit(x_train_scaled, y_train)
print("Best alpha:", grid_search.best_params_)

alphas = [0.01, 0.05, 0.1, 0.15, 0.5, 1.0, 10.0]

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, x_train_scaled, y_train, cv = 5)
    print(f'Alpha: {alpha}, Mean CV Score: {np.mean(scores)}')

modelRR = Ridge(alpha = 0.01)
modelRR.fit(x_train_scaled, y_train)
predictionsRR = modelRR.predict(x_test_scaled)
print("\nRidge Regression Predictions:")
rr_predictions_df = pd.DataFrame({'Actual Wins': y_test, 'Predicted Wins (RR)': predictionsRR})
rr_predictions_df = pd.concat([x_test.reset_index(drop=True), rr_predictions_df], axis=1)
rr_predictions_df = rr_predictions_df.dropna(subset=['Actual Wins', 'Predicted Wins (RR)'])
print(rr_predictions_df.head())

mae_rrm = mean_absolute_error(y_test, predictionsRR)
print("Mean Absolute Error (Ridge Regression):", mae_rrm)

plt.scatter(y_test, predictionsRR, color = '#88c999', edgecolor = '#36503D')
plt.plot(y_test, y_test, color = 'black', linestyle = '--')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs. Predictions for Ridge Regression')
plt.savefig('ridge_regression.png', dpi = 300)
plt.close()