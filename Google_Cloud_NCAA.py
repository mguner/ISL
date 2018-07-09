
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn


# In[2]:


#returns a dataframe that includes all the events happened in 'year' since 2009 - 2010
def event_data(year):
    return(pd.read_csv('/Users/user/Desktop/proje/EGMStudyGroup/NCAA/data/PlayByPlay_{}/Events_{}.csv'.format(year,year)))
# Returns a dataframe with variables playerid, teamid and playerName since 2009-2010.
def player_data(year):
    return pd.read_csv('/Users/user/Desktop/proje/EGMStudyGroup/NCAA/data/PlayByPlay_{}/Players_{}.csv'.format(year,year))


# In[3]:


df = event_data(2015)
df['new'] = 1
df.head()


# In[4]:


#let's create a table that shows every players statistics and TeamId for a given game.
table = pd.pivot_table(df,fill_value= 0, values= 'new', index=['DayNum', 'EventTeamID','EventPlayerID'],
                     columns=['EventType'], aggfunc = np.sum )
#let's see a particular subset of this table
table.query('DayNum == 11 and (EventTeamID == 1103 or EventTeamID == 1420)')


# In[5]:


# List of columns in table.
table.loc[:,'assist': 'made1_free']
table.columns


# In[6]:


# table_player_2015 gives the total of events for each player in 2015 we have 5442 players' data in this datafram
table_player_2015 = table.groupby('EventPlayerID').apply(lambda x: x[['assist', 'block',
       'foul_pers', 'foul_tech', 'made1_free', 'made2_dunk', 'made2_jump',
       'made2_lay', 'made2_tip', 'made3_jump', 'miss1_free', 'miss2_dunk',
       'miss2_jump', 'miss2_lay', 'miss2_tip', 'miss3_jump', 'reb_dead',
       'reb_def', 'reb_off', 'steal', 'sub_in', 'sub_out', 'timeout',
       'timeout_tv', 'turnover']].sum() )
table_player_2015.shape


#  __[For detailed explanations of the datasets: Link to Kaggle](https://www.kaggle.com/c/mens-machine-learning-competition-2018/data)__

# In[7]:


#This file identifies the different college teams present in the dataset.
teams = pd.read_csv('/Users/user/Desktop/proje/EGMStudyGroup/NCAA/data/DataFiles/Teams.csv')
teams.head()


# In[8]:


#This file identifies the different seasons included in the historical data,
#along with certain season-level properties.
seasons = pd.read_csv('/Users/user/Desktop/proje/EGMStudyGroup/NCAA/data/DataFiles/Seasons.csv')
seasons.head()


# In[9]:


# This file identifies the seeds for all teams in each NCAA® tournament, for all seasons of historical data.
seeds = pd.read_csv('/Users/user/Desktop/proje/EGMStudyGroup/NCAA/data/DataFiles/NCAATourneySeeds.csv')
seeds.head()


# In[10]:


#This file identifies the game-by-game results for many seasons of historical data, starting with the 1985 season
reg_season_cpt = pd.read_csv('/Users/user/Desktop/proje/EGMStudyGroup/NCAA/data/DataFiles/RegularSeasonCompactResults.csv')
reg_season_cpt.head()


# In[11]:


reg_season_cpt.shape


# In[12]:


#This file identifies the game-by-game NCAA® tournament results for all seasons of historical data.
tour_results = pd.read_csv('/Users/user/Desktop/proje/EGMStudyGroup/NCAA/data/DataFiles/NCAATourneyCompactResults.csv')
tour_results.head()


# In[13]:


#This file provides team-level box scores for many regular seasons of historical data, starting with the 2003 season. 
seas_detail = pd.read_csv('/Users/user/Desktop/proje/EGMStudyGroup/NCAA/data/DataFiles/RegularSeasonDetailedResults.csv')
seas_detail.head()
#seas_detail.columns


# In[14]:


seas_detail.shape


# In[15]:


#This file provides team-level box scores for many NCAA® tournaments, starting with the 2003 season.
tour_detail = pd.read_csv('/Users/user/Desktop/proje/EGMStudyGroup/NCAA/data/DataFiles/NCAATourneyDetailedResults.csv')
tour_detail.head()


# In[16]:


tour_detail.shape


# In[17]:


#This file indicates the Division I conferences that have existed over the years since 1985.
conferences = pd.read_csv('/Users/user/Desktop/proje/EGMStudyGroup/NCAA/data/DataFiles/Conferences.csv')
conferences.head()


# In[18]:


#This file indicates the conference affiliations for each team during each season. 
#Some conferences have changed their names from year to year, and/or changed which teams are part of the conference.
#This file tracks this information historically.

team_conferences = pd.read_csv('/Users/user/Desktop/proje/EGMStudyGroup/NCAA/data/DataFiles/TeamConferences.csv')


# In[19]:


#This file indicates the head coach for each team in each season, 
#including a start/finish range of DayNums to indicate a mid-season coaching change. 
coaches = pd.read_csv('/Users/user/Desktop/proje/EGMStudyGroup/NCAA/data/DataFiles/TeamCoaches.csv')
coaches.head()


# ### Most Succesfull Coach?

# In[20]:


# Let's find the coaches changes teams during the season
coaches[(coaches['FirstDayNum'] != 0)|(coaches['LastDayNum'] != 154)];


# In[21]:


# Let's see the coaches and the number of teams they worked for.
coaches.groupby(['CoachName']).TeamID.unique(); # this will give the list of teams for each coach and the teams they have worked.
coaches.groupby(['CoachName', 'TeamID']).Season.count(); # number of seasons a coach worked for a team.


# ### Most Succesfull Teams? By Number of Champs

# In[22]:


champ_count = tour_results[tour_results['DayNum'] == 154].WTeamID.value_counts()

champ_teams = teams[teams.TeamID.isin(list(champ_count.index))][['TeamID','TeamName']]
champ_teams['no_champs'] = champ_teams.TeamID.map(champ_count)
Champ_Teams = champ_teams.sort_values(by = 'no_champs', ascending = False)


# # Most Succesfull Team? By Average seed:

# In[23]:


def TeamID_to_Tname(series_TeamID):
    series_teamname = teams.set_index('TeamName').TeamID.map(series_TeamID).sort_values(ascending = False)
    return series_teamname
def PlayerID_to_TeamID(list_PlayerID, season):
    team_player = pd.read_csv('/Users/user/Desktop/proje/EGMStudyGroup/NCAA/data/PlayByPlay_{}/Players_{}.csv'.format(season,season))
    teamID = team_player[team_player.isin(list_PlayerID)]
    return teamID


# In[24]:


seeds['Seed'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
seeds['Seed_Numeric'] = pd.to_numeric(seeds.Seed)
Seed_Means = seeds.groupby('TeamID').Seed_Numeric.mean();


# In[25]:


Seeds_Teams = teams[teams.TeamID.isin(list(Seed_Means.index))][['TeamID','TeamName']]
# Seeds_Teams['avg_seeds'] = Seeds_Teams.TeamID.map(Seed_Means)
# Seed_Teams_15 = Seeds_Teams.sort_values(by = 'avg_seeds', ascending = True)[0:16]
# Seed_Teams_15[['TeamName', 'avg_seeds']].plot.bar(x = 'TeamName')



# In[26]:


Seed_Best = seeds[seeds['Seed_Numeric']==1].TeamID.value_counts()
Seeds_Best_Teams = teams[teams.TeamID.isin(list(Seed_Best.index))][['TeamID','TeamName']]
Seeds_Best_Teams['No_first_seed'] = Seeds_Best_Teams.TeamID.map(Seed_Best)
Seeds_Best_Teams = Seeds_Best_Teams.sort_values( by = 'No_first_seed', ascending = False)[0:14]



# In[27]:


def create_game_stats(events):
    # Day number, winning team id and losing team id uniquely determines the game id
    col = events['DayNum'].astype(str) + "_" + events['WTeamID'].astype(str) + "_" + events['LTeamID'].astype(str)
    events['GameID'] = col
    # new column indicating the winning team
    events.EventTeamID = (events.WTeamID == events.EventTeamID)
    events.EventTeamID = events.EventTeamID.astype(int)

    g = events.groupby(['GameID', 'EventTeamID', 'EventType']).agg({'EventType': 'count'})
    us = g.unstack(['EventTeamID', 'EventType'], fill_value=0)

    return us.sort_index(axis=1)


# In[28]:


season_wins = reg_season_cpt['WTeamID']
s1 = season_wins.value_counts()


# In[29]:


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

f = fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (15,15))
f11 = TeamID_to_Tname(s1)[0:17].plot.barh(ax = ax1)
f11.set_title('Wins Between 1983 -2017')
f12 = Seeds_Best_Teams[['TeamName', 'No_first_seed']].plot.barh(ax = ax2, x = 'TeamName')
f12.set_title('Best Seeds 1983 - 2017')
f21 = TeamID_to_Tname(Seed_Means).sort_values(ascending = True)[0:16].plot.bar(ax = ax3, x = 'TeamName')
f21.set_title('Avg Seed')
f22 = Champ_Teams[['no_champs','TeamName']].plot.bar(ax = ax4, x = 'TeamName')
f22.set_title('Number of Champs 1983 - 2017')


# # Here we want to predict game result

# First we want to write a function that will return our training sets

# In[41]:


def create_dataset(stats, matches):
#     """
#     Create dataset from team statistics and match results.
#     In input, each row is the team stats of first team followed by team stats of second team.
#     In output, each row is the result of the corresponding match, with 1 denoting first team win.

#     :param stats: A dataframe of team statistics with total event counts for each team in each row.
#     :param matches: A dataframe of match results with winning team and losing team ids for one match in each row.
#     :return: x, y. Two numpy arrays of stats and match results.
#     """
    num_matches = matches.shape[0]
    feats_per_team = stats.shape[1]

    # allocate space for x and y
    x = np.zeros((num_matches, feats_per_team * 2))
    y = np.zeros(num_matches)

    i = 0
    skipped = 0  # if we can't find a team in stats dataframe, we skip that match.
    for _, row in matches.iterrows():  # loop over matches
        wteam = row.WTeamID
        lteam = row.LTeamID
        try:
            wteam_stats = stats.loc[wteam].values
            lteam_stats = stats.loc[lteam].values
        except KeyError:
            #print("Can't find either {} or {}. Skipping.".format(wteam, lteam))
            skipped += 1
            continue

        # pick a random order for winning/losing teams.
        if np.random.rand() > 0.5:
            x[i, 0:feats_per_team] = wteam_stats
            x[i, feats_per_team:] = lteam_stats
            y[i] = 1
        else:
            x[i, 0:feats_per_team] = lteam_stats
            x[i, feats_per_team:] = wteam_stats
            y[i] = 0
        i += 1

    # if we skipped some matches, x and y has extra empty rows. do not return these.
    x = x[0:(num_matches-skipped)]
    y = y[0:(num_matches-skipped)]
    return x, y


# We will read the datasets and manipulate them in such a form that we can feed the previous function

# In[62]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
stats_season = 2017
predict_season = 2017

# load precalculated stats from disk
# team_stats = pd.read_csv('data/TeamStats_2010.csv', index_col=0, header=[0, 1])



# In[63]:


# get stats for each team in 2017 season
events = pd.read_csv('data/PlayByPlay_{}/Events_{}.csv'.format(stats_season, stats_season))
team_stats = events[events['DayNum'] < 134].groupby(['EventTeamID', 'EventType']).agg({'EventType': 'count'})
team_stats = team_stats.unstack('EventType')
team_stats.fillna(0, inplace=True)
team_stats.head()


# In[64]:


# get match results for 2017 season
season_results = pd.read_csv('data/RegularSeasonCompactResults.csv')
season_results = season_results[season_results.Season == predict_season]

ncaa_results = pd.read_csv('data/NCAATourneyCompactResults.csv')
ncaa_results = ncaa_results[ncaa_results.Season == predict_season]
ncaa_results.head()


# We will call the create_dataset function with ncaa_results as matches and team_stats as stats

# In[65]:


test_x, test_y = create_dataset(team_stats, ncaa_results)
train_x, train_y = create_dataset(team_stats, season_results)


# In[66]:


# Let's double check whether our datasets are in appropriate forms
shapes_of_data = (test_x.shape , test_y.shape, train_x.shape, train_y.shape )
shapes_of_data


# In[74]:


# Let's see how KNN- algorithm predict the results: K = 5
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(train_x, train_y)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(train_x, train_y)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(test_x, test_y)))


# In[68]:


# Let's see how KNN - algorithm with normalization predict the results: K = 5
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train_x)
X_test_scaled = scaler.transform(test_x)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_scaled, train_y)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train_scaled, train_y)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test_scaled, test_y)))


# In[69]:


# Let's see how KNN- algorithm predict the results: K = 4
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train_x)
X_test_scaled = scaler.transform(test_x)
knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(X_train_scaled, train_y)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train_scaled, train_y)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test_scaled, test_y)))


# In[70]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
X_train_normalized = preprocessing.normalize(train_x, norm='l2')
X_test_normalized = preprocessing.normalize(test_x, norm='l2')
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train_normalized, train_y)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train_normalized, train_y)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test_normalized, test_y)))


# In[71]:


# Let's see how KNN- algorithm predict the results: K = 2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train_x)
X_test_scaled = scaler.transform(test_x)
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train_scaled, train_y)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train_scaled, train_y)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test_scaled, test_y)))


# In[72]:


# Now we will use logistic regression to predict game
#results for 2010 NCAA results from regular season data 2010
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(train_x, train_y)
test_acc = clf.score(test_x, test_y)
training_acc = clf.score(train_x , train_y)
print("Test accuracy: {}".format(test_acc))
print("Training accuracy: {}".format(training_acc))


# In[73]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty= 'l1', C = .01, max_iter= 4000, solver = 'saga')
clf.fit(train_x, train_y)
test_acc = clf.score(test_x, test_y)
training_acc = clf.score(train_x , train_y)
print("Test accuracy : {}".format(test_acc))
print("Training accuracy : {}".format(training_acc))


# In[61]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty= 'l2', C = 1, max_iter= 4000, solver = 'sag')
clf.fit(train_x, train_y)
test_acc = clf.score(test_x, test_y)
training_acc = clf.score(train_x , train_y)
print("Test accuracy : {}".format(test_acc))
print("Training accuracy : {}".format(training_acc))


# In[49]:


# Now we will use logistic regression with l1 penalty and various C values 
# to predict game results for 2010 NCAA results
from sklearn.linear_model import LogisticRegression
for param in list(np.linspace(1,100,5)):
    clf = LogisticRegression(penalty= 'l1', C = param, max_iter= 700, solver = 'saga')
    clf.fit(train_x, train_y)
    test_acc = clf.score(test_x, test_y)
    training_acc = clf.score(train_x , train_y)
    print("Test accuracy C = {: .2f}: {}".format(param,test_acc))
    print("Training accuracy C = {: .2f}: {}".format(param, training_acc))
# The same result without penalty


# In[ ]:


# Now we will use logistic regression with l2 penalty with various C
# to predict game results for 2010 NCAA results
from sklearn.linear_model import LogisticRegression
for param in np.linspace(1,100,10):
    clf = LogisticRegression(penalty= 'l2', C = param, max_iter= 1000, solver= 'newton-cg')
    clf.fit(train_x, train_y)
    test_acc = clf.score(test_x, test_y)
    training_acc = clf.score(train_x , train_y)
    print("Test accuracy for C = {}: {}".format(param, test_acc))
    print("Training accuracy for C = {} : {}".format(param, training_acc))
# The same result without penalty


# In[ ]:


# Normalized Logistic Reg. with L2 penalty with various C
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train_x)
X_test_scaled = scaler.transform(test_x)
for param in list(np.linspace(1,100,10)):
    clf = LogisticRegression(penalty= 'l2', C = param, solver= 'lbfgs' )
    clf.fit(X_train_scaled, train_y)
    test_acc = clf.score(X_test_scaled, test_y)
    training_acc = clf.score(X_train_scaled , train_y)
    print("Test accuracy C = {:.2f} : {}".format(param, test_acc))
    print("Training accuracy C = {:.2f} : {}".format(param, training_acc))
# A little better than without scaling


# In[ ]:


# Normalized Logistic Reg. with L2 penalty with various C
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train_x)
X_test_scaled = scaler.transform(test_x)
for param in list(np.linspace(1,100,10)):
    clf = LogisticRegression(penalty= 'l2', C = param, solver = 'lbfgs', max_iter=1000 )
    clf.fit(X_train_scaled, train_y)
    test_acc = clf.score(X_test_scaled, test_y)
    training_acc = clf.score(X_train_scaled , train_y)
    print("Test accuracy C = {:.2f} : {}".format(param, test_acc))
    print("Training accuracy C = {:.2f} : {}".format(param, training_acc))
# A little better than without scaling


# In[ ]:


# Normalized Logistic Reg. with L1 penalty with various C
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train_x)
X_test_scaled = scaler.transform(test_x)
for param in list(np.linspace(1,100,10)):
    clf = LogisticRegression(penalty= 'l1', C = param)
    clf.fit(X_train_scaled, train_y)
    test_acc = clf.score(X_test_scaled, test_y)
    training_acc = clf.score(X_train_scaled , train_y)
    print("Test accuracy C = {:.2f} : {}".format(param, test_acc))
    print("Training accuracy C = {:.2f} : {}".format(param, training_acc))
# A little better than without scaling

