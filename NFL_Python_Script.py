#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sportsreference.nfl.teams import Teams
from sportsreference.nfl.roster import Roster
from sportsreference.nfl.roster import Player

import numpy as np
import pandas as pd


# In[3]:


team_array = []

teams = Teams()
for team in teams:
    team_array.append(team.abbreviation)
print(team_array)


# In[4]:


player_array = []
tm_n = len(team_array)

for yr in range (2008, 2019):
    for i in range(tm_n):
        try:
            roster = Roster(team_array[i], yr)
            print(team_array[i], str(yr))
            for player in roster.players:
                player_array.append(player.player_id)
        except:
            print(team_array[i] + "does not exist " + str(yr))


# In[8]:


final_player_array = np.unique(player_array)
np.savetxt('player_array1', final_player_array, fmt='%5s', delimiter=',')


# In[89]:


player_df = pd.DataFrame()
n_players = len(final_player_array)

for j in range(1, n_players):
    try:
        player = Player(final_player_array[j]).dataframe
        index=[i[0] for i in player.index.values].index('Career')
        player_df = player_df.append(player.iloc[[index]])
    except:
        print("Player " + final_player_array[j] + " has no data")


# In[90]:


print(len(final_player_array))


# In[91]:


print(len(player_array))


# In[93]:


for col in player.columns:
    print(col)


# In[18]:


final_player_array)


# In[82]:


test_df=Player(final_player_array[6]).dataframe
index=[i[0] for i in test_df.index.values].index('Career')
test_df.iloc[[index]]


# In[86]:


player_df = pd.DataFrame()

for j in range(1, 7):
    player = Player(final_player_array[j]).dataframe
    index=[i[0] for i in player.index.values].index('Career')
    player_df = player_df.append(player.iloc[[index]])
    
player_df


# In[94]:


player_df.to_csv('NFL_player_df.csv')


# In[ ]:




