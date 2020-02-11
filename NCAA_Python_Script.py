
# coding: utf-8

# In[29]:


from sportsreference.ncaaf.teams import Teams
from sportsreference.ncaaf.roster import Roster
from sportsreference.ncaaf.roster import Player
import numpy as np
import pandas as pd


# In[2]:


team_array = []

teams = Teams()
for team in teams:
    team_array.append(team.abbreviation)


# In[3]:


print(team_array)


# In[16]:


player_vector = []
tm_n = len(team_array)

for yr in range(2007, 2018):
    for i in range(tm_n):
        try:
            roster = Roster(team_array[i], yr)
            for player in roster.players:
                player_vector.append(player.player_id)
        except:
            print(team_array[i] + " " + str(yr))


# In[19]:


final_player_vector = np.unique(player_vector)
np.savetxt('player_array', final_player_vector, fmt='%5s', delimiter=',')


# In[63]:


player_df = pd.DataFrame()
n_players = len(final_player_vector)

for j in range(1, n_players):
    try:
        player = Player(final_player_vector[j]).dataframe
        player_df = player_df.append(player.loc[player['year']==''])
    except:
        print("Player " + final_player_vector[j] + " has no data")


# In[27]:


for col in player.columns:
    print(col)


# In[37]:


player_df = pd.DataFrame()
X = {'Name': ['Cody']}
Y = pd.DataFrame(X, columns=['Name'])


# In[38]:


player_df.append(Y)


# In[53]:


print(len(final_player_vector))


# In[62]:


print(Player('aamir-holmes-1').dataframe)


# In[74]:


player_df.to_csv('NCAA_player_df.csv')


# # 
