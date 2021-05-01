'''
final project


i plan on taking a dataset of kobe bryants shots and training different machine learning models
we will be predicting if a shot is to go in or not based on the attributes of the data
'''

#libraries 
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn import preprocessing
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.patches as mpatches
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
'''
pre-processing
'''

#load in the data on to pandas 
data = pd.read_csv('data.csv')
#size 30697x25

#first we look for what columns have na values
#print(data.isna().any())
#we see that shot_made_flag contains NA's, since shot_made_flag is our classifier we will remove all rows that have NA.
data = data.dropna()
#size 25697x25

#next we look for any columns we can get rid of or combine
#print(data.head())

'''
game_id is a unique classifier for the game so that isn't needed
lat and long are not needed either due to loc_x and loc_y
we can potentially combine period/minutes_remaining/seconds_remaining
team_id and team_name can be dropped since kobe only played for one team 
we can potentially change game date to player age (days)
we can change match up to two values (@,vs) to indicate home game or away
we can drop shot_id since we do not care for a unique identifier for a shot
we might not need opponent so i'll remove it anyway since teams change often
'''

#lets begin dropping what we don't need
data = data.drop(['game_event_id','game_id','lat','lon','team_id','team_name','shot_id','opponent'],axis=1)

#lets change the matchup columns to only home and away and name the column game_type
data.loc[data['matchup'].str.contains('@'), 'matchup'] = 'away'
data.loc[data['matchup'].str.contains('vs'), 'matchup'] = 'home'
data = data.rename({'matchup':'game_type'},axis=1)

#lets find age in days for each game 
def age_at_date(x):
    x = datetime.strptime(x, "%Y-%m-%d").date()
    born = datetime.strptime("1978-08-23", "%Y-%m-%d").date()
    return (x-born).days
data['game_date'] = data['game_date'].apply(age_at_date)
data = data.rename({'game_date':'age_in_days'},axis=1)



#lets create seconds remaining in each period instead of minutes and seconds
data['seconds_remaining'] = (60 * data['minutes_remaining']) + data['seconds_remaining']
data = data.drop('minutes_remaining',axis=1)

#lets make a copy of our df
df = data

'''
create shot plots
plt.figure(figsize=(10,10))
plt.ylim(-30,400)
plt.xticks([])
plt.yticks([])
colors = np.where(df['shot_made_flag']==1, "#552583","#fdb927")
plt.scatter(df.loc_x,df.loc_y,color=colors,s=4,alpha=0.5)
pop_a = mpatches.Patch(color='#552583', label='Make')
pop_b = mpatches.Patch(color='#fdb927', label='Miss')
plt.legend(handles=[pop_a,pop_b])
plt.title('Kobe Bryant Career Shots')
plt.show()
'''

#begin splitting

label = np.array(df['shot_made_flag']).astype(int)
data = df.drop('shot_made_flag',1)

data_train, data_test, label_train, label_test = train_test_split(data,label,test_size=0.2)

#standardize x and y
A = data_train[['loc_x','loc_y']]
scaler = preprocessing.StandardScaler().fit(A)
A_st = scaler.transform(A)
data_train['loc_x'] =  A_st[:,0]
data_train['loc_y'] = A_st[:,1]

B = data_test[['loc_x','loc_y']]
B_st = scaler.transform(B)
data_test['loc_x'] =  B_st[:,0]
data_test['loc_y'] = B_st[:,1]


#make dummy variables for categories
data_train = pd.get_dummies(data_train)
data_test = pd.get_dummies(data_test)

#make sure both dataframes have the same dummy columns
unique_train = np.setdiff1d(data_train.columns,data_test.columns)
unique_test = np.setdiff1d(data_test.columns,data_train.columns)
for i in unique_train:
    data_test[i] = 0
data_test = data_test[data_train.columns]

#create the algorithm plots
def algorithmplots(data_test,data_train,label_test,label_train):
    r = np.arange(1,5000,500)
    dt = []
    dtt = []
    perceptron = []
    perceptront = []
    sdgc = []
    sdgct = []
    svc = []
    svct = []
    rf = []
    rft = []
    for i in r :
        #decision tree
        start = timer()
        model = tree.DecisionTreeClassifier(min_samples_leaf=i,class_weight='balanced')
        model.fit(data_train, label_train)
        end = timer()
        dtt.append(end - start)
        dt.append(model.score(data_test,label_test))
        #perceptron
        start = timer()
        model2 = Perceptron(max_iter=i,eta0=1)
        model2.fit(data_train, label_train)
        end = timer()
        perceptront.append(end - start)
        prediction_test = model2.predict(data_test)
        perceptron.append(accuracy_score(prediction_test,label_test))
        #sdgc
        start = timer()
        model3 = linear_model.SGDClassifier(max_iter=i)
        model3.fit(data_train, label_train)
        end = timer()
        sdgct.append(end - start)
        prediction_test2 = model3.predict(data_test)
        sdgc.append(accuracy_score(prediction_test2,label_test))
        #svc
        start = timer()
        model4 = LinearSVC(dual=False,max_iter=i)
        model4.fit(data_train, label_train)
        end = timer()
        svct.append(end - start)
        prediction_test3 = model4.predict(data_test)
        svc.append(accuracy_score(prediction_test3,label_test))
        #random forest
        start = timer()
        model5 = RandomForestClassifier(n_estimators = i)
        model5.fit(data_train, label_train)
        end = timer()
        rft.append(end - start)
        prediction_test4 = model5.predict(data_test)
        rf.append(accuracy_score(prediction_test4,label_test))
    plt.figure()
    plt.plot(r,dt,label='decision tree')
    plt.plot(r,perceptron,label='perceptron')
    plt.plot(r,sdgc,label='sdgc')
    plt.plot(r,svc,label='svc')
    plt.plot(r,rf,label='random forest')
    plt.xlim(0,5000)
    plt.ylim(0.5,0.7)
    plt.xlabel("Number of iterations")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Acurracy of Different Classification Algorithms')
    plt.show()
    '''
    Time complexity and max accuracy and index
    plt.figure()
    plt.plot(r,dtt,label='decision tree')
    plt.plot(r,perceptront,label='perceptron')
    plt.plot(r,sdgct,label='sdgc')
    plt.plot(r,svct,label='svc')
    plt.plot(r,rft,label='random forest')
    plt.xlabel("Number of iterations")
    plt.ylabel("Number of seconds")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Time complexity of Different Classification Algorithms')
    plt.show()
    plt.figure()
    plt.plot(r,dtt,label='decision tree')
    plt.plot(r,perceptront,label='perceptron')
    plt.plot(r,sdgct,label='sdgc')
    plt.plot(r,svct,label='svc')
    plt.xlabel("Number of iterations")
    plt.ylabel("Number of seconds")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Time Complexity of Different Classification Algorithms')
    plt.show()
    print("The highest prediction accuracy for perceptron was " + str(max(perceptron)) + " at " + str(r[perceptron.index(max(perceptron))]) )
    print("The highest prediction accuracy for decision tree was " + str(max(dt)) + " at " + str(r[dt.index(max(dt))]))
    print("The highest prediction accuracy for sdgc was " + str(max(sdgc)) + " at " + str(r[sdgc.index(max(sdgc))]))
    print("The highest prediction accuracy for svc was " + str(max(svc)) + " at " + str(r[svc.index(max(svc))]))
    print("The highest prediction accuracy for rf was " + str(max(rf)) + " at " + str(r[rf.index(max(rf))]))
    '''

#run for results
algorithmplots(data_test,data_train,label_test,label_train)
