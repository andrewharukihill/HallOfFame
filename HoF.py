#Hall of Fame Prediction Model for Players that started at least in the 1960's


#Deficiencies:
#Does not properly weight college performance
#Does not properly weight defensive teams nor DPOY


def fitForestModel(df):
	"""Fits RandomForestClassifier to data"""
	X_names = ['yearsPlayed', 'WS', 'percentPlayoffApp', 'numAllStar', 'numChampionships', 
	'numMVP', 'numAllNBA', 'isROY', 'numPlayoffApp', 'numFinalApp', 'peakWS', 'totalPTS',
	 'isCollegeAllAmerican']

	y_name = 'isHoF'

	X = df[X_names]

	y = df[y_name]

	#testForest(X, y)

	from sklearn.ensemble import RandomForestClassifier

	clf = RandomForestClassifier(max_depth=5, random_state=42, min_samples_leaf=3)
	clf.fit(X,y)
	print('\nFeature Importances:')
	for i in range(len(X_names)):
		print(X_names[i] + ': ' + str(clf.feature_importances_[i]))

	return clf

def testForest(X, y):
	"""Tests the optimal max depth of Forest"""

	from sklearn.ensemble import RandomForestClassifier
	from sklearn.model_selection import train_test_split
	import numpy as np 
	totalAvg = []
	for i in range(1,10):
		avgScoreList = []
		for j in range(1,20):
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
										random_state=j, stratify=y)
			clf = RandomForestClassifier(criterion='gini', max_depth=i, random_state=j, min_samples_leaf=3)
			clf.fit(X_train, y_train)
			avgScoreList.append(clf.score(X_test, y_test))
		print('Max Depth ' + str(i) + ' Avg. Score: '  + str(np.mean(avgScoreList)))
		totalAvg.append(np.mean(avgScoreList))
	print('Total Average = ' + str(np.mean(totalAvg)))


def fitTreeModel(df):
	"""Fits DecisionTreeClassifier to data"""

	X_names = ['yearsPlayed', 'numPTSLeader',
	'numAllStar', 'numChampionships', 'numMVP', 'numAllNBA', 'isROY', 'numFinalApp',
	 'numOlympicApp', 'isOlympicGold', 'isCollegeChampion', 'isCollegeAllAmerican']

	y_name = 'isHoF'

	dfClean = removeOutliers(df)

	X = dfClean[X_names]

	y = dfClean[y_name]

	testTree(X, y)

	from sklearn.tree import DecisionTreeClassifier
	from pydotplus import graph_from_dot_data
	from sklearn.tree import export_graphviz
	
	maxDepthList = [3,4,5,6]

	if(True):
		for i in range(len(maxDepthList)):
			tree = DecisionTreeClassifier(criterion='gini', max_depth=maxDepthList[i], random_state=1, min_samples_leaf=5)
			tree.fit(X, y)
			dot_data = export_graphviz(tree, filled=True, rounded=True, 
									class_names= ['0', '1'],
									feature_names=X_names,
									out_file=None)
			graph = graph_from_dot_data(dot_data)
			fileName = 'HoFTree(Outlier)Depth' + str(maxDepthList[i]) + '.png'
			graph.write_png(fileName)

def testTree(X, y):
	"""Tests the optimal max depth of tree"""

	from sklearn.tree import DecisionTreeClassifier
	from sklearn.model_selection import train_test_split
	import numpy as np 
	totalAvg = []
	for i in range(3,7):
		avgScoreList = []
		for j in range(1,20):
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
										random_state=j, stratify=y)
			tree = DecisionTreeClassifier(criterion='gini', max_depth=i, random_state=j, min_samples_leaf=3)
			tree.fit(X_train, y_train)
			avgScoreList.append(tree.score(X_test, y_test))
		print('Max Depth ' + str(i) + ' Avg. Score: '  + str(np.mean(avgScoreList)))
		totalAvg.append(np.mean(avgScoreList))
	print('Total Average = ' + str(np.mean(totalAvg)))


def removeOutliers(df):
	"""removes Outliers and returns a clean dataframe"""

	cleanDf = df

	if (True):
		#K.C. Jones 
		#	-Zero All-Stars, but won 8 championships w/ BOS in '60s
		#	-Won 3 championships as Assistant Coach and Head Coach so inclusion may be
		#		based on post-playing career
		cleanDf = cleanDf[cleanDf['Name'] != 'K.C. Jones']

		#Frank Ramsey
		#	-Zero All-Stars, but won 7 championships w/ BOS in '60s
		cleanDf = cleanDf[cleanDf['Name'] != 'Frank Ramsey']

		#Calvin Murphy
		#	-One All-Star
		#	-Inclusion may be because of his college production
		#	-Two Finals App.
		cleanDf = cleanDf[cleanDf['Name'] != 'Calvin Murphy']

		#Bill Bradley
		#	-One All-Star
		#	-Inclusion may be because of his college production
		#	-Two Championships
		cleanDf = cleanDf[cleanDf['Name'] != 'Bill Bradley']

		#Ralph Sampson
		#	-Short Career
		#	-Four All-Stars
		#	-College production
		cleanDf = cleanDf[cleanDf['Name'] != 'Ralph Sampson']

		#Guy Rodgers
		#	-Unremarkable Career
		#	-Four All-Stars
		cleanDf = cleanDf[cleanDf['Name'] != 'Guy Rodgers']

	cleanDf = cleanDf[cleanDf['startYear'] >= 1960]
	#cleanDf = cleanDf[cleanDf['playedABA'] == 0]

	return cleanDf

def makeFeatures(df):
	"""creates features"""
	import numpy as np

	#Position dummies
	df = pd.concat([df, pd.get_dummies(df.Position)], axis=1)

	#Big/Small
	df['Big'] = df['Position'].apply(lambda x: 1 if (x == 'C') | (x == 'PF') else 0)

	#AST/REB Leader
	df['numASTREBLeader'] = df['numASTLeader'] + df['numREBLeader']

	#PercentPlayoffApp
	df['percentPlayoffApp'] = df['numPlayoffApp']/df['yearsPlayed']


	return df

import pandas as pd 


df = pd.read_csv('HoF.csv')

df = makeFeatures(df)

#strInput = str(input('Tree or Forest: '))
strInput='Tree'
while((strInput != 'Tree') & (strInput != 'Forest')):
	strInput = str(input('Tree or Forest: '))
if (strInput == 'Tree'):
	fitTreeModel(df)
else:
	clf = fitForestModel(df)





