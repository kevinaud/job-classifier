""" Outside Libraries """
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyprind

""" My Libraries """
from tokenize_data import tokenizer
from tokenize_data import tokenizer_porter
from data_ops import CsvJobData, DataSetSplitter
import job_fields
import data_ops

""" Adjustable Options """
initFile = CsvJobData("data/initial_jobs.csv")
additFile = CsvJobData("data/additional_jobs.csv")
numIter = 25 

""" Classification Algorithm Settings """
text_clf = Pipeline([('vect', CountVectorizer(max_df=0.75)),
					 ('tfidf', TfidfTransformer(norm='l1')),
					 ('clf', SGDClassifier(loss='hinge', 
										   penalty='l2',
										   alpha=1e-4,
										   n_iter=50,
										   random_state=42)),
])

n = 500 
bar_prog = pyprind.ProgBar(n, stream=1)
accuracyList = []
sampleSizeList = []

for sampleSize in range(50, 550, 25):
	avg_accuracy = 0.0
	for iteration in range(numIter):
		""" Data Set Creation """
		trainData = "data/size_" + str(sampleSize) + "/train_" + str(iteration) + ".csv"
		testData = "data/size_" + str(sampleSize) + "/test_" + str(iteration) + ".csv"

		splitter = DataSetSplitter(initFile.relPath, additFile.relPath, sampleSize)
		splitter.splitToFiles(trainData, testData) 

		""" Data Preparation """
		dfTrain = pd.read_csv(trainData)
		dfTest = pd.read_csv(testData)

		X_train = dfTrain[['title', 'company', 'description']]
		y_train = dfTrain[['job_field_id']]
		X_test = dfTest[['title', 'company', 'description']]
		y_test = dfTest[['job_field_id']]

		columns = ('title', 'company', 'description')

		X_train = data_ops.combineTextColumns(X_train, columns)
		X_train = data_ops.processJobTextDf(X_train, 'text')

		X_test = data_ops.combineTextColumns(X_test, columns)
		X_test = data_ops.processJobTextDf(X_test, 'text')

		X_train = X_train.loc[:, 'text'].values
		y_train = y_train.loc[:, 'job_field_id'].values
		job_fields.shiftIdsDown(y_train)

		X_test = X_test.loc[:, 'text'].values
		y_test = y_test.loc[:, 'job_field_id'].values
		job_fields.shiftIdsDown(y_test)

		""" Model Training """
		text_clf = text_clf.fit(X_train, y_train)

		""" Model Evaluation """
		predicted = text_clf.predict(X_test)
		iter_accuracy = np.mean(predicted == y_test)
		avg_accuracy = avg_accuracy + iter_accuracy
		bar_prog.update()


	avg_accuracy = avg_accuracy / numIter
	sampleSizeList.append(sampleSize)
	accuracyList.append(avg_accuracy)
	
print()
for i in range(20):
	print("{} => {}".format(sampleSizeList[i], accuracyList[i]))
print()

plt.scatter(sampleSizeList, accuracyList)
plt.xlabel('# of Samples')
plt.ylabel('Accuracy')
plt.axis([0, 1000, 0, 1])
plt.show()












