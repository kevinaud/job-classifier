""" Outside Libraries """
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

""" My Libraries """
from tokenize_data import tokenizer
from tokenize_data import tokenizer_porter
from data_ops import CsvJobData, DataSetSplitter
import job_fields
import data_ops

""" Adjustable Options """
initFile = CsvJobData("data/initial_jobs.csv")
additFile = CsvJobData("data/additional_jobs.csv")
sampleSize = 230;
numIter = 10

""" Classification Algorithm Settings """
text_clf = Pipeline([('vect', CountVectorizer(max_df=0.75)),
					 ('tfidf', TfidfTransformer(norm='l1')),
					 ('clf', SGDClassifier(loss='hinge', 
										   penalty='l2',
										   alpha=1e-4,
										   n_iter=50,
										   random_state=42)),
])

avg_accuracy = 0.0
for iteration in range(numIter):
	""" Data Set Creation """
	trainData = "data/train_" + str(sampleSize) + "_" + str(iteration) + ".csv"
	testData = "data/test_" + str(sampleSize) + "_" + str(iteration) + ".csv"

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
	print('Iteration {} Accuracy: {}'.format(iteration, iter_accuracy))

avg_accuracy = avg_accuracy / numIter
print()
print("------------------------------------------------------------")
print()
print("Avg Accuracy: {}".format(avg_accuracy))














