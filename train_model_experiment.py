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
import job_fields
import data_ops

""" Data Preparation """
trainData = "data/train230.csv"
testData = "data/test230.csv"

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
stop = stopwords.words('english')

text_clf = Pipeline([('vect', CountVectorizer(max_df=0.75)),
					 ('tfidf', TfidfTransformer(norm='l1')),
					 ('clf', SGDClassifier(loss='hinge', 
						 				   penalty='l2',
						 				   alpha=1e-4,
										   n_iter=50,
										   random_state=42)),
])

"""param_grid = {	 'vect__max_df': (0.5, 0.75, 1.0),
				 'vect__ngram_range': ((1, 1), (1, 2), (1, 3), (1, 4)),
				 'vect__stop_words': (stop, None),
				 'tfidf__use_idf': (True, False),
				 'tfidf__norm': ('l1', 'l1'),
				 'clf__alpha': (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7),
				 'clf__penalty': ('l2', 'elasticnet'),
				 'clf__n_iter':(10, 50, 80)}

gs_clf = GridSearchCV(text_clf, param_grid, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, y_train)

best_params, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
for param_name in sorted(param_grid.keys()):
	print("{}: {}".format(param_name, best_params[param_name]))

print()
print("score: {}".format(score))"""

text_clf = text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
print(metrics.classification_report(y_test, 
									predicted,
									target_names=job_fields.names))
print()
print(metrics.confusion_matrix(y_test, predicted))

print()
print()
print()
print("accuracy: {}".format(np.mean(predicted == y_test)))
# print(type(X_test))
"""for i in range(X_test.size):
	print()
	print("---------------------------------------------------------------------")
	print()
	print('TEXT: {}'.format(X_test[i][:60]))
	print('      {}'.format(X_test[i][60:120]))
	print('ACTUAL: {}'.format(job_fields.names[y_test[i]]))
	print('PREDICTED: {}'.format(job_fields.names[predicted[i]]))
"""















