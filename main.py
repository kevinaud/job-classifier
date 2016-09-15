import csv
from data_ops import CsvJobData, DataSetSplitter
import pandas as pd
from tokenize_data import tokenizer
from tokenize_data import tokenizer_porter
from nltk.corpus import stopwords

initFile = CsvJobData("data/initial_jobs.csv")
additFile = CsvJobData("data/additional_jobs.csv")

splitter = DataSetSplitter(initFile.relPath, additFile.relPath, 230)
splitter.splitToFiles("data/train230.csv", "data/test230.csv")

train230 = CsvJobData("data/train230.csv")
test230 = CsvJobData("data/test230.csv")

print('train230 numJobs: {}'.format(train230.numJobs()))
print('test230 numJobs: {}'.format(test230.numJobs()))

"""trainData = "data/train50.csv"
testData = "data/test50.csv"

# preprocessor = JobDataPreprocessor()
# preprocessor.processFile("data/train50.csv", "data/processed_train50.csv")

dfTrain = pd.read_csv(trainData)
dfTest = pd.read_csv(testData)

X_train = dfTrain[['title', 'company', 'description']]
y_train = dfTrain[['job_field_id']]

X_test = dfTest[['title', 'company', 'description']]
y_test = dfTest[['job_field_id']]

columns = ('title', 'company', 'description')

print()
X_train = data_ops.combineTextColumns(X_train, columns)
X_train = data_ops.processJobTextDf(X_train, 'text')
print("*****************Train******************")
print(X_train.head())
print(y_train.head())
print()

print()
X_test = data_ops.combineTextColumns(X_test, columns)
X_test = data_ops.processJobTextDf(X_test, 'text')
print("*****************Test******************")
print(X_test.head())
print(y_train.head())
print()"""















