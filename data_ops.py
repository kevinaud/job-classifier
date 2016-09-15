# from abc import ABCMeta, abstractmethod, abstractproperty
import re
import csv
import csv_util
import pandas as pd
import numpy as np

class CsvJobData:
	def __init__(self, relPath):
		self._relPath = relPath
		self._fhandle = open(self.relPath)

	@property
	def relPath(self):
		return self._relPath

	@relPath.setter
	def relPath(self, newName):
		self._relPath = newName

	@property
	def fhandle(self):
		return self._fhandle

	def getDictReader(self):
		reader = csv.DictReader(self.fhandle)
		return reader
				
	def numJobs(self):
		numJobs = sum(1 for line in self.fhandle)
		# subtract 1 for column titles
		numJobs = numJobs - 1
		return numJobs

class DataSetSplitter:
	def __init__(self, initJobsPath, additJobsPath, size, testRatio=0.3):
		self.initJobsPath = initJobsPath
		self.additJobsPath = additJobsPath
		self.size = size
		self.testRatio = testRatio

		self.shuffleFiles()
		
		self.numInit = csv_util.countRows(self.initJobsPath) 
		self.numAddit = csv_util.countRows(self.additJobsPath) 
		self.numSamples = self.numInit + self.numAddit

	def shuffleFiles(self):
		csv_util.shuffleRows(self.initJobsPath)
		csv_util.shuffleRows(self.additJobsPath)

	def splitToFiles(self, trainFile, testFile):
		numTest = self.calcNumTest()
		numTrain = self.size - numTest

		additCsv =	CsvJobData(self.additJobsPath) 
		additReader = additCsv.getDictReader()
		dfTrain = self.getDfTrain(numTrain, additReader)
		dfTrain.to_csv(trainFile, index=False)

		additCsv =	CsvJobData(self.additJobsPath) 
		additReader = additCsv.getDictReader()
		dfTest = self.getDfTest(numTest, numTrain, additReader)
		dfTest.to_csv(testFile, index=False)

	def getDfTrain(self, numTrain, additReader):
		numAdditReq = numTrain - self.numInit 
		dfTrain = pd.read_csv(self.initJobsPath)
		i = 0;

		for line in additReader:
			if (i < numAdditReq):
				#line = additReader[i]
				dfLine = pd.DataFrame([[line['job_field_id'],line['title'], line['company'], line['description']]])
				dfLine.columns = ['job_field_id', 'title', 'company', 'description']
				dfTrain = dfTrain.append(dfLine, ignore_index=True)
				i = i + 1

		return dfTrain

	def getDfTest(self, numTest, numTrain, additReader):
		dfTest = pd.DataFrame(columns = ('job_field_id', 'title', 'company', 'description'))
		i = 0;
		beginIndex = int(numTrain - self.numInit)
		endIndex = int((numTest + numTrain) - self.numInit)
		for line in additReader:
			if(i >= beginIndex) and (i < endIndex):
				dfLine = pd.DataFrame([[line['job_field_id'],line['title'], line['company'], line['description']]])
				dfLine.columns = ['job_field_id', 'title', 'company', 'description']
				dfTest = dfTest.append(dfLine, ignore_index=True)
			i = i + 1
		
		return dfTest

	def calcNumTest(self):
		numTest = self.size * self.testRatio
		numTest = round(numTest)
		return numTest

def processJobTextDf(df, columnName):
	processedDf = df
	for index, row in processedDf.iterrows():
		row[columnName] = processJobText(row[columnName])
	return processedDf

def processJobText(jobText):
	text = jobText
	text = text.replace('\\',' ')
	text = text.replace('/',' ')
	text = text.replace('(',' ')
	text = text.replace(')',' ')
	text = text.replace('*',' ')
	text = re.sub('\.{2,}', '.', text) 
	text = re.sub('\s{2,}', ' ', text) 

	return text

def combineTextColumns(df, columnNames):
	newDf = pd.DataFrame()
	for index, row in df.iterrows():
		combinedText = ""
		for column in columnNames:
			combinedText = combinedText + str(df.loc[index, column])
			combinedText = combinedText + " " 
		dfLine = pd.DataFrame([[combinedText]])
		newDf = newDf.append(dfLine, ignore_index=True)

	newDf.columns = ['text']
	return newDf
	
























	
