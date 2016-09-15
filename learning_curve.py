from scipy.stats.distributions import t
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import pylab
import math
import csv

class Param:
	def __init__(self, key, value):
		self.key = key
		self.value = value
		self.lowerConf = value
		self.upperConf = value

class CurveData:
	def __init__(self, num_samples, accuracy):
		self.num_samples = num_samples
		self.accuracy = accuracy
		self

class LearningCurve:
	def __init__(self, a_init=0.05, b_init=2.0, c_init=-0.5):
		self.params = (Param('a', a_init), 
					   Param('b', b_init),
					   Param('c', c_init))
		self.data = []

	def model(self, X, a, b, c):
		y = (1 - a) - b * X ** c
		return y

	def fitParams(self):
		init_vals = [self.params[0].value,
				     self.params[1].value,
					 self.params[2].value]
		x = ([int(point.num_samples) for point in self.data])
		y = ([float(point.accuracy) for point in self.data])

		best_vals, covar = curve_fit(self.model, x, y, 
									 p0=init_vals, sigma=self.weights)

		tval = self.calcTval()
		self.setFittedParamVals(best_vals, tval, covar)

	def calcTval(self):
		alpha = 0.05
		numDataPoints = len(self.data)
		numParams = len(self.params)
		degreesOfFreedom = max(0, (numDataPoints - numParams))
		tval = t.ppf((1.0 - alpha)/2.0, degreesOfFreedom)
		return tval

	def setFittedParamVals(self, best_vals, tval, covar):
		for param, var, newVal in zip(self.params, np.diag(covar), best_vals):
			sigma = var**0.5
			param.value = newVal
			param.lowerConf = param.value - (sigma * tval)
			param.upperConf = param.value + (sigma * tval)

		if self.params[2].upperConf > 0:
			self.params[2].upperConf = -0.00001

	def upper(self, X):
		a = self.params[0].upperConf
		b = self.params[1].upperConf
		c = self.params[2].upperConf
		return self.model(X, a, b, c)

	def fitted(self, X):
		a = self.params[0].value
		b = self.params[1].value
		c = self.params[2].value
		return self.model(X, a, b, c)

	def lower(self, X):
		a = self.params[0].lowerConf
		b = self.params[1].lowerConf
		c = self.params[2].lowerConf
		return self.model(X, a, b, c)

	def loadCsvData(self, file_path):
		file_handle = open(file_path)
		reader = csv.DictReader(file_handle)
		for row in  reader:
			self.data.append(CurveData(row['num_samples'],
								  row['accuracy']))
		self.weights = ([((i + 1)/len(self.data)) for i in range(len(self.data))])
		samples = ([point.num_samples for point in self.data])

	def printData(self):
		for point in self.data:
			print('{} =>  {}'.format(point.num_samples, point.accuracy))

	def printParams(self):
		for param in self.params:
			print('{} | {} [{}, {}]'.format(param.key,
									   param.value, 
									   param.lowerConf, 
									   param.upperConf))

class LCurveGraph():
	def __init__(self, lCurve):
		self.lCurve = lCurve

	def plotPoints(self):
		data = self.lCurve.data
		sampleSizeList = ([point.num_samples for point in data])
		accuracyList = ([float(point.accuracy) for point in data])

		pylab.axis([0, 10000, 0, 1])
		pylab.ylabel('Classification Accuracy')
		pylab.xlabel('Size of Training Set')

		acc_point, = pylab.plot(sampleSizeList, accuracyList, 'bo')
		x = np.linspace(1, 10000, 10000)
		y = np.array([self.lCurve.fitted(val) for val in x])
		predicted, = pylab.plot(x, y, 'g')
		y = np.array([self.lCurve.upper(val) for val in x])
		conf, = pylab.plot(x, y, 'r')
		y = np.array([self.lCurve.lower(val) for val in x])
		pylab.plot(x, y, 'r')

		pylab.legend([acc_point, predicted, conf], ["Measured Accuracy","Predicted Learning Curve","95% Confidence Interval (upper and lower range)"])

		pylab.show()
		#plt.scatter(sampleSizeList, accuracyList)
		#plt.plot(
		#plt.xlabel('# of Samples')
		#plt.ylabel('Accuracy')
		#plt.axis([0, 1000, 0, 1])
		#plt.show()



























