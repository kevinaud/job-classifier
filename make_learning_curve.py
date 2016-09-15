from scipy.stats.distributions import t
from learning_curve import LearningCurve
from learning_curve import LCurveGraph
from learning_curve import Param
import csv

file_path = "learn_data.csv"

l_curve = LearningCurve()
l_curve.loadCsvData(file_path)
l_curve.fitParams()
l_curve.printParams()

graph = LCurveGraph(l_curve)
graph.plotPoints()

"""file_handle = open(file_path)
reader = csv.DictReader(file_handle)
data = []

for row in reader:
	data.append(LearnData(row['num_samples'], row['accuracy']))

a = Param('a', 0.05)
b = Param('b', 2.0)
c = Param('c', -0.5)

alpha = 0.05
numPoints = len(data)
numParams = 3
degsFreedom = max(0, numPoints - numParams)"""
