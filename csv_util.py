import pandas as pd 
import numpy as np

def shuffleRows(csvFilePath):
	df = pd.read_csv(csvFilePath)
	np.random.seed(0)
	df = df.reindex(np.random.permutation(df.index))
	df.to_csv(csvFilePath, index=False)

def countRows(csvFilePath):
	fhandle = open(csvFilePath)
	count = (sum(1 for line in fhandle) - 1)
	fhandle.close()	
	return count
