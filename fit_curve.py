from lmfit import minimize, Parameters
from scipy.optimize import curve_fit

def learning_curve(numSamples, a, b, c):
	# a = params['a'].value
	# b = params['b'].value
	# c = params['c'].value

	model = (1 - a) - b * numSamples**c

	return model

"""params = Parameters()
params.add('a', value=0.05)
params.add('a', value=2)
params.add('a', value=-0.5, max=0, min=-1)"""

y = (0.26133333333333336,
0.33272727272727276,
0.3760000000000001,
0.3821052631578947,
0.4524444444444444,
0.44153846153846155,
0.4826666666666666,
0.4823529411764706,
0.49439999999999984,
0.49609756097560975,
0.5164444444444444,
0.5277551020408162,
0.5409523809523809,
0.5503571428571429,
0.5433333333333333,
0.5590625,
0.5662222222222223,
0.5752112676056338,
0.5917333333333333,
0.5954430379746835)

x=(50, 75, 100, 125, 150, 175, 200,
225, 250, 275, 300, 325, 350, 375,
400, 425, 450, 475, 500,
525)

init_vals=[0.05,2,-0.5]
best_vals, covar = curve_fit(learning_curve, x, y, p0=init_vals)
print(best_vals)






