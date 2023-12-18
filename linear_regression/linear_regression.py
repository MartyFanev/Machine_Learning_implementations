from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor



# create dummy data 
tau = 1

def calculateWeights(x_list,x_eval):

    '''
    Calculate the weights for a particular evaluated x

    arguments:
        -x_list: a list of x_values in the domain
        - x_eval: the value that we are tryign to evaluate
    '''

    weights = []
    for i in x_list:
        weights.append(np.exp(-(i-x_eval)**2/(2*tau**2)))
    weights = np.array(weights)
    return weights



x = np.random.rand(100,1)*10
y = 4 + 5*x**3 + np.random.rand(100,1)


weights = calculateWeights(x,6)
print(weights)

plt.scatter(x,weights,color='black',marker='.')


#x = np.array(x).reshape(-1,1)
#regressor = SGDRegressor(max_iter=1000, tol=1e-3,eta0=0.01,random_state=7)
#regressor.fit(x,y.ravel())
#plt.plot(x,regressor.predict(x))
#plt.scatter(x,y,color='black',marker='.')
plt.savefig('testing_linear_regression.png')

