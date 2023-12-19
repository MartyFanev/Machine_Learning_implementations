import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class linear_regressor:
    def __init__(self):
        self.theta = np.random.rand(2).reshape((-1,1))
        print(self.theta)


    def predict(self,X):
        """
        predict a single data point

        """
        theta = self.theta.reshape((1,-1))
        X = np.c_[np.ones((100,1)),X]
        result = np.dot(theta,np.transpose(X)).reshape((-1,1))


        return result


    def learn(self,lr,X,Y,epochs):

        x = np.c_[np.ones((100,1)),X]

        while epochs > 0:
            random_number = np.random.randint(0,np.shape(x)[0])
            x_stochastic,y_stochastic = x[random_number],Y[random_number]

            theta_new = self.theta.reshape((-1,1)) - lr*(np.dot( self.theta.reshape((1,-1)),x_stochastic) - y_stochastic )*x_stochastic.reshape((-1,1)) #gradient descent update 
            self.theta = theta_new

            epochs = epochs - 1

            print(f"theta parameter is : {self.theta}, for epoch : {epochs}")

