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


class weighted_regressor:
    def __init__(self):
        self.theta = np.random.rand(2).reshape((-1,1))

    def predict(self,X):
        """
        predict a single data point

        """
        theta = self.theta.reshape((1,-1))
        X = np.c_[np.ones((100,1)),X]
        result = np.dot(theta,np.transpose(X)).reshape((-1,1))


        return result

    def get_weights(self,x,x_test,tau):
        weights = []
        # x_test needs ot be a vector
        for i in x_test:
            result = np.exp(-((i-x)**2)/(2*tau**2))
            weights.append(result)

        return np.array(weights)





    def learn(self,lr,X,x,Y,epochs):
        weights = self.get_weights(x,X,0.5)

        x = np.c_[np.ones((100,1)),X]

        while epochs > 0:
            random_number = np.random.randint(0,np.shape(x)[0])
            x_stochastic,y_stochastic,weights_stochastic = x[random_number],Y[random_number],weights[random_number]

            theta_new = self.theta.reshape((-1,1)) - weights_stochastic*lr*(np.dot( self.theta.reshape((1,-1)),x_stochastic) - y_stochastic )*x_stochastic.reshape((-1,1)) #gradient descent update 
            self.theta = theta_new

            epochs = epochs - 1

            print(f"theta parameter is : {self.theta}, for epoch : {epochs}")

        

        
if __name__ == "__main__":
    model = linear_regressor()
    w_model = weighted_regressor()


    x_test = np.random.rand(100,1)*3
    y_test = 4+6*x_test + np.random.randn(100,1)


    plt.plot(x_test,y_test,"b.")

    model.learn(0.01,x_test,y_test,500)

    w_model.learn(0.01,x_test,1.5,y_test,500)

    y_predict_w = w_model.predict(x_test)
    y_predict = model.predict(x_test)

    plt.plot(x_test,model.predict(x_test),"r-") 
    plt.plot(x_test,w_model.predict(x_test),"g.")
    plt.savefig('scatterplot.png')

