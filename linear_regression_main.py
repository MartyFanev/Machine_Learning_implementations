import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mmodels.linear_regressor import linear_regressor
from mmodels.weighted_linear_regressor import weighted_linear_regressor


        
if __name__ == "__main__":
    model = linear_regressor()


    x_test = np.random.rand(100,1)*3
    y_test = 4+6*x_test + np.random.randn(100,1)

    plt.plot(x_test,y_test,"b.")

    model.learn(0.01,x_test,y_test,500)


    y_predict = model.predict(x_test)

    plt.plot(x_test,model.predict(x_test),"r-") 
    plt.savefig('scatterplot.png')

