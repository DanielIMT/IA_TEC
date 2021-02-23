import numpy as np
import statistics as st
import matplotlib.pyplot as plt



class CreateDS:
    def __init__(self, numero):
        self.n = numero
        self.Jcost = list()
        self.datosX = [0 for i in range(numero)]
        self.datosY = [0 for i in range(numero)]
        self.datosX_SD = [0 for i in range(numero)]
        self.datosY_SD = [0 for i in range(numero)]
        self.m_current = self.b_current = 0
        self.iteration = 0
        self.learning_rate = 0

    def instances(self):
        for i in range(self.n):
            self.datosX[i] = float(input("Input Data X" + str(i + 1) + " = "))
            self.datosY[i] = float(input("Input Data Y" + str(i + 1) + " = "))

        for i in range(self.n):
            self.datosX_SD[i] = (self.datosX[i] - (sum(self.datosX) / self.n)) / st.stdev(self.datosX)
            self.datosY_SD[i] = (self.datosY[i] - (sum(self.datosY) / self.n)) / st.stdev(self.datosY)

    def iterateData(self):
        self.iteration = int(input("How many iterations you wanna perform? -->"))
        self.learning_rate = float(input("Learning rate? -->"))

    def gradient_descent(self, option):
        if option == 1:
            self.m_current = self.b_current =0
            self.Jcost = list()

        x_sd = np.array(self.datosX_SD)
        y_sd = np.array(self.datosY_SD)
        m = list()
        b = list()

        for i in range(self.iteration):
            y_predicted = self.m_current * x_sd + self.b_current
            m_gradient = -(2 / self.n) * sum(self.datosX_SD * (y_sd - y_predicted))  # Sum all values of the list
            b_gradient = -(2 / self.n) * sum(y_sd - y_predicted)
            self.Jcost.append((1 / self.n) * sum([val ** 2 for val in (y_sd - y_predicted)]))
            self.m_current = self.m_current - self.learning_rate * m_gradient
            self.b_current = self.b_current - self.learning_rate * b_gradient
            print("m {}, b {}, iteration {}, cost {}".format(self.m_current, self.b_current, i, self.Jcost[i]))
            m.append(self.m_current)
            b.append(self.b_current)
        return m, b

        # print("m {}, b {}, iteration {}, cost {}".format(self.m_current, self.b_current, i, cost[i]))


i = 0
while (True):
    if (i == 0):
        ejemplo = CreateDS(int(input("How many instances? ---> ")))
        ejemplo.instances()

    ejemplo.iterateData()
    m, b = ejemplo.gradient_descent(i)


    """PLOTING"""
    plt.plot(m, ejemplo.Jcost, color="BLUE", label="m")
    plt.plot(b, ejemplo.Jcost, color="RED", label="b")
    plt.xlabel('PARAMETERS', fontsize=20)
    plt.ylabel('COST', fontsize=20)
    plt.title("Gradient Descent")
    plt.legend()

    fig = plt.figure()
    a = fig.add_subplot()
    a.scatter(ejemplo.datosX_SD, ejemplo.datosY_SD, c='r', marker='o')
    fx = lambda x: ejemplo.m_current * x + ejemplo.b_current
    x = np.linspace(-10, 10, 20)
    linearReg = fx(x)
    a.plot(x, linearReg, color="Green")
    a.autoscale(enable=True, axis=u'both', tight=False)
    plt.xlabel('X', fontsize=20)
    plt.ylabel('Y', fontsize=20)
    plt.title("Linear Regresion")

    plt.show()

    print("\n1.Change # of iterations or learning rate \n2.Other data \n3. exit")
    opcion = int(input())
    if (opcion == 1):
        i = 1
    elif (opcion == 2):
        i = 0
    else:
        break
