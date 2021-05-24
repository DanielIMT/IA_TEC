
"""load DS into a data frame"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('IRIS.csv')
df.fillna(0, inplace=True)  # Fill with cero in case of empty data

"""USER SELECTS TWO FEATURES"""
select = int(input("Select features: \n"
                   "1.  Sepal L Vs Sepal W\n"
                   "2.  Sepal L Vs Petal L\n"
                   "3.  Sepal L Vs Petal W\n"
                   "4.  Sepal W Vs Sepal L\n"
                   "5.  Sepal W Vs Petal L\n"
                   "6.  Sepal W Vs Sepal W\n"
                   "7.  Petal L Vs Sepal L\n"
                   "8.  Petal L Vs Sepal W\n"
                   "9.  Petal L Vs Petal W\n"
                   "10. Petal W Vs Sepal L\n"
                   "11. Petal W Vs Sepal W\n"
                   "12. Petal W Vs Petal L\n"))
select_aux = np.array([[1,2],[1,3],[1,4],[2,1],[2,3],[2,4],[3,1],[3,2],[3,4],[4,1],[4,2],[4,3]])
global_features = ["sepal_length","sepal_width","petal_length","petal_width"]
features = [global_features[select_aux[select-1][0]-1],global_features[select_aux[select-1][1]-1]]
print(features)
df_X = df[features]
df_Y = df["species"]


"""SPLIT IN TRAINING AND TEST DATASETS"""
from sklearn.model_selection import train_test_split
dfX_train, dfX_test, dfY_train, dfY_test = train_test_split(df_X, df_Y,test_size=0.5)  # means 50 percent of DS goes to testing
dfX_train = dfX_train.to_numpy()
dfX_test = dfX_test.to_numpy()
dfY_train = dfY_train.to_numpy()
dfY_test = dfY_test.to_numpy()


"""MAKE UNIT VECTORS"""
def make_label(x):
    vec = np.zeros((len(x), 1))  # make a matrix of 3 lines, 1 colum and fill it with 0
    for i in range(len(x)):
        vec[i] = x[i]
    return vec  # vec is actually unit vectors

"""DATA IS READJUSTED FOR THE NEURAL NETWORK PROCESSING"""
def normalize_data(x):
    x = (x - x.mean(axis=0)) / x.std(axis=0)


"""APPLY FUNCTIONS TO DATA FRAMES"""

dfY_train = make_label(dfY_train)
dfY_test = make_label(dfY_test)
normalize_data(dfX_test)
normalize_data(dfX_train)


"""USEFUL FUNCTIONS"""
def convert_label(x):
    vec = np.zeros((3, 1))  # make a matrix of 3 lines, 1 colum and fill it with 0
    vec[int(x - 1)] = 1
    return vec  # vec is actually unit vectors


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def accuracy(dataX,labelsY):
    num_correct = 0
    for d, l in zip(dataX, labelsY):
        a = d.reshape(dataX.shape[1], 1)  #14x1
        y = convert_label(l)              #3x1
        for w, b in zip(weights, biases):
            a = sigmoid(np.dot(w, a) + b)
        prediction = np.rint(a)
        if np.array_equal(prediction, y):
            num_correct += 1
    return num_correct / datalen

def prove(First_axis, Second_axis):
        a = np.array([First_axis, Second_axis])
        a = a.reshape(dfX_train.shape[1], 1)
        for w, b in zip(weights, biases):
            a = sigmoid(np.dot(w, a) + b)
        prediction = np.rint(a)
        if (prediction[0] == 1):
            print("That is an IRIS-SETOSA")
        elif(prediction[1] == 1):
            print("That is an IRIS-VERSICOLOR")
        else:
            print("That is an IRIS-VIRGINICA")


def create_meshgrid(data):
    h = 0.02
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))
    return xx, yy, np.ones(xx.shape)

def plot_neural (weights,biases,grid,data,labels,epoch,accuracy, type):

    xx, yy, z = grid

    for i in range (xx.shape[0]): #row
        for j in range (yy.shape[1]): #colum
            a = np.array([xx[i][j], yy[i][j]]).reshape(2,1)
            for w, b in zip(weights, biases):
                a = sigmoid(np.dot(w,a)+b)
            score = np.argmax(a)
            z[i][j] = score
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    if (not type):
        plt.title("Accurracy at the epoch {} for training is --> {}%\n\nNETWORK SHAPE --> [2,{},{},3]]".format(epoch,accuracy,layer1,layer2))
    else:
        plt.title("Accurracy at testing after {} epochs is --> {}%".format(epoch, accuracy))
    ax.contourf(xx,yy,z, cmap = plt.cm.binary, alpha = 0.1, zorder = 15)
    ax.scatter(data[:,0], data[:,1], c=labels, s= 50, cmap = plt.cm.Set3, zorder = 50)
    ax.set_aspect('equal')
    ax.set_ylabel(features[1])
    ax.set_xlabel(features[0])
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

    """Color blue stand for Iris -SETOSA
       Color Green stands for  IRIS-VERSICOLOR 
       Color Yellow stands for IRIS-VIRGINICA"""


# IMPLEMENT NEURAL NETWORK


print("_________________________________________________")
epochs = int(input("How many epochs do you want to perform? -->"))
layer1 = int(input("How many neurons you want to use in hidden layer 1 -->"))
layer2 = int(input("How many neurons you want to use in hidden layer 2 -->"))
learning_rate = float(input("Learning rate? --> "))
print("_________________________________________________")

grid_test = create_meshgrid(dfX_test)
grid = create_meshgrid(dfX_train)
global_epoch = 0
step_size = 1
sizes = [dfX_train.shape[1], layer1, layer2, 3]
datalen = dfX_train.shape[0]  #To select a random number from 1 to that value
numb_layers = len(sizes)

biases = [np.random.randn(a, 1) for a in sizes[1:]]  # List of lists
weights = [np.random.randn(a, b) for a, b in zip(sizes[1:], sizes[:-1])]
g_w = [np.zeros(W.shape) for W in weights]
g_b = [np.zeros(b.shape) for b in biases]

for epoch in range(epochs):
    index = np.random.randint(datalen)
    a = dfX_train[index].reshape(dfX_train.shape[1], 1)
    y = convert_label(dfY_train[index])

    # Forward
    activations = [a]
    weighted_sums = []
    for w, b in zip(weights, biases):
        z = np.dot(w, a) + b
        a = sigmoid(z)
        activations.append(a)
        weighted_sums.append(z)
    # Cost
    c = np.sum((activations[-1] - y) ** 2)
    # Backprop
    gc = 2 * (activations[-1] - y)
    # Delta for output layer computes different
    delta = gc * activations[-1] * (1 - activations[-1])
    g_b[-1] = delta
    g_w[-1] = np.dot(delta, activations[-2].T)

    # Delta for hidden layers
    for i in range(2, len(sizes)):
        delta = activations[-i] * (1 - activations[-i]) * np.dot(weights[-i + 1].T, delta)
        g_b[-i] = delta
        g_w[-i] = np.dot(delta, activations[-i - 1].T)

    # Update values for g_w and g_b
    weights = [w - (gw * learning_rate) for w, gw in zip(weights, g_w)]
    biases  = [b - (gb * learning_rate) for b, gb in zip(biases, g_b)]

    if (epoch % (epochs/10) == 0):
        acc = np.round((accuracy(dfX_train,dfY_train))*100, 4)
        plot_neural(weights, biases, grid, dfX_train, dfY_train, epoch, acc, False)
        print("Accuracy for training @ epoch {} --> {} %".format(epoch,acc))
    global_epoch = epoch


print("_________________________________________________")
acc_test = np.round((accuracy(dfX_test,dfY_test))*100, 4)
plot_neural(weights, biases, grid, dfX_test, dfY_test, global_epoch, acc_test, True)
print("Accuracy for testing using trained model is --> {} %".format(acc_test))


print("_________________________________________________")
print("TRY IT")
while(True):
    x = float(input("Give me {}-->".format(features[0])))
    y = float(input("Give me {}-->".format(features[1])))
    prove(x,y)
    print("_________________________________________________")