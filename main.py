import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import deque


def read_input(df):
    data = {}
    for item in df.columns:
        data[item] = df[item].to_numpy()
    return data

class LinearRegression:
    """
    Gradient descent using linear regression. It uses barzilai-borwein accelleration method and armijo rule
    to compute the length of the step.
    """
    def __init__(self, train, tol=None, sigma=None, beta=None, lam=None, s=None, ro_min=None, ro_max=None, m=1, sg=False):
        self.sg = sg
        self.train = train
        # m is number of samples
        self.loss_history = []
        self.m = m
        self.M = deque(maxlen=self.m)
        self.Y = data['labels']  # 1xm
        self.samples = self.Y.shape[0]
        self.lam = random.random() if not lam else lam
        self.dim = len(data)-1  # dimension of samples
        self.X = np.column_stack((data['x1'], data['x2']))  # mxn
        # xk and xk-1 are the points on the x axis of the loss function

        # which are the weights and the bias of our model!

        # initialize weights
        self.w = np.random.rand(self.dim, 1)  # nx1
        self.w_pre = np.random.rand(self.dim, 1)  # nx1
        # self.w = np.zeros((self.dim, 1))
        # self.w_pre = np.zeros((self.dim, 1))

        # initialize bias
        self.c = random.random()
        self.c_pre = random.random()

        self.beta = random.random() if not beta else beta
        self.sigma = random.random() if not sigma else sigma
        self.tol = random.random() if not tol else tol
        self.ro_min = random.random() if not ro_min else ro_min
        self.ro_max = random.random() if not ro_max else ro_max

        self.grad = np.random.rand(self.dim + 1, 1)  # (n+1)x1
        self.s = random.random() if not s else s
        # the following two points are points on the loss dataset
        # the loss is a function of the parameters, which are n+1

    def __str__(self):
        print("Labels : {}".format(str(self.Y)))
        print("Number of samples : {}".format(str(self.samples)))
        print("Data :\n {}".format(str(self.X)))
        print("--------PARAMETERS-----------")
        print("Weights : {}".format(str(self.w)))
        print("Bias : {}".format(str(self.c)))
        print("Beta : {}".format(str(self.beta)))
        print("Sigma : {}".format(str(self.sigma)))
        print("Tol : {}".format(str(self.tol)))
        print("Lambda : {}".format(str(self.lam)))
        print("Grad : \n{}".format(str(self.grad)))
        return "-------------"

    def predict(self, x, w):
        """
        linear classifier, f(x) = w^t x + c
        input: sample x, weights w

        output: scalar, which is the prediction
        Need to specify the weights because in linear regression
        we need to make predictions using old weights
        """
        result = (w[:-1].T @ x).item() + w[-1].item()
        return result

    def loss(self, w):
        """
        :input: weights (note that w needs to have the bias as the last element)
        :return: logistic loss value
        Computes the logistic loss using the mean on the whole dataset.
        """
        sum = 0
        for i, sample in enumerate(self.X):
            sum += np.log(1+np.exp(-self.Y[i]*(self.predict(sample, w))))
        normalized_summation = sum / self.samples
        # w alredy contains c as the last element
        compl_factor = (self.lam/2) * (w.T @ w).item()
        return normalized_summation + compl_factor

    def gradient(self, w, c):
        
        """
        :input: weights w, bias c
        :return: column vector, partial derivative for each dimension, bias included
        computes the gradient of the loss function with respect to the given w and bias
        Note that w and the bias are separate, they don't have to be contained
        in the same vector.
        """
        
        grad_F = []  # full gradient matrix
        for i in range(self.samples):
            common_numerator = np.exp(-self.Y[i]*(self.predict(self.X[i], np.vstack((w,c)))))
            common_denominator = 1 + common_numerator
            common_numerator *= -self.Y[i]
            # can be done with just 1 for (iterating over samples), can be improved
            grad_f = [(common_numerator * self.X[i][j])/common_denominator if j < self.dim else common_numerator/common_denominator for j in range(self.dim+1)]
            grad_F.append(grad_f.copy())
            grad_f.clear()
        grad_F = np.array(grad_F).T
        # add regularization term
        if self.stoc:
            random_column_index = np.random.choice(grad_F.shape[1])
            final_gradient = grad_F[:, random_column_index].reshape(-1, 1)
        else:
            final_gradient = np.mean(grad_F, axis=1, keepdims=True)
        # return mean
        return final_gradient + self.lam * np.vstack((w, c))

    def plot_loss(self):
        dict = {'loss': self.loss_history}
        df = pd.DataFrame(data=dict)
        # df.to_csv('loss.csv')
        plt.plot(df.index, df.values)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Logistic Loss")
        plt.show()

    def plot_classifier(self):
        ub = max(df['x1'].max(), df['x2'].max())
        w = self.w
        c = self.c
        
        dens = 200  # You can adjust this value as needed

        # Generate x and y values
        x_values = np.linspace(0, ub, dens)
        y_values = np.linspace(0, ub, dens)

        # Create a grid of x and y values using meshgrid
        x, y = np.meshgrid(x_values, y_values)

        # Reshape the x and y values to create a (n*n, 2) array
        points = np.column_stack((x.ravel(), y.ravel()))
        self.grid = points
        self.boundary = ['red' if np.sign(self.predict(p, np.vstack((w,c)))) > 0 else 'blue' for p in points]

        rx = []
        ry = []
        bx = []
        by = []

        for i, p in enumerate(self.grid):
            if self.boundary[i] == 'red':
                rx.append(p[0])
                ry.append(p[1])
            else:
                bx.append(p[0])
                by.append(p[1])
        plt.scatter(rx, ry, color="red")
        plt.scatter(bx, by, color="blue")
        plt.scatter(df['x1'], df['x2'], marker="^", color="yellow")
        plt.show()

    def sec_checks(self, settings):
        new_vars = settings
        if settings['stoc'] and (settings['arm'] or settings['bb']):
            print("Warning: you enabled bot stoc gradient and full gradient options.")
            print("Disabling full gradient and relative options for security purposes.")
            time.sleep(2)
            new_vars['arm'] = False
            new_vars['bb'] = False
        return new_vars
            
        
    # def fit(self, arm=False, bb=False, stoc=False):
    def fit(self, settings):
        
        """
        with stepest descent and without armijo it converges with around 1400 iterations,
        with barziali-borwein it converges with 700 iterations.
        """
        cleaned_vars = self.sec_checks(settings)
        arm = cleaned_vars['arm']
        bb = cleaned_vars['bb']
        stoc = cleaned_vars['stoc']
        self.stoc = stoc
        self.grad = self.gradient(self.w, self.c)
        k = 0
        while(np.linalg.norm(self.grad)) >= self.tol:  # termination rule
            k += 1
            print(k)
            self.grad = self.gradient(self.w, self.c)
            # print("gradient\n {}\n w \n {}\n c:{}".format(self.grad,self.w,self.c))
            s_current = np.vstack((self.w, self.c)) - np.vstack((self.w_pre, self.c_pre))  # col
            y_current = self.grad - self.gradient(self.w_pre, self.c_pre)  # col
            ro_current = (s_current.T @ s_current).item() / (s_current.T @ y_current).item()
            ro_current = np.maximum(self.ro_min, np.minimum(ro_current, self.ro_max))
            if bb:
                d_current = -ro_current * self.grad  # column
            else:
                d_current = - self.grad
            alpha_current = self.s
            print("----------------")
            self.M.append(self.loss(np.vstack((self.w, self.c))).item())
            # armijo rule
            if arm:
                while self.loss(np.vstack((self.w, self.c)) + (alpha_current * d_current)).item() > max(self.M) + self.sigma*alpha_current*(self.grad.T @ d_current).item():
                    alpha_current *= self.beta
            # if stoc and not arm:
            #     alpha_current = 1/k
            w_next = self.w + (alpha_current * d_current[:-1])
            c_next = self.c + (alpha_current * d_current[-1].item())
            # current values become old values
            self.w_pre = self.w
            self.c_pre = self.c
            # next values become current values
            self.w = w_next
            self.c = c_next
            loss = self.loss(np.vstack((self.w, self.c)))
            grad_norm = np.linalg.norm(self.grad)
            print("Loss: {} \t Grad norm: {}".format(loss, grad_norm))
            self.loss_history.append(loss)
        print("process ended with gradient norm {}".format(grad_norm))
        print("Iterations needed: {}".format(len(self.loss_history)))

def initialize_settings():
    settings = {'arm': False,
                'bb': False,
                'stoc': False,
                }
    return settings
if __name__ == "__main__":
    # try toy data for a smaller dataset
    settings = initialize_settings()
    dataset = "data.csv"
    df = pd.read_csv(dataset)
    data = read_input(df)
    ln = LinearRegression(data, beta=0.9, tol=0.001, sigma=0.01, s=0.01, lam=0.00001, ro_min=0.1, ro_max=2, m=10)

    #need a better way to to this
    settings['arm'] = True
    settings['bb'] = True
    settings['stoc'] = True
    
    ln.fit(settings)
    ln.plot_loss()
    ln.plot_classifier()
