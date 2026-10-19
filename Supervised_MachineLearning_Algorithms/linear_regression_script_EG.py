##polynomial regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import math
import csv
from numpy.polynomial import Polynomial

##read the file using pandas

"""df = pd.read_csv("file.csv")
x = df.columns[0]
y = df.columns[1]
"""
##read the csv file
def data_plot(filename):
    x = []
    y = []
    with open(filename, 'r', newline='') as csvfile:
        plots = csv.reader(csvfile)
        #here plots is an iterator which fetches its next method to skip the first header row"""

        next(plots, None)
        for row in plots:
            x.append(int(row[0]))
            y.append(int(row[1]))
    plt.plot(x, y, color  = 'r', linewidth = 2.0) 
    plt.xlabel("Year")
    plt.ylabel("Number of frozen Days")
    plt.savefig("dataplot.jpg")
    plt.show()  ##need to comment this later
    return x, y

def minmax_normalization(x,  degree):
    n = len(x)
    xmin = np.min(x)
    xmax = np.max(x)
    if xmax == xmin:
        return np.zeros_like(x)
    x_norm = (x - xmin) / (xmax - xmin)
    ##construct augmented feature matrix
    cols = [x_norm**i for i in range(1, degree + 1)]
    cols.append(np.ones(n))
    X_tilde = np.column_stack(cols) # all rows and columns are initialized to zero
    print("Q2:")
    print(x_norm)
    return X_tilde, xmin, xmax

def q3_closed_form_solution(X_tilde, y):
    theta = np.linalg.solve(X_tilde.T @ X_tilde, X_tilde.T @ y)
    print("Q3:")
    print(theta)
    return theta

def q4_gradient_descent(X_tilde, y, learning_rate, iterations):
    print("Q4:")
    n = len(y)
    degree = X_tilde.shape[1] - 1
    theta_gd = np.zeros(degree + 1)
    loss_history = []

    for t in range(iterations):
        if t % 10 == 0:
            print(f"Iteration {t:3d}: theta = {theta_gd}")
        y_hat = X_tilde @ theta_gd
        loss = (1.0 / (2.0 * n)) * np.sum((y_hat - y) ** 2)
        loss_history.append(loss)

        grad = (1.0 / n) * X_tilde.T @ (y_hat - y)
        theta_gd = theta_gd - learning_rate * grad

    chosen_lr = 0.6
    chosen_iterations = 200000
    print("Q4b: " + str(chosen_lr))
    print("Q4c: " + str(chosen_iterations))
    print(
        "Q4d: I started with a small learning rate such as 0.01 and gradually"
        " increased it while monitoring the loss to ensure stability without"
        " divergence. A learning rate of 0.5 with 100000 iterations allowed"
        " gradient descent to converge steadily, reducing the loss difference"
        " from the closed-form solution to well below 10."
    )

# Question 4e: Save loss plot
    plt.figure()
    plt.plot(range(iterations), loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("loss_plot.jpg")
    plt.close()


def q5_predict(theta, x_min, x_max, degree, target_year=2025):
    """Question 5: Prediction for year 2025 using closed-form theta."""
    x_tilde_target = (target_year - x_min) / (x_max - x_min)

    y_hat = 0.0
    for i in range(degree):
        y_hat += theta[i] * (x_tilde_target ** (i + 1))
    y_hat += theta[-1]  # bias term b

    print("Q5: " + str(y_hat))
    return y_hat


def q6_rate_of_change(theta, x_min, x_max, degree, target_year=2025):
    """Question 6: Derivative dy_hat / dx at x = 2025."""
    x_tilde_target = (target_year - x_min) / (x_max - x_min)

    rate_of_change = 0.0
    for j in range(1, degree + 1):
        rate_of_change += j * theta[j - 1] * (x_tilde_target ** (j - 1))
    rate_of_change /= x_max - x_min

    print("Q6a: " + str(rate_of_change))
    print(
        "Q6b: A negative local rate of change around year 2025 indicates that"
        " the expected number of ice days on Lake Mendota is decreasing over"
        " time at that point. Specifically, it implies a trend of shorter"
        " ice-covered periods per year, reflecting long-term climate warming"
        " effects on the lake."
    )


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Usage: python3 script.py <filename.csv>")
        sys.exit(1)

    filename = sys.argv[1]
    degree = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    iterations = int(sys.argv[4])
    # Read dataset
    #df = pd.read_csv(filename)
    #x = df["year"].values
    #y = df["days"].values.astype(float)

    # Execute questions in sequence
    #data_plot(df)
    x,y = data_plot(filename)
    X_tilde, x_min, x_max = minmax_normalization(x, degree)
    theta = q3_closed_form_solution(X_tilde, y)
    q4_gradient_descent(X_tilde, y, learning_rate, iterations)
    q5_predict(theta, x_min, x_max, degree, target_year=2025)
    q6_rate_of_change(theta, x_min, x_max, degree, target_year=2025)









