import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# Read the data
df = pd.read_csv("TestModel.csv")

# Search for the best alpha value

alpha_values = np.arange(0.01, 50, 0.001)

A_0 = 0.02927602 #keep 40.7 constant for denominator if you have no A_0 over Total Compounds at T = 0 Value

##########################################################################################
# Define X and Y data
x_column = 'Time'
y_column = 'Concentration'
x = df[x_column]
y = df[y_column]
Ymax = np.max(y)
Xmax = x[np.argmax(y)]

# Define the compartmental equation
def compartmental_equation(alpha, K1, y):
    return (np.exp(-alpha * K1 * y) * (A_0 + ((1 / (alpha - 1)) * (((np.exp((alpha - 1) * K1 * y)) - 1))))) / (((alpha - (A_0 * alpha ** 2) + (alpha * A_0)) ** (-alpha / (alpha - 1))) * (A_0 - (A_0 * alpha) + 1))


# Search for the best alpha value
rss_values = []

for alpha in alpha_values:
    denominator_value = alpha * (1 + A_0) - (A_0 * (alpha ** 2))
    if denominator_value <= 0:
        continue  # Skip this iteration if the denominator value is non-positive

    K1_denominator = Xmax * (alpha - 1)
    if K1_denominator == 0:
        continue  # Skip this iteration if the K1 denominator is zero

    K1 = math.log(denominator_value) / K1_denominator
    K2 = K1 * alpha

    eq_values = compartmental_equation(alpha, K1, x)
    rss = np.sum((eq_values - y / Ymax) ** 2)
    if not np.isnan(rss):  # Skip this iteration if rss is NaN
        rss_values.append(rss)

best_alpha = alpha_values[np.argmin(rss_values)]
best_K1 = math.log(best_alpha * (1 + A_0) - (A_0 * (best_alpha ** 2))) / (Xmax * (best_alpha - 1))
best_K2 = best_K1 * best_alpha

# Calculate the result equation with the best alpha value
result_eq = compartmental_equation(best_alpha, best_K1, x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(alpha_values[:len(rss_values)], rss_values)
plt.xlabel('Alpha Value')
plt.ylabel('RSS')
# Define the desired tick positions and labels
plt.title('RSS vs Alpha Value')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(x, y/Ymax, label='Actual Data', color='blue')
plt.plot(x, result_eq, label='Model', color='red')
plt.xlabel('Time')
plt.ylabel('B/Bmax')
plt.legend()
plt.show()

print(f"Best Alpha: {best_alpha}")
print(f"K1: {best_K1}")
print(f"K2: {best_K2}")
print("R squared:", r2_score(y/Ymax, result_eq))
print("RSS:",np.sum(y/Ymax - result_eq) **2)
