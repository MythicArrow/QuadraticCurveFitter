import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Generate synthetic data based on user input
def generate_data(a, b, c, n_points, noise_level):
    x = np.linspace(-10, 10, n_points)
    y = a * x**2 + b * x + c + np.random.randn(n_points) * noise_level  # Adding noise
    return x, y

# Step 2: Ask the user for inputs
print("Quadratic Curve Fitting with Neural Networks")
a = float(input("Enter the coefficient 'a' for the quadratic equation (ax^2): "))
b = float(input("Enter the coefficient 'b' for the quadratic equation (bx): "))
c = float(input("Enter the constant 'c' for the quadratic equation: "))
n_points = int(input("Enter the number of data points to generate: "))
noise_level = float(input("Enter the noise level (e.g., 5 for high noise, 0 for no noise): "))

# Step 3: Generate data using the user inputs
x_data, y_data = generate_data(a, b, c, n_points, noise_level)

# Step 4: Create and train a neural network model to learn the quadratic curve
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))  # Output layer for predicting y

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_data, y_data, epochs=500, verbose=0)  # Train on the generated data

# Step 5: Use the model to predict the quadratic curve
x_values = np.linspace(-10, 10, 400)
y_predicted = model.predict(x_values)

# Step 6: Plot the true data and the AI-predicted curve
plt.scatter(x_data, y_data, label='Data points', color='blue')  # Original data points
plt.plot(x_values, y_predicted, label='AI-Predicted Curve', color='red')  # Predicted curve
plt.title("Quadratic Curve Fitting with Neural Networks")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.legend()
plt.grid(True)
plt.show()
