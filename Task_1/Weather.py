import numpy as np
import pandas as pd

df = pd.read_csv("weatherHistory.csv")
df = df['Wind Speed','Temperature']
df.dropna(inplace=True)

X = df['Wind Speed'].values
y = df['Temperature'].values
n = len(X)

w = 0 
b = 0  
lr = 0.1  
epochs = 100

for i in range(epochs):
    y_pred = w * X + b
    
    dw = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    w = w - lr * dw
    b = b - lr * db


print("\n Training Completed.")
print(f"Final weight (w): {w}")
print(f"Final bias (b): {b}")
print(f"Final Temperature (y_pred): {y_pred}")