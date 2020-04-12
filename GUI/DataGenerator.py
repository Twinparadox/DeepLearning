import numpy as np
import pandas as pd


X1 = np.random.randint(1, 1000, size = 5000)
X2 = np.random.randint(1, 1000, size = 5000)
X3 = np.random.randint(1, 1000, size = 5000)
Y = X1 + X2 - X3

df = pd.DataFrame(np.array([X1, X2, X3, Y]).T, columns=['X1', 'X2', 'X3', 'Y'])
print(df)
df.to_csv('data.csv', index=False)