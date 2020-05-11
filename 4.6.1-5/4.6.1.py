import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 

df = pd.read_csv('Smarket.csv')

x = np.array(df['Count'])
y = np.array(df['Volume']).reshape((-1, 1))

m, b = np.polyfit(x, y, 1)

print(m)
print(b)

plt.plot(x, y, 'o')
plt.plot(x, m*x +b)
plt.title("Correlation between Volume and time")
plt.xlabel("Time") 
plt.ylabel("Volume") 

plt.savefig('bar.png')
plt.show()
