import numpy as np

#1
A = np.array([1,2,3,4])
B = np.array([4,5,6,7])

vertical = np.vstack((A,B))
horizontal = np.hstack((A,B))

print(vertical)
print('\n')
print(horizontal)
print('\n')

#2
common = np.intersect1d(A,B)
print(common)
print('\n')

#3
extracting_numbers = A[(A >= 1) & (A <=3)]
print(extracting_numbers)
print('\n')

#4
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

filter_row = iris_2d[(iris_2d[:, 2] > 1.5) & (iris_2d[:,0] < 5.0)]
print(filter_row)
print('\n')

#5
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

filter_info = df.loc[::20, ['Manufacturer', 'Model', 'Type']]
print(filter_info)
print('\n')

#6
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

df['Min.Price'].fillna(df['Min.Price'].mean(), inplace=True)
df['Max.Price'].fillna(df['Max.Price'].mean(), inplace=True)

print(df[['Manufacturer','Min.Price', 'Max.Price']])
print('\n')

#7
df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))

sum_rows = df[df.apply(lambda row: row.sum(), axis=1) > 100]
print(sum_rows)


