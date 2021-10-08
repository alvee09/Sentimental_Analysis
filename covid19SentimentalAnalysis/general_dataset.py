import pandas as pd
import csv

from numpy.distutils.fcompiler import none

file = open('dataset.csv')
csvreader = csv.reader(file)
print(type(csvreader))
#
# header = []
# header = next(csvreader)
# print(header)

df = pd.read_csv(file)
df = df.iloc[:10000]
print(df.shape)