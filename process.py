import pandas as pd
from datetime import datetime
import numpy as np


file = pd.read_csv('EuroMillions_numbers.csv', sep = ';', index_col=0)

file = file.sort_index(axis=0,ascending=True)
numbers = []

balls = list(range(1,51))
numbers = np.random.choice(balls,5,replace= False) 

star = list(range(1,13))
stars = np.random.choice(star,2,replace= False)

for i in range (1,50):
    print(stars)
