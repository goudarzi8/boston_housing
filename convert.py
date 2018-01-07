import numpy as np
import scipy as sp
import pandas as pd

import csv

# This part is designed to chande .data file into csv
# I am making csv file to use pandas

with open('data/housing.data') as input_file: #locading the data file
   lines = input_file.readlines()
   newLines = [] # creating empty array to save the lines and move them into csv file
   for line in lines:
      newLine = line.strip().split()
      newLines.append( newLine )

with open('data/data.csv', 'wb') as test_file: # creating csv file and saving data file into it
   file_writer = csv.writer(test_file)
   file_writer.writerows( newLines ) # moving the lines into the csv file