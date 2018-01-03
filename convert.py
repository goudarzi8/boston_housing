import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
import sklearn
import csv

# This part is designed to chande .data file into csv
# I am making csv file to use pandas

with open('data/housing.data') as input_file:
   lines = input_file.readlines()
   newLines = []
   for line in lines:
      newLine = line.strip().split()
      newLines.append( newLine )

with open('data/data.csv', 'wb') as test_file:
   file_writer = csv.writer(test_file)
   file_writer.writerows( newLines )