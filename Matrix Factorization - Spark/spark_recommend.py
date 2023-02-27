# Load required packages
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkContext
import argparse
import os
import shutil

# Create argument parser
parser = argparse.ArgumentParser()
# Add arguments
parser.add_argument('--filename', type = str, default = 'small_ratings.csv')
parser.add_argument('--rank', type = int, default = 15)
parser.add_argument('--epochs', type = int, default = 15)
parser.add_argument('--output_path', type = str, default = 'als_recommender')
# Parse arguments 
args = parser.parse_args()

# Increase Memory if data is big
# SparkContext.setSystemProperty('spark.driver.memory', '10g')
# SparkContext.setSystemProperty('spark.executor.memory', '10g')

# Start Spark Context
sc = SparkContext("local", "MF mllib implementation") # local machine

# Load data
data = sc.textFile(args.filename)
header = data.first() # extract header
data = data.filter(lambda row: row != header) # filter out header

# Convert the rows into Rating objectS - ((user, movie), rating) format
ratings = data.map(lambda row: row.split(',')).map(lambda row: Rating(int(row[0]), int(row[1]), float(row[2])))

# Split into Train and Test set
train, test = ratings.randomSplit([0.8, 0.2])

# Create ALS model
model = ALS.train(train, args.rank, args.epochs)
# Evaluate the model
x = train.map(lambda p: (p[0], p[1]))                             # convert data into ((user,movie),rating) format
p = model.predictAll(x).map(lambda r: ((r[0], r[1]), r[2]))       # make predictions
ratesAndPreds = train.map(lambda r: ((r[0], r[1]), r[2])).join(p) # actual and predicited rating values
mse = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()  # calculate mean squared error
print("Train MSE: %s" % mse)

# Repeat the same for test data
x = test.map(lambda p: (p[0], p[1]))
p = model.predictAll(x).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = test.map(lambda r: ((r[0], r[1]), r[2])).join(p)
mse = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Test MSE: %s" % mse)

if os.path.isdir(args.output_path):
  shutil.rmtree(args.output_path)
model.save(sc, args.output_path)
