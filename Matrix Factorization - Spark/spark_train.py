from pyspark.mllib.recommendation import  MatrixFactorizationModel
from pyspark import SparkContext
import argparse

# Create argument parser, add and parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type = str, help = 'path of model artifact')
parser.add_argument('--user_id', type = int , default = 90, help = 'id of the currect user')
parser.add_argument('--numRecs', type = int , default = 10, help = 'number of recommendations')
args = parser.parse_args()
userId = args.user_id
numRecs = args.numRecs  # number of movies to recommend

# Start Spark Context
sc = SparkContext("local", "MF Recommender from saved model") # local machine

# Load the model
savedModel = MatrixFactorizationModel.load(sc, args.model_path)

# Generate top recommendations for a user
topRecommendations = savedModel.recommendProducts(userId, numRecs)
print("\n\nTop recommendations for user " + str(userId) + ":")
for recommendation in topRecommendations:
    print(recommendation.product, recommendation.rating)
