
import joblib
from scipy.sparse import load_npz
import tensorflow as tf
import numpy as np
import argparse

# Create argument parser
arg_parser = argparse.ArgumentParser()

# Add arguments
arg_parser.add_argument('--user_id', type = int, default = 1, help = 'Id of the user')
arg_parser.add_argument('--top_n', type = int, default = 10, help = 'The number of movies to recommend' )

# Parse argument
args = arg_parser.parse_args()

# Access the argument
user_id = args.user_id
n = args.top_n

# Load model weights
W, b, c = joblib.load('weights.pkl')
K = 10
ratings = load_npz('train_sparse.npz')
N = ratings.shape[0]

def dot1(V, W):
    # V is N x D x K (batch of visible units)
    # W is D x K x M (weights)
    # returns N x M (hidden layer size)
    return tf.tensordot(V, W, axes=[[1,2], [0,1]])

def dot2(H, W):
    # H is N x M (batch of hiddens)
    # W is D x K x M (weights transposed)
    # returns N x D x K (visible)
    return tf.tensordot(H, W, axes=[[1], [2]])

def forward_hidden(v):
    return tf.nn.sigmoid(dot1(v, W) + c)

def forward_logits(v):
    Z = forward_hidden(v)
    return dot2(Z, W) + b

def forward_output(v):
    return tf.nn.softmax(forward_logits(v))


def generate_ranking(user_id):
    # Get user ratings 
    user_rating = ratings[user_id,:].toarray()
    user_input = tf.constant(user_rating*2-1, dtype = tf.int32) 
    # Convert to binary             
    user_input_hot = tf.one_hot(user_input, K, dtype = tf.float32)   
    # Predict user ratings for all movies                  
    one_to_ten = tf.constant((np.arange(10) + 1).astype(np.float32) / 2)                
    output_visible = forward_output(user_input_hot)
    pred = tf.squeeze(tf.tensordot(output_visible, one_to_ten, axes=[[2], [0]]))
    # Get the indices of movies already watched by the user
    watched = tf.cast(tf.squeeze(user_rating)>0, dtype = tf.float32)

    # Calculate MAE for the user rating
    pred_watched = np.where(watched,pred,0)
    diff = tf.abs(user_rating - pred_watched)
    count = tf.reduce_sum(watched)
    mae = tf.reduce_sum(diff).numpy()/count.numpy()
    print(f'\n\nAverage prediction deviance for user {user_id} = {mae}')

    # Rank the unwatched movies by rating in descending order
    pred_not_watched = np.where(watched, 0, pred)
    print( f'User has watched {int(tf.reduce_sum(watched).numpy())} movies in total')
    movie_rating = {movie:rating for movie, rating in enumerate(pred_not_watched)}
    top_movies = sorted(movie_rating, key = lambda x: movie_rating[x], reverse = True)
    print(f'Top {n} new recommended movies (ids) for the user {user_id} is: {top_movies[:n]}')
    return user_input, pred

# Get top n recommendations
rating_history, predicted_ratings = generate_ranking(user_id)
