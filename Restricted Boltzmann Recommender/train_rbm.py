
import numpy as np
from datetime import datetime
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.sparse import csr_matrix, load_npz
import matplotlib.pyplot as plt
import joblib
import argparse

# Create an argument parser
parser = argparse.ArgumentParser()
# Add arguments
parser.add_argument('--epochs', type = int, default  = 10, help = 'Number of epochs')
parser.add_argument('--batch_size', type = int, default = 128, help = 'Batch size')
parser.add_argument('--n_neurons', type = int, default = 256, help = 'Size of Hidden layer')
parser.add_argument('--cd_k', type = int, default = 1, help = 'Number of steps for Gibbs sampling')
parser.add_argument('--learning_rate', type = float, default = 0.01, help = 'Learning rate for training')
parser.add_argument('--output_file', type = str, default = 'weights.pkl', help = 'path for saving model weights')

# Parse the arguments
args = parser.parse_args()

# Load sparse train and test matrices
ratings = load_npz('train_sparse.npz')
test_ratings = load_npz('test_sparse.npz')


# Define parameters
D = ratings.shape[1]  # Number of movies
K = 10  # Number of rating categories (e.g. 1-5 stars)

# Access the arguments and assign to parameters 
M = args.n_neurons  # Number of hidden units
batch_size = args.batch_size  # Batch size for training
learning_rate = args.learning_rate  # Learning rate for Adam optimizer
cd_k = args.cd_k # The number of steps for gibbs sampling
epochs = args.epochs  # Number of epochs to train

# Define auxilary dot product functions for forward and backward layer calculations 
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

# Define the RBM model
class RBM(tf.keras.Model):
    def __init__(self, D, K, M):
        super(RBM, self).__init__()
        self.W = tf.Variable(tf.random.normal([D, K, M], stddev= np.sqrt(2/M)))
        self.b = tf.Variable(tf.zeros([D, K], dtype = tf.float32))
        self.c = tf.Variable(tf.zeros([M],dtype = tf.float32))

    def forward_hidden(self, v):
        return tf.nn.sigmoid(dot1(v, self.W) + self.c)

    def forward_logits(self, v):
        Z = self.forward_hidden(v)
        return dot2(Z, self.W) + self.b

    def forward_output(self, v):
        return tf.nn.softmax(self.forward_logits(v))

    def call(self, v):
        p_h_given_v = self.forward_hidden(v) #tf.nn.sigmoid(dot1(v, self.W) + self.c)
        r = tf.random.uniform(shape=tf.shape(input=p_h_given_v))
        h = tf.cast(r < p_h_given_v, dtype=tf.float32)
        return h, p_h_given_v
        
    def free_energy(self, v):
        first_term = -tf.reduce_sum(input_tensor=dot1(v, self.b))
        second_term = -tf.reduce_sum(
            # tf.log(1 + tf.exp(tf.matmul(V, self.W) + self.c)),
            input_tensor=tf.nn.softplus(dot1(v, self.W) + self.c), axis=1)
        return first_term + second_term

    def contrastive_divergence(self, v, mask, k=cd_k):
        h, p_h_given_v = self.call(v)
        for i in range(k):
            logits = dot2(h, self.W) + self.b
            cdist = tfp.distributions.Categorical(logits = logits)
            v = cdist.sample() # shape is (N, D)
            v = tf.one_hot(v, depth=K)*mask # turn it into (N, D, K)
            h, p_h_given_v = self.call(v)
        return v
    
# Define the RBM model object and optimizer
rbm = RBM(D, K, M)
optimizer = tf.optimizers.Adam(learning_rate)

# Define the training loop
n_batch = ratings.shape[0]//batch_size
train_test_error = {'train_sse':[], 'test_sse': []}
for epoch in range(epochs):
    sse_sum, tsse_sum = 0, 0
    sse_count, tsse_count = 0, 0
    for i in range(0, ratings.shape[0], batch_size):
        # Extract the current batch of ratings data and create mask
        x = ratings[i:i+batch_size].toarray()
        x_hot = tf.one_hot(x*2 - 1, K)
        mask2d = tf.cast(x > 0, tf.float32)
        mask = tf.stack([mask2d]*K, axis=-1)

        # Perform one step of contrastive divergence
        with tf.GradientTape() as tape:
            v = rbm.contrastive_divergence(x_hot, mask)
            cost = tf.reduce_mean(rbm.free_energy(x_hot)) - tf.reduce_mean(rbm.free_energy(v))

        # Update the model parameters using the Adam optimizer
        grads = tape.gradient(cost, rbm.trainable_variables)
        optimizer.apply_gradients(zip(grads, rbm.trainable_variables))

        # Calculate train sse
        one_to_ten = tf.constant((np.arange(10) + 1).astype(np.float32) / 2)
        output_visible = rbm.forward_output(x_hot)
        pred = tf.tensordot(output_visible, one_to_ten, axes=[[2], [0]])
        sse = tf.reduce_sum(mask2d * (x - pred) * (x - pred))
        sse_sum = sse_sum + sse
        sse_count = sse_count + tf.reduce_sum(mask2d)

        # Calculate the test sse
        x_test = test_ratings[i:i+batch_size].toarray()
        test_x_hot = tf.one_hot(x_test*2 - 1, K)
        test_mask2d = tf.cast(x_test > 0, tf.float32)
        tsse = tf.reduce_sum(test_mask2d * (x_test - pred) * (x_test - pred))
        tsse_sum = tsse_sum + tsse
        tsse_count = tsse_count + tf.reduce_sum(test_mask2d)
        train_sse = sse_sum/sse_count
        test_sse = tsse_sum/tsse_count

    # Append the costs to train_test_error dictionary
    train_test_error['train_sse'].append(train_sse.numpy())
    train_test_error['test_sse'].append(test_sse.numpy())
    print(f'Epoch {epoch + 1}: Train SSE = {train_sse.numpy()} and Test SSE = {test_sse.numpy()}') 

# Plot mean squared error by epoch
try:
  assert epochs > 0
  plt.figure(figsize = (10,8))
  plt.plot(np.arange(epochs), train_test_error['train_sse'], label = 'Train SSE')
  plt.plot(np.arange(epochs), train_test_error['test_sse'], label = 'Test SSE')
  plt.legend()
  plt.savefig('model_performance')
except: 
  print('No history')
# Save the model weights
joblib.dump(rbm.get_weights(), args.output_file)
