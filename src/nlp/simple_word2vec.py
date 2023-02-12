#!/usr/bin/env python3
# -*- coding:utf8 -*-

"""
a simple word2vec model
"""
import collections
import random

import tensorflow as tf
import numpy as np
import math
from loader import load_text

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

vocabulary_size = 50000


def build_dataset(words) -> tuple:
    # build word frequency
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    # word, index map
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    index_list = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        index_list.append(index)
    count[0][1] = unk_count
    reversed_dict = dict(zip(dictionary.values(), dictionary.keys()))
    return index_list, count, dictionary, reversed_dict


def generate_batch(global_index: int, batch_size, num_skips, skip_window, index_list):
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(index_list[global_index])
        global_index = (global_index + 1) % len(index_list)

    for i in range(batch_size // num_skips):
        target = skip_window
        target_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in target_to_avoid:
                target = random.randint(0, span - 1)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
            target_to_avoid.append(target)
        buffer.append(index_list[global_index])
        global_index = (global_index + 1) % len(index_list)

    return batch, labels


words = load_text.load_data("../../data/text8.zip")
data, count, dictionary, reversed_dict = build_dataset(words)
print(count[0:5])
print("", data[0:10])

global_index = 0
batch, labels = generate_batch(global_index, 8, 2, 1, data)
print(batch.shape)
print(batch[0:8])
print(labels[0:8])

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
# ttmp=np.arange(valid_window)
tmp = random.sample(range(valid_window), valid_size)
valid_examples = np.array(tmp)
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Construct the variables.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    print(embeddings.dtype)
    print(embed.dtype)
    print(train_labels.dtype)
    loss = tf.reduce_mean(
        tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed,
                       num_sampled, vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

# Step 6: Begin training
num_steps = 30001

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    tf.initialize_all_variables().run()
    print("Initialized")

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(global_index, batch_size, num_skips, skip_window, data)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reversed_dict[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                # argsort 正常从小到大排序，加负号用来逆序排列
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = reversed_dict[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()


# Step 7: Visualize the embeddings.

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)


try:
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reversed_dict[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print("Please install sklearn and matplotlib to visualize embeddings.")

# def main():
#     words = load_text.load_data("../../data/text8.zip")
#     data, count, dictionary, reversed_dict = build_dataset(words)
#     print(count[0:5])
#     print("", data[0:10])
#
#     global_index = 0
#     batch, labels = generate_batch(global_index, 8, 2, 1, data)
#     print(batch.shape)
#     print(batch[0:8])
#     print(labels[0:8])


# if __name__ == '__main__':
#     tf.app.run()
