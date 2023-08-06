#!/usr/bin/env python3
# -*- coding:utf8 -*-

"""
文本分类
"""

import tensorflow as tf
import numpy as np
from sklearn import metrics
from loader import load_text

MAX_DOCUMENT_LENGTH = 50
WORDS_FEATURE = 'words'
EMBEDDING_SIZE = 15
MAX_LABEL = 15

n_words = 0


def estimator_spec_for_softmax_classification(logits, labels, mode):
    """Returns EstimatorSpec instance for softmax classification."""
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'class': predicted_classes,
                'prob': tf.nn.softmax(logits)
            })

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy':
            tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def bag_of_words_model(features, labels, mode):
    print("n_words in BoW: ", n_words)
    bow_column = tf.feature_column.categorical_column_with_identity(
        WORDS_FEATURE, num_buckets=n_words)
    bow_embedding_column = tf.feature_column.embedding_column(
        bow_column, dimension=EMBEDDING_SIZE)
    bow = tf.feature_column.input_layer(
        features, feature_columns=[bow_embedding_column])
    logits = tf.layers.dense(bow, MAX_LABEL, activation=None)

    return estimator_spec_for_softmax_classification(
        logits=logits, labels=labels, mode=mode)


def rnn_model(features, labels, mode):
    """RNN model to predict from sequence of words to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = tf.contrib.layers.embed_sequence(
        features[WORDS_FEATURE], vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    word_list = tf.unstack(word_vectors, axis=1)

    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)

    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for softmax
    # classification over output classes.
    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
    return estimator_spec_for_softmax_classification(
        logits=logits, labels=labels, mode=mode)


def main():
    datasets = load_text.load_dbpedia()
    print(datasets.train.features[0:5])
    x_train_s = datasets.train.features

    x_test_s = datasets.test.features

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)

    x_transform_train = vocab_processor.fit_transform([x[1] for x in x_train_s])
    x_transform_test = vocab_processor.transform([x[1] for x in x_test_s])

    x_train = np.array(list(x_transform_train))
    x_test = np.array(list(x_transform_test))
    global n_words
    n_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % n_words)

    # x_train -= 1
    # x_test -= 1
    model_fn = rnn_model
    classifier = tf.estimator.Estimator(model_fn=model_fn)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={WORDS_FEATURE: x_train},
        y=np.array(datasets.train.label, dtype=np.int32),
        batch_size=len(x_train),
        num_epochs=None,
        shuffle=True)
    classifier.train(input_fn=train_input_fn, steps=100)

    # Predict.
    y_test = np.array(datasets.test.label, dtype=np.int32)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={WORDS_FEATURE: x_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)
    predictions = classifier.predict(input_fn=test_input_fn)
    y_predicted = np.array(list(p['class'] for p in predictions))
    y_predicted = y_predicted.reshape(np.array(y_test).shape)

    # Score with sklearn.
    score = metrics.accuracy_score(y_test, y_predicted)
    print('Accuracy (sklearn): {0:f}'.format(score))

    # Score with tensorflow.
    scores = classifier.evaluate(input_fn=test_input_fn)
    print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
    main()
