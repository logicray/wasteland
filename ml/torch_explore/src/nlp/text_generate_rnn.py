#!/usr/bin/env python3
# -*- coding:utf8 -*-

import tensorflow as tf
import numpy as np
import os
import time


def load():
    file_path = tf.keras.utils.get_file('shakespeare.txt',
                                        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    print(file_path)
    return file_path


def read_data(file_path):
    text = open(file_path, 'rb').read().decode('utf-8')
    print(len(text))
    vocab = sorted(set(text))
    print("vocab num:", vocab)
    return text, vocab


def text_from_ids(chars_from_ids, ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


def encoding(vocab, text):
    ids_from_chars = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(),
                                                                             invert=True, mask_token=None)
    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    for ids in ids_dataset.take(10):
        print(chars_from_ids(ids), '---------')

    seq_length = 100
    examples_per_epoch = len(text)
    sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)
    for seq in sequences.take(3):
        print(text_from_ids(chars_from_ids, seq).numpy(), '====')
    print("all_ids", all_ids)
    dataset = sequences.map(split_input_target)
    for input_example, target_example in dataset.take(2):
        print("input", text_from_ids(chars_from_ids, input_example).numpy())
        print("output", text_from_ids(chars_from_ids, target_example).numpy())
    return ids_from_chars, chars_from_ids, dataset


def split_input_target(sequence):
    # sequence = list(sequence)
    input_text = sequence[:-1]
    output_text = sequence[1:]
    return input_text, output_text


class GenerateModel(tf.keras.Model):
    # def get_config(self):
    #     pass

    def __init__(self, vocab_size, embedding_dim, rnn_unit):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_unit, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_states=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_states:
            return x, states
        else:
            return x


class CustomTraining(GenerateModel):
    """
    diy model train step
    """

    @tf.function
    def train_step(self, inputs):
        inputs, labels = inputs
        with tf.GradientTape as tape:
            predictions = self(inputs, training=True)
            loss = self.loss(labels, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return {'loss': loss}


class OneStep(tf.keras.Model):
    def call(self, inputs, training=None, mask=None):
        pass

    # def get_config(self):
    #     pass

    def __init__(self, model: GenerateModel, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        skip_ids = self.ids_from_chars(['UNK'])[:, None]
        sparse_mask = tf.SparseTensor(
            values=[-float('inf')] * len(skip_ids),
            indices=skip_ids,
            dense_shape=[len(ids_from_chars.get_vocabulary())]
        )
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # run model
        predicted_logits, states = self.model(inputs=input_ids, states=states, return_states=True)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        predicted_logits = predicted_logits + self.prediction_mask

        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)
        predicted_chars = self.chars_from_ids(predicted_ids)

        return predicted_chars, states


def main():
    path = load()
    text, vocab = read_data(path)
    ids_from_chars, chars_from_ids, dataset = encoding(vocab, text)

    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    dataset = (
        dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
    print(dataset)

    embedding_dim = 256
    rnn_units = 1024
    model = GenerateModel(len(ids_from_chars.get_vocabulary()), embedding_dim, rnn_units)

    for input_batch, target_batch in dataset.take(1):
        example_batch_predictions = model(input_batch)
        print(example_batch_predictions.shape)
        print(model.summary())

        sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
        sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
        print(sampled_indices)

        print(text_from_ids(chars_from_ids, input_batch[0]).numpy())
        print("===========")
        print("next char:", text_from_ids(chars_from_ids, sampled_indices).numpy())

        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        example_batch_mean_loss = loss(target_batch, example_batch_predictions)
        print("prediction shape:", example_batch_predictions.shape)
        print("mean loss:", example_batch_mean_loss)

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)
    checkpoint_dir = "./checkpoint"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
    EPOCHS = 2
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    # generate/inference
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
    start = time.time()
    states = None
    next_char = tf.constant(['ROMEO:'])
    result = [next_char]
    for n in range(1000):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)
    result = tf.strings.join(result)
    end = time.time()
    print(result[0].numpy().decode('utf-8'), '\n\n' + '*' * 80)
    print("time consumed:", (end - start), " s")

    # model save and load
    print("start model save and load:")
    tf.saved_model.save(one_step_model, 'one_step')
    one_step_reloaded = tf.saved_model.load('one_step')

    states = None
    next_char = tf.constant(['ROMEO:'])
    result = [next_char]
    for n in range(100):
        next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
        result.append(next_char)
    result = tf.strings.join(result)
    print(result)


if __name__ == '__main__':
    main()
    # print(split_input_target(list("tensorflow")))
