#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf

"""
with tf2.5
"""
tf.debugging.set_log_device_placement(True)
with tf.device('/GPU:0'):
    vocab = sorted(set('abcdefghijklmnopqrstuvwxyz'))
    example_texts = ['abcdefg', 'xyz']
    chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
    print(chars)
    ids_from_chars = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=list(vocab), mask_token=None)
    ids = ids_from_chars(chars)
    print(ids)
    # decoding
    chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(),
                                                                             invert=True, mask_token=None)
    chars_recovery = chars_from_ids(ids)
    print(chars_recovery)
    print(tf.strings.reduce_join(chars_recovery, axis=-1).numpy())
