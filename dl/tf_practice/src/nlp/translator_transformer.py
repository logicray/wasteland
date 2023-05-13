#!/usr/bin/env python3
# -*- coding:utf8 -*-


import typing
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow_datasets as tfds


MAX_TOKENS = 128


def read_data(file_path):
    text = open(file_path, 'rb').readlines()
    print(len(text))
    print(text[0].decode('utf-8'))
    lines = [line.decode('utf-8') for line in text]
    pairs = [line.split("\t") for line in lines]
    # print(pairs[10])
    src = [src for tgt, src in pairs]
    tgt = [tgt for tgt, src in pairs]
    # vocab = sorted(set(text))
    return src, tgt


def to_dataset(src, tgt):
    BUFFER_SIZE = len(src)
    BATCH_SIZE = 64
    data = tf.data.Dataset.from_tensor_slices((src,  tgt)).shuffle(BUFFER_SIZE)
    dataset = data.batch(BATCH_SIZE)
    return dataset


def lower_and_split_punct(text):
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


def filter_max_tokens(sp, en):
    num_tokens = tf.maximum(tf.shape(sp)[1], tf.shape(en)[1])
    return num_tokens < MAX_TOKENS


def get_angles(pos, i, d_model):
    angle_rate = 1 / np.power(10000, 2*(i//2)/d_model)
    return pos * angle_rate


def positional_encoding(pos, d_model):
    angle_rads = get_angles(np.arange(pos)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def scaled_dot_product_attention(q, k, v, mask):
    matmal_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmal_qk / tf.math.sqrt(dk)

    # add mask to tensor
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAtten(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAtten, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        split last dim to num_heads, depth
        transpose
        :param x:
        :param batch_size:
        :return:
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=(0, 2, 1,3))

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_atten, atten_weight = scaled_dot_product_attention(q, k, v, mask)
        scaled_atten = tf.transpose(scaled_atten, perm=(0,2,3,1))
        concat_atten = tf.reshape(scaled_atten, (batch_size, -1, self.d_model))
        output = self.dense(concat_atten)
        return output, atten_weight


def point_wise_ffn(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAtten(d_model, num_heads)
        self.ffn = point_wise_ffn(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, **kwargs):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training= training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training= training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAtten(d_model=d_model, num_heads=num_heads)
        self.mha2 = MultiHeadAtten(d_model=d_model, num_heads=num_heads)

        self.ffn = point_wise_ffn(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate=rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=rate)
        self.dropout3 = tf.keras.layers.Dropout(rate=rate)


    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weight_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weight_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        return out3, attn_weight_block1, attn_weight_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        #
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(MAX_TOKENS, d_model)
        self.enc_layers = [
            EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate=rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # embedding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # positional encoding
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, output_vocab_size, rate=0.1):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        # step 1
        self.embedding = tf.keras.layers.Embedding(output_vocab_size, d_model)
        self.pos_encoding = positional_encoding(MAX_TOKENS, d_model)
        # step 2
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate=rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        # embedding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # positional encoding
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer_{i+1}_block1'] = block1
            attention_weights[f'decoder_layer_{i+1}_block2'] = block2

        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, output_vocab_size, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, output_vocab_size)
        self.final_layer = tf.keras.layers.Dense(output_vocab_size)

    def call(self, inputs, training=None, mask=None):
        input_text, output_text = inputs
        padding_mask, look_ahead_mask = self.create_mask(input_text, output_text)
        enc_output = self.encoder(input_text, training, padding_mask)
        dec_output, attention_weights = self.decoder(output_text, enc_output, training, look_ahead_mask, padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights

    @staticmethod
    def create_mask(src, tgt):
        padding_mask = create_padding_mask(src)

        look_ahead_mask = create_look_ahead_mask(tf.shape(tgt)[1])
        dec_target_padding_mask = create_padding_mask(tgt)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return padding_mask, look_ahead_mask


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step, **kwargs):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]


def preprocess():
    BUFFER_SIZE = 20000
    BATCH_SIZE = 64
    src, tgt = read_data("/home/page/workspace/deeplab/data/spa-eng/spa.txt")

    dataset = to_dataset(src, tgt)

    max_vocab_size = 5000
    src_text_processor = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=lower_and_split_punct,
        max_tokens=max_vocab_size
    )
    src_text_processor.adapt(src)
    print(src_text_processor.get_vocabulary()[0:10])

    tgt_text_processor = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=lower_and_split_punct,
        max_tokens=max_vocab_size
    )
    tgt_text_processor.adapt(tgt)

    def tokenize_pairs(sp, en):
        sp = src_text_processor(sp)
        en = tgt_text_processor(en)

        return sp, en

    def make_batches(ds, buffer_size, batch_size):
        return (
            ds
                .cache()
                .shuffle(buffer_size)
                .batch(batch_size)
                .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
                .filter(filter_max_tokens)
                .prefetch(tf.data.AUTOTUNE))

    train_examples = make_batches(dataset, BUFFER_SIZE, BATCH_SIZE)

    return train_examples, src_text_processor.vocabulary_size(), tgt_text_processor.vocabulary_size()


def preprocess2():
    BUFFER_SIZE = 20000
    BATCH_SIZE = 64
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                                   as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    model_name = 'ted_hrlr_translate_pt_en_converter'
    tf.keras.utils.get_file(
        f'{model_name}.zip',
        f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
        cache_dir='.', cache_subdir='', extract=True
    )
    tokenizers = tf.saved_model.load(model_name)

    def tokenize_pairs(pt, en):
        pt = tokenizers.pt.tokenize(pt)
        # Convert from ragged to dense, padding with zeros.
        pt = pt.to_tensor()

        en = tokenizers.en.tokenize(en)
        # Convert from ragged to dense, padding with zeros.
        en = en.to_tensor()
        return pt, en

    def make_batches(ds):
        return (
            ds
                .cache()
                .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE)
                .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
                .filter(filter_max_tokens)
                .prefetch(tf.data.AUTOTUNE))

    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)
    return train_batches, val_batches, tokenizers


def train():
    EPOCHS = 2
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    training_batches,val_batches, tokenizers = preprocess2()
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
        output_vocab_size=tokenizers.en.get_vocab_size().numpy(),
        rate=dropout_rate
    )

    checkpoint_path = './transformer/train'
    ckpt = tf.train.Checkpoint(transformer = transformer, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("latest checkpoint restored")

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_rea = tar[:, 1:]
        with tf.GradientTape() as tape:
            predictions, _ = transformer([inp, tar_inp], training=True)
            loss = loss_function(tar_rea, predictions)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        train_loss(loss)
        train_accuracy(accuracy_function(tar_rea, predictions))

    # for epoch in range(EPOCHS):
    #     start = time.time()
    #     train_loss.reset_states()
    #     train_accuracy.reset_states()
    #
    #     for (batch, (inp, tar)) in enumerate(training_batches):
    #         train_step(inp, tar)
    #
    #         if batch % 50 == 0:
    #             print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
    #     if (epoch + 1) % 5 == 0:
    #         ckpt_save_path = ckpt_manager.save()
    #         print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')
    #
    #     print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
    #     print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
    return tokenizers, transformer


def read_ck_and_inference(tokenizers, transformer):
    translator0 = Translator(tokenizers, transformer)

    sentence = 'este é um problema que temos que resolver.'
    ground_truth = 'this is a problem we have to solve .'
    translated_text, translated_tokens, attention_weights = translator0(
        tf.constant(sentence))
    print_translation(sentence, translated_text, ground_truth)

    translator = ExportTranslator(translator0)
    tf.saved_model.save(translator, export_dir='translator')


class Translator(tf.Module):
    def __init__(self, tokenizers, transformer):
        self.tokenizers = tokenizers
        self.transformer = transformer
        super(Translator, self).__init__()

    def __call__(self, sentence: tf.Tensor, max_length=MAX_TOKENS):
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()
        encoder_input = sentence

        start_end = self.tokenizers.en.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # for speed
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)
        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions, _ = self.transformer([encoder_input, output], training=False)
            predictions = predictions[:, -1:, :]
            predicted_id = tf.argmax(predictions, axis=-1)
            output_array = output_array.write(i+1, predicted_id[0])
            if predicted_id == end:
                break
        output = tf.transpose(output_array.stack())
        text = self.tokenizers.en.detokenize(output)[0]
        tokens = self.tokenizers.en.lookup(output)[0]
        _, attention_weights = self.transformer([encoder_input, output[:, :-1]], training=False)
        return text, tokens, attention_weights


def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input: 15s"}: {sentence}')
    print(f'Predictions: 15s:{tokens.numpy().decode("utf-8")}')
    print(f'ground truth: 15s: {ground_truth}')


class ExportTranslator(tf.Module):
    def __init__(self, translator:Translator):
        self.translator = translator
        super(ExportTranslator, self).__init__()

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        (result, tokens, attention_weights) = self.translator(sentence, max_length=MAX_TOKENS)
        return result


def main():
    # n, d = 2048, 512
    # res = positional_encoding(n, d)
    # print(res.shape)
    #
    # pe = res[0]
    # print(pe[1])
    # print("shape of pe: ", pe.shape)
    # pe = tf.reshape(pe, (n, d // 2, 2))
    # print("after 1 ", pe.shape)
    # pe = tf.transpose(pe, (2, 1, 0))
    # print("after 2 ", pe.shape)
    # pe = tf.reshape(pe, (d, n))

    # print(pe[:, 0])
    # plt.pcolormesh(pe, cmap='RdBu')
    # plt.ylabel('Depth')
    # plt.xlabel('Position')
    # plt.colorbar()
    # plt.show()

    tmp_k = tf.constant([[10, 0, 0],
                         [0, 10, 0],
                         [0, 0, 10],
                         [0, 0, 10]
                         ], dtype=tf.float32)
    tmp_v = tf.constant([[1, 0],
                         [10,0],
                         [100, 5],
                         [1000, 6]
                         ], dtype=tf.float32)

    tmp_q = tf.constant([[0,10,0]], dtype=tf.float32)
    print("shape q:", tmp_q.shape)

    out, atte_w = scaled_dot_product_attention(tmp_q, tmp_k, tmp_v, None)
    np.set_printoptions(suppress=True)
    print(out)
    print("atten:", atte_w)

    tmp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)
    out, atte_w = scaled_dot_product_attention(tmp_q, tmp_k, tmp_v, None)
    print(out)
    print("atten:", atte_w)

    tmp_mha = MultiHeadAtten(512, 8)
    y = tf.random.uniform((1, 60 ,512))
    out, atte = tmp_mha(y, y, y, None)
    print(out.shape, atte.shape)

    # x = tf.random.uniform((1, 3))
    # tmp = create_look_ahead_mask(x.shape[1])
    # print(tmp)
    sample_ffn = point_wise_ffn(512, 2048)
    tmp = sample_ffn(tf.random.uniform((64, 50, 512)))
    print(tmp.shape)

    sample_encoder_layer = EncoderLayer(512, 8, 2048)
    sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None)
    print(sample_encoder_layer_output.shape)

    sample_decoder_layer = DecoderLayer(d_model=512, num_heads=8, dff=2048)
    sample_decoder_layer_output = sample_decoder_layer(tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
                                                       False, None, None)
    print(sample_decoder_layer_output[0].shape)

    sample_transformer = Transformer(num_layers=2, d_model=512, num_heads=8, dff=2048,
                                     input_vocab_size=8500, output_vocab_size=8000)
    tmp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    tmp_output = tf.random.uniform((64,36), dtype=tf.int64, minval=0, maxval=200)
    fn_out, _ = sample_transformer([tmp_input, tmp_output], training=False)
    print(fn_out.shape)

    # learning_rate = CustomSchedule(d_model=512)
    # plt.plot(learning_rate(tf.range(40000, dtype=tf.float32)))
    # plt.ylabel('learning rate')
    # plt.xlabel('train step')
    # plt.show()
    # tokenizers, transformer = train()
    # read_ck_and_inference(tokenizers, transformer)

    reloaded = tf.saved_model.load('translator')
    print(reloaded('este é o primeiro livro que eu fiz.').numpy())
    # training_examples, src_vocab_size, tgt_vocab_size = preprocess()
    # for (batch, (inp, tar)) in enumerate(training_examples):
    #     print(batch)


if __name__ == '__main__':
    main()
    # training_examples, src_vocab_size, tgt_vocab_size = preprocess()
    # print("preprocess end!", )
    # print(type(training_examples))
    # (inp, tar) = training_examples.batch(batch_size=64)
    # print(inp[0])