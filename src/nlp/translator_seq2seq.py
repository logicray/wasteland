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


def preprocess():
    pass


class Encoder(tf.keras.layers.Layer):
    def __init__(self, src_vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.enc_units = enc_units
        #
        self.embedding = tf.keras.layers.Embedding(src_vocab_size, embedding_dim)

        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_state=True,
                                       return_sequences=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, tokens, state=None, **kwargs):
        vectors = self.embedding(tokens)
        output, state = self.gru(vectors, initial_state=state)

        return output, state


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)
        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value, mask):
        w1_query = self.W1(query)
        w2_key = self.W2(value)
        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask
        context_vector, attention_weight = self.attention(
            inputs = [w1_query, value, w2_key],
            mask = [query_mask, value_mask],
            return_attention_scores = True
        )
        return  context_vector, attention_weight


class DecoderInput(typing.NamedTuple):
    new_tokens: typing.Any
    enc_output: typing.Any
    mask: typing.Any


class DecoderOutput(typing.NamedTuple):
    logits: typing.Any
    attention_weights: typing.Any


class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.dec_units = dec_units

        # step 1
        self.embedding = tf.keras.layers.Embedding(output_vocab_size, embedding_dim)

        # step 2
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform'
                                       )
        self.attention = BahdanauAttention(dec_units)

        self.Wc = tf.keras.layers.Dense(dec_units, activation='tanh', use_bias=False)

        self.fc = tf.keras.layers.Dense(output_vocab_size)

    def call(self, inputs:DecoderInput, state=None, *args, **kwargs)->typing.Tuple[DecoderOutput, tf.Tensor]:
        vectors = self.embedding(inputs.new_tokens)
        rnn_output, state = self.gru(vectors, state)

        context_vector, attention_weights = self.attention(
            query = rnn_output,
            value = inputs.enc_output,
            mask = inputs.mask
        )

        context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)
        attention_vector = self.Wc(context_and_rnn_output)

        logits = self.fc(attention_vector)

        return DecoderOutput(logits, attention_weights), state


class MaskedLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        self.__call__(y_true, y_pred, )

    def __init__(self):
        super(MaskedLoss, self).__init__()
        self.name = 'masked_loss'
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def __call__(self, y, y_pred, **kwargs):
        loss = self.loss(y, y_pred)
        mask = tf.cast(y != 0, tf.float32)
        loss *= mask
        return tf.reduce_sum(loss)


class TrainTranslator(tf.keras.Model):
    def __init__(self,embedding_dim:int, units:int, input_text_processor:TextVectorization,
                 output_text_processor:TextVectorization,use_tf_func=True):
        super(TrainTranslator, self).__init__()
        self.encoder = Encoder(input_text_processor.vocabulary_size(), embedding_dim, units)
        self.decoder = Decoder(output_text_processor.vocabulary_size(), embedding_dim, units)
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        self.use_tf_func = use_tf_func

    def train_step(self, inputs):
        if self.use_tf_func:
            return self._tf_train_step(inputs)
        else:
            return self._train_step(inputs)

    def _preprocess(self, input_text, output_text):
        input_tokens = self.input_text_processor(input_text)
        output_tokens = self.output_text_processor(output_text)
        input_mask = input_tokens !=0
        output_mask = output_tokens!=0
        return input_tokens, input_mask, output_tokens, output_mask

    def _train_step(self, inputs):
        input_text, output_text = inputs
        (input_tokens, input_mask, output_tokens, output_mask) = self._preprocess(input_text, output_text)
        max_output_length = tf.shape(output_tokens)[1]

        with tf.GradientTape() as tape:
            enc_output, enc_state = self.encoder(input_tokens)
            dec_state = enc_state
            loss = tf.constant(0.0)

            for t in tf.range(max_output_length -1):
                new_tokens = output_tokens[:, t:t+2]
                step_loss, dec_state = self._loop_step(new_tokens, input_mask, enc_output, dec_state)

                loss += step_loss
            avg_loss = loss / tf.reduce_sum(tf.cast(output_mask, tf.float32))
        variables = self.trainable_variables
        gradients = tape.gradient(avg_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return {'batch_loss' : avg_loss}

    def _loop_step(self, new_tokens, input_mask, enc_output, dec_state):
        input_token, target_token = new_tokens[:,0:1], new_tokens[:, 1:2]
        decoder_input = DecoderInput(input_token, enc_output, input_mask)
        dec_res, dec_state = self.decoder(decoder_input, state = dec_state)
        y = target_token
        y_pred = dec_res.logits
        step_loss = self.loss(y, y_pred)

        return step_loss, dec_state

    @tf.function(input_signature=[[tf.TensorSpec(dtype=tf.string, shape=[None]),
                                   tf.TensorSpec(dtype=tf.string, shape=[None])]])
    def _tf_train_step(self, inputs):
        return self._train_step(inputs)


class BatchLogs(tf.keras.callbacks.Callback):
    def __init__(self, key):
        super(BatchLogs, self).__init__()
        self.key = key
        self.logs = []

    def on_train_batch_end(self, n, logs):
        self.logs.append(logs[self.key])


class Translator(tf.Module):
    def __init__(self, encoder: Encoder, decoder:Decoder,
                 input_text_processor:TextVectorization, output_text_processor:TextVectorization):
        super(Translator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor

        self.output_token_string_from_index = (
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=output_text_processor.get_vocabulary(),mask_token='',invert=True))

        index_from_string = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(),mask_token='',)

        token_mask_ids = index_from_string(['', '[UNK]', '[START]']).numpy()
        token_mask = np.zeros([index_from_string.vocabulary_size()], dtype=np.bool)
        token_mask[np.array(token_mask_ids)] = True
        self.token_mask = token_mask
        self.start_token = index_from_string(tf.constant('[START]'))
        self.end_token = index_from_string(tf.constant('[END]'))

    def token_to_text(self, result_tokens):
        result_text_tokens = self.output_token_string_from_index(result_tokens)
        result_text = tf.strings.reduce_join(result_text_tokens, axis=1, separator=' ')
        result_text = tf.strings.strip(result_text)
        return result_text

    def sample(self, logits, temperature):
        token_mask = self.token_mask[tf.newaxis, tf.newaxis, :]
        logits = tf.where(self.token_mask, -np.inf, logits)

        if temperature == 0.0:
            new_tokens = tf.argmax(logits, axis=-1)
        else:
            logits = tf.squeeze(logits, axis=1)
            new_tokens = tf.random.categorical(logits/temperature, num_samples=1)

        return new_tokens

    def translate(self, input_text, max_length=50, return_attention=True, temperature=1.0):
        batch_size = tf.shape(input_text)[0]
        input_tokens = self.input_text_processor(input_text)
        enc_output, enc_state = self.encoder(input_tokens)
        dec_state = enc_state
        new_tokens = tf.fill([batch_size, 1], self.start_token)

        result_tokens = []
        attention = []
        done = tf.zeros([batch_size, 1], tf.bool)
        for _ in range(max_length):
            dec_input = DecoderInput(new_tokens= new_tokens, enc_output= enc_output, mask=(input_tokens!=0))
            dec_result, dec_state = self.decoder(dec_input, state=dec_state)
            attention.append(dec_result.attention_weights)

            new_tokens = self.sample(dec_result.logits, temperature)
            done = done | (new_tokens == self.end_token)
            new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)
            result_tokens.append(new_tokens)
            if tf.executing_eagerly() and tf.reduce_all(done):
                break
        result_tokens = tf.concat(result_tokens, axis=-1)
        result_text = self.token_to_text(result_tokens)

        if return_attention:
            attention_stack = tf.concat(attention, axis=-1)
            return {'text':result_text, 'attention':attention_stack}
        else:
            return {'text':result_text}

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def tf_translate(self, input_text):
        return self.translate(input_text)


def train_auto():
    src, tgt = read_data("/home/page/workspace/deeplab/data/spa-eng/spa.txt")
    embedding_dim = 256
    units = 1024
    max_vocab_size = 5000

    print(src[-10], tgt[-1])
    dataset = to_dataset(src, tgt)

    src_text_processor = TextVectorization(standardize=lower_and_split_punct, max_tokens=max_vocab_size)
    src_text_processor.adapt(src)
    print(src_text_processor.get_vocabulary()[0:10])

    tgt_text_processor = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=lower_and_split_punct,
        max_tokens=max_vocab_size
    )
    tgt_text_processor.adapt(tgt)

    print(np.log(tgt_text_processor.vocabulary_size()))
    translator_train = TrainTranslator(embedding_dim, units, src_text_processor, tgt_text_processor)
    translator_train.compile(optimizer=tf.optimizers.Adam(), loss=MaskedLoss())

    batch_loss = BatchLogs('batch_loss')
    translator_train.fit(dataset, epochs=5, callbacks=[batch_loss])

    # inference
    translator = Translator(encoder=translator_train.encoder, decoder=translator_train.decoder,
                            input_text_processor=src_text_processor, output_text_processor=tgt_text_processor)
    example_output_tokens = tf.random.uniform(
        shape=[5, 2], minval=0, dtype=tf.int64,
        maxval=tgt_text_processor.vocabulary_size()
    )
    print(translator.token_to_text(example_output_tokens).numpy())
    example_logits = tf.random.normal([5, 1, tgt_text_processor.vocabulary_size()])
    example_output_tokens = translator.sample(example_logits, temperature=1.0)
    print(example_output_tokens)

    input_text = tf.constant([
        'hace mucho frio aqui.',  # "It's really cold here."
        'Esta es mi vida.',  # "This is my life.""
    ])
    result = translator.translate(input_text)
    print(result['text'][0].numpy().decode())
    print(result['text'][1].numpy().decode())
    a = result['attention'][0]
    plt.bar(range(len(a[0, :])), a[0, :])
    plt.show()

    plt.imshow(np.array(a), vmin=0.0)

    tf.saved_model.save(translator, 'translator',
                        signatures={'serving_default': translator.tf_translate})


def process():
    embedding_dim = 256
    units = 1024
    src, tgt = read_data("/home/page/workspace/deeplab/data/spa-eng/spa.txt")
    print(src[-10], tgt[-1])
    dataset = to_dataset(src, tgt)
    for src_batch, tgt_batch in dataset.take(1):
        print("src_batch[:5] : ", src_batch[:5])
        print()
        print(tgt_batch[:5])
        print()
        print(src_batch[5].numpy().decode())
        print(lower_and_split_punct(src_batch[5]).numpy().decode())
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

    for src_batch, tgt_batch in dataset.take(1):
        print("src_batch[:5] : ", src_batch[:5])
        print()
        print(tgt_batch[:5])
        print()
        print(src_batch[5].numpy().decode())
        print(lower_and_split_punct(src_batch[5]).numpy().decode())
        example_src_tokens = src_text_processor(src_batch)

        encoder = Encoder(src_text_processor.vocabulary_size(), embedding_dim, units)
        example_enc_output, example_enc_state = encoder(example_src_tokens)
        print(f'Input batch, shape (batch): {src_batch.shape}')
        print(f'Input batch tokens, shape (batch, s): {example_src_tokens.shape}')
        print(f'Encoder output, shape (batch, s, units): {example_enc_output.shape}')
        print(f'Encoder state, shape (batch, units): {example_enc_state.shape}')

        # print(example_src_tokens[0:3, 0:10])
        # input_vocab = np.array(src_text_processor.get_vocabulary())
        # tokens = input_vocab[example_src_tokens[0].numpy()]
        # print(" ".join(tokens))
        #
        # attention_layers = BahdanauAttention(units)
        # example_attention_query = tf.random.normal(shape= [len(example_src_tokens), 2, 10])
        # context_vector, attention_weight = attention_layers(
        #     query = example_attention_query,
        #     value = example_enc_output,
        #     mask = (example_src_tokens != 0)
        # )
        # print(f'attention vector shape(batch_size, query_seq_length, units) {context_vector.shape}')
        # print(f'attention weight shape(batch_size, query_seq_length, value_seq_length) {attention_weight.shape}')

        # decoder = Decoder(tgt_text_processor.vocabulary_size(), embedding_dim, units)
        # example_tgt_tokens = tgt_text_processor(tgt_batch)
        # start_index = tgt_text_processor.get_vocabulary().index('[START]')
        # first_token = tf.constant([[start_index]] * example_tgt_tokens.shape[0])
        # dec_res, dec_state = decoder(
        #     inputs = DecoderInput(
        #         new_tokens= first_token,
        #         enc_output=example_enc_output,
        #         mask=(example_src_tokens!=0)
        #     ),
        #     state = example_enc_state
        # )
        # print(f'logits shape(batch_size, t, output_vocab_size) {dec_res.logits.shape}')
        # print(f'state shape(batch_size, dec_units) {dec_state.shape}')
        #
        # sampled_token = tf.random.categorical(dec_res.logits[:,0,:], num_samples=1)
        # vocab = np.array(tgt_text_processor.get_vocabulary())
        # first_word = vocab[sampled_token.numpy()]
        # print(first_word[:5])
        #
        # dec_res, dec_state = decoder(
        #     inputs = DecoderInput(
        #         new_tokens= sampled_token,
        #         enc_output=example_enc_output,
        #         mask=(example_src_tokens!=0)
        #     ),
        #     state = dec_state
        # )

        # sampled_token = tf.random.categorical(dec_res.logits[:, 0, :], num_samples=1)
        # vocab = np.array(tgt_text_processor.get_vocabulary())
        # first_word = vocab[sampled_token.numpy()]
        # print(first_word[:5])

        # plt.subplot(1,2,1)
        # plt.pcolormesh(example_src_tokens)
        # plt.title('token ids')
        # plt.subplot(1,2,2)
        # plt.pcolormesh(example_src_tokens != 0)
        # plt.title('mask')
        # plt.show()
    print(np.log(tgt_text_processor.vocabulary_size()))
    translator_train = TrainTranslator(embedding_dim, units, src_text_processor, tgt_text_processor)
    translator_train.compile(optimizer=tf.optimizers.Adam(), loss=MaskedLoss())

    batch_loss = BatchLogs('batch_loss')
    translator_train.fit(dataset, epochs=5, callbacks=[batch_loss])

    # plt.plot(batch_loss.logs)
    # plt.ylim([0, 3])
    # plt.xlabel('Batch #')
    # plt.ylabel('CE/token')
        # start = time.time()
        # for n in range(10):
        #     print(translator.train_step([src_batch, tgt_batch]))
        # print("time interval:",time.time() - start)
        # translator.train_step([src_batch, tgt_batch])
        #
        # start = time.time()
        # for n in range(10):
        #     print(translator.train_step([src_batch, tgt_batch]))
        # print("time interval2:", time.time() - start)

        # losses = []
        # for n in range(100):
        #     print('.', end='')
        #     logs = translator.train_step([src_batch, tgt_batch])
        #     losses.append(logs['batch_loss'].numpy())
        # plt.plot(losses)
        # plt.show()
    translator = Translator(encoder=translator_train.encoder, decoder=translator_train.decoder,
                            input_text_processor=src_text_processor, output_text_processor=tgt_text_processor)
    example_output_tokens = tf.random.uniform(
        shape=[5,2], minval=0, dtype=tf.int64,
        maxval=tgt_text_processor.vocabulary_size()
    )
    print(translator.token_to_text(example_output_tokens).numpy())
    example_logits = tf.random.normal([5, 1, tgt_text_processor.vocabulary_size()])
    example_output_tokens = translator.sample(example_logits, temperature=1.0)
    print(example_output_tokens)

    input_text = tf.constant([
        'hace mucho frio aqui.',  # "It's really cold here."
        'Esta es mi vida.',  # "This is my life.""
    ])
    result = translator.translate(input_text)
    print(result['text'][0].numpy().decode())
    print(result['text'][1].numpy().decode())
    a = result['attention'][0]
    plt.bar(range(len(a[0,:])), a[0,:])
    plt.show()

    plt.imshow(np.array(a), vmin=0.0)

    tf.saved_model.save(translator, 'translator',
                        signatures={'serving_default':translator.tf_translate})


def main():
    # process()
    reloaded = tf.saved_model.load('translator')
    three_input_text = tf.constant([
        # This is my life.
        'Esta es mi vida.',
        # Are they still home?
        '¿Todavía están en casa?',
        # Try to find out.'
        'Tratar de descubrir.',
        'Los chinos no engañan a los chinos.',
    ])
    three_result = reloaded.tf_translate(three_input_text)
    for tr in three_result['text']:
        print(tr.numpy().decode())


if __name__ == '__main__':
    main()

