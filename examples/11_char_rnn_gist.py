""" A clean, no_frills character-level generative language model.
Created by Danijar Hafner (danijar.com), edited by Chip Huyen
for the class CS 20SI: "TensorFlow for Deep Learning Research"

Based on Andrej Karpathy's blog: 
http://karpathy.github.io/2015/05/21/rnn-effectiveness/
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.append('..')

import time

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import utils

DATA_PATH = 'data/arvix_abstracts.txt'
HIDDEN_SIZE = 200
BATCH_SIZE = 64
NUM_STEPS = 50
SKIP_STEP = 40
TEMPRATURE = 0.7
LR = 0.003
LEN_GENERATED = 300

def vocab_encode(text, vocab):
    return [vocab.index(x) + 1 for x in text if x in vocab]

def vocab_decode(array, vocab):
    return ''.join([vocab[x - 1] for x in array])

def read_data(filename, vocab, window=NUM_STEPS, overlap=NUM_STEPS//2):
    for text in open(filename):
        text = vocab_encode(text, vocab)
        for start in range(0, len(text) - window, overlap):
            chunk = text[start: start + window]
            chunk += [0] * (window - len(chunk))
            yield chunk

def read_batch(stream, batch_size=BATCH_SIZE):
    batch = []
    for element in stream:
        batch.append(element)
        if len(batch) == batch_size:
            yield batch
            batch = []
    yield batch

def create_rnn(seq, hidden_size=HIDDEN_SIZE):
    cell = tf.contrib.rnn.GRUCell(hidden_size)
    in_state = tf.placeholder_with_default(
            cell.zero_state(tf.shape(seq)[0], tf.float32), [None, hidden_size])
    # this line to calculate the real length of seq
    # all seq are padded to be of the same length which is NUM_STEPS
    length = tf.reduce_sum(tf.reduce_max(tf.sign(seq), 2), 1)

    output, out_state = tf.nn.dynamic_rnn(cell, seq, length, in_state)
    # output is [BATCH_SIZE, NUM_STEPS, HIDDEN_SIZE]
    print("shape of output is {}".format(output.shape))
    return output, in_state, out_state, length

def create_model(seq, temp, vocab, hidden=HIDDEN_SIZE):
    # seq is [BATCH_SIZE, NUM_STEPS, len(vocab)]
    seq = tf.one_hot(seq, len(vocab))
    output, in_state, out_state, length = create_rnn(seq, hidden)
    # fully_connected is syntactic sugar for tf.matmul(w, output) + b
    # it will create w and b for us
    # output is [BATCH_SIZE, NUM_STEPS, HIDDEN_SIZE]
    # w is [HIDDEN_SIZE, len(vocab)]
    logits = tf.contrib.layers.fully_connected(output, len(vocab), None)
    # logits is [BATCH_SIZE, NUM_STEPS, len(vocab)]
    print("shape of logits is {}".format(logits.shape))
    # compare with itself(right shift one word)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits[:, :-1], labels=seq[:, 1:]))
    # sample the next character from Maxwell-Boltzmann Distribution with temperature temp
    # it works equally well without tf.exp
    # samle is [BATCH_SIZE, 1]
    sample = tf.multinomial(tf.exp(logits[:, -1] / temp), 1)[:, 0]

    return loss, sample, in_state, out_state, length, logits, seq

def training(vocab, seq, loss, optimizer, global_step, temp, sample, in_state, out_state, length, logits, seq_hot):
    saver = tf.train.Saver()
    start = time.time()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('graphs/gist', sess.graph)
        sess.run(tf.global_variables_initializer())
        
        # ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/arvix/checkpoint'))
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        
        # iteration = global_step.eval()
        iteration = 0
        for batch in read_batch(read_data(DATA_PATH, vocab)):
            # if iteration == 1:
            #     break

            batch_loss, _, seq_length, logits_, seq_hot_ = sess.run([loss, optimizer, length, logits, seq_hot], {seq: batch})
            # print('Iter {} \nlogits \n{}  \nLoss \n{} \nTime \n{} \nseq_hot \n{}'.format(iteration, logits_[:, :-1], batch_loss, time.time() - start, seq_hot_[:, 1:]))
            # online_inference(sess, vocab, seq, sample, temp, in_state, out_state)
            start = time.time()
            if (iteration + 1) % SKIP_STEP == 0:
                print('Iter {} \n Loss {}. Time {}'.format(iteration, batch_loss, time.time() - start))
                online_inference(sess, vocab, seq, sample, temp, in_state, out_state)
                start = time.time()
                #saver.save(sess, 'checkpoints/arvix/char-rnn', iteration)

            iteration += 1


def online_inference(sess, vocab, seq, sample, temp, in_state, out_state, seed='R'):
    """ Generate sequence one character at a time, based on the previous character
    """
    sentence = seed
    state = None
    for _ in range(LEN_GENERATED):
        batch = [vocab_encode(sentence[-1], vocab)]
        feed = {seq: batch, temp: TEMPRATURE}
        # for the first decoder step, the state is None
        if state is not None:
            feed.update({in_state: state})
        index, state = sess.run([sample, out_state], feed)
        sentence += vocab_decode(index, vocab)
    print(sentence)

def main():
    vocab = (
            " $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "\\^_abcdefghijklmnopqrstuvwxyz{|}")
    print("length of vocab is {}".format(len(vocab)))
    seq = tf.placeholder(tf.int32, [None, None])
    temp = tf.placeholder(tf.float32)
    loss, sample, in_state, out_state, length, logits, seq_hot = create_model(seq, temp, vocab)
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(LR).minimize(loss, global_step=global_step)
    utils.make_dir('checkpoints')
    utils.make_dir('checkpoints/arvix')
    training(vocab, seq, loss, optimizer, global_step, temp, sample, in_state, out_state, length, logits, seq_hot)
    
if __name__ == '__main__':
    main()