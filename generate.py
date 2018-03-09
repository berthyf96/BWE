from __future__ import print_function
import argparse
from datetime import datetime
import json
import os
import sys
import time

import librosa
import numpy as np
import scipy
import tensorflow as tf
from tensorflow.python.client import timeline
from hrnn import HRNNModel, AudioReader, mu_law_decode, optimizer_factory

from tensorflow.python.tools import inspect_checkpoint as chkp

RATIO = 2
NB_DATA_DIRECTORY = './data_abr/nb'
WB_DATA_DIRECTORY = './data_abr/wb'
LOGDIR_ROOT = './logdir_2'
CHECKPOINT_EVERY = 20
NUM_STEPS = int(1e5)
LEARNING_RATE = 1e-3
SAMPLE_SIZE = 100000
L2_REG_STRENGTH = 0
SILENCE_THRESHOLD = 0.3
MOMENTUM = 0.9
MAX_TO_KEEP = 5

N_SECS = 3
SAMPLE_RATE = 16000
SAMPLE_SIZE = N_SECS*SAMPLE_RATE

BATCH_SIZE = 1
NUM_GPU = 1
WAV_OUT_PATH = './test'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=NUM_GPU)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--nb_data_dir',      type=str,   default=NB_DATA_DIRECTORY)
    parser.add_argument('--wb_data_dir',      type=str,   default=WB_DATA_DIRECTORY)
    parser.add_argument('--logdir_root',      type=str,   default=LOGDIR_ROOT)
    parser.add_argument('--checkpoint_every', type=int,   default=CHECKPOINT_EVERY)
    parser.add_argument('--num_steps',        type=int,   default=NUM_STEPS)
    parser.add_argument('--learning_rate',    type=float, default=LEARNING_RATE)
    parser.add_argument('--sample_size',      type=int,   default=SAMPLE_SIZE)
    parser.add_argument('--sample_rate',      type=int,   default=SAMPLE_RATE)
    parser.add_argument('--l2_reg_strength', type=float, default=L2_REG_STRENGTH)
    parser.add_argument('--momentum', type=float, default=MOMENTUM)
    parser.add_argument('--silence_threshold',type=float, default=SILENCE_THRESHOLD)
    parser.add_argument('--optimizer',        type=str,   default='adam', choices=optimizer_factory.keys())
    parser.add_argument('--seq_len',          type=int, required=True)
    parser.add_argument('--big_frame_size',   type=int, required=True)
    parser.add_argument('--frame_size',       type=int, required=True)
    parser.add_argument('--q_levels',         type=int, required=True)
    parser.add_argument('--dim',              type=int, required=True)
    parser.add_argument('--n_rnn',            type=int, choices=xrange(1,6), required=True)
    parser.add_argument('--emb_size',         type=int, required=True)
    parser.add_argument('--rnn_type', choices=['LSTM', 'GRU'], required=True)
    parser.add_argument('--max_checkpoints',  type=int, default=MAX_TO_KEEP)
    return parser.parse_args()


def load(saver, sess, logdir):
    ckpt = tf.train.get_checkpoint_state(logdir)
    #chkp.print_tensors_in_checkpoint_file(
        #ckpt.model_checkpoint_path, tensor_name='', all_tensors=True)
    
    if ckpt:
        print("Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("Restoring global step: {}".format(global_step))
        saver.restore(sess, ckpt.model_checkpoint_path)
        return global_step
    else:
        print("No checkpoint found")
    return None


def create_model(args):
    net = HRNNModel(batch_size=args.batch_size,
                    big_frame_size=args.big_frame_size,
                    frame_size=args.frame_size,
                    q_levels=args.q_levels,
                    rnn_type=args.rnn_type,
                    dim=args.dim,
                    n_rnn=args.n_rnn,
                    seq_len=args.seq_len,
                    emb_size=args.emb_size)
    return net

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
    
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))
    

def create_gen_wav_para(net):
    with tf.name_scope('infe_para'):
        infe_para = dict()
        # Big frame
        infe_para['infe_big_frame_inp'] = tf.get_variable(
            'infe_big_frame_inp',
            [net.batch_size, net.big_frame_size, 1],
            dtype=tf.float32)
        infe_para['infe_big_frame_outp'] = tf.get_variable(
            'infe_big_frame_outp',
            [net.batch_size, net.big_frame_size/net.frame_size, net.dim],
            dtype=tf.float32)
        # Frame
        infe_para['infe_big_frame_outp_slices'] = tf.get_variable(
            'infe_big_frame_outp_slices',
            [net.batch_size, 1, net.dim],
            dtype=tf.float32)
        infe_para['infe_frame_inp'] = tf.get_variable(
            'infe_frame_inp',
            [net.batch_size, net.frame_size, 1],
            dtype=tf.float32)
        infe_para['infe_frame_outp'] = tf.get_variable(
            'infe_frame_outp',
            [net.batch_size, net.frame_size, net.dim],
            dtype=tf.float32)
        # Sample
        infe_para['infe_frame_outp_slices'] = tf.get_variable(
            'infe_frame_outp_slices',
            [net.batch_size, 1, net.dim],
            dtype=tf.float32)
        infe_para['infe_sample_inp'] = tf.get_variable(
            'infe_sample_inp',
            [net.batch_size, net.frame_size, 1],
            dtype=tf.int32)
        # States
        infe_para['infe_big_frame_state'] = net.big_cell.zero_state(
            net.batch_size, tf.float32)
        infe_para['infe_frame_state'] = net.cell.zero_state(
            net.batch_size, tf.float32)

        tf.get_variable_scope().reuse_variables()

        # Outputs
        infe_para['infe_big_frame_outp'], infe_para['infe_final_big_frame_state'] = \
            net._create_network_BigFrame(num_steps=1,
                big_frame_state=infe_para['infe_big_frame_state'],
                big_input_sequences=infe_para['infe_big_frame_inp'])
        infe_para['infe_frame_outp'], infe_para['infe_final_frame_state'] = \
            net._create_network_Frame(num_steps=1,
                big_frame_outputs=infe_para['infe_big_frame_outp_slices'],
                frame_state=infe_para['infe_frame_state'],
                input_sequences=infe_para['infe_frame_inp'])

        sample_out = net._create_network_Sample(
            frame_outputs=infe_para['infe_frame_outp_slices'],
            sample_input_sequences=infe_para['infe_sample_inp'])
        sample_out = tf.reshape(sample_out, [-1, net.q_levels])
        infe_para['infe_sample_outp'] = tf.cast(
            tf.nn.softmax(tf.cast(sample_out, tf.float64)), tf.float32)
        infe_para['infe_sample_decode_inp'] = tf.placeholder(tf.int32)
        infe_para['infe_decode'] = mu_law_decode(
            infe_para['infe_sample_decode_inp'], net.q_levels)
        
        return infe_para


def generate_and_save_samples(net, infe_para, sess, nb_inputs, wb_inputs):
    nb_samples = np.reshape(nb_inputs, (net.batch_size, SAMPLE_SIZE, 1)).astype(
        'int32')
    nb_samples[:, :net.big_frame_size, :] = np.int32(net.q_levels//2)

    wb_samples = np.reshape(wb_inputs, (net.batch_size, SAMPLE_SIZE, 1))

    
    final_big_s, final_s = \
        sess.run([net.big_initial_state, net.initial_state])
    big_frame_out = None
    frame_out = None
    sample_out = None

    output_samples_batch = np.zeros(
        (net.batch_size, SAMPLE_SIZE, 1), dtype='int32')

    for t in range(SAMPLE_SIZE):
        # Copy over samples from WB input, skipping by RATIO
        if  (t % RATIO == 0):
            for i in range(net.batch_size):
                output_samples_batch[i, t, :] = wb_samples[i, t, :]
                
    for t in xrange(net.big_frame_size, SAMPLE_SIZE):
        # Big frame
        if t % net.big_frame_size == 0:
            big_input_sequences = nb_samples[:, t-net.big_frame_size:t,:].astype(
                'float32')
            big_frame_out, final_big_s = sess.run(
                [infe_para['infe_big_frame_outp'],
                 infe_para['infe_final_big_frame_state']],
                feed_dict={
                    infe_para['infe_big_frame_inp'] : big_input_sequences,
                    infe_para['infe_big_frame_state'] : final_big_s})
        # Frame
        if t % net.frame_size == 0:
            frame_input_sequences = nb_samples[:, t-net.frame_size:t,:].astype(
                'float32')
            big_frame_outp_idx = \
                (t/net.frame_size)%(net.big_frame_size/net.frame_size)
            big_frame_outp_slices = big_frame_out[:, [big_frame_outp_idx], :]
            frame_out, final_s = sess.run(
                [infe_para['infe_frame_outp'],
                 infe_para['infe_final_frame_state']],
                feed_dict={
                    infe_para['infe_big_frame_outp_slices'] : big_frame_outp_slices,
                    infe_para['infe_frame_inp'] : frame_input_sequences,
                    infe_para['infe_frame_state'] : final_s})
        # Sample
        sample_input_sequences = nb_samples[:, t-net.frame_size:t,:]
        frame_outp_idx = t % net.frame_size
        frame_outp_slices = frame_out[:, [frame_outp_idx], :]
        sample_out = sess.run(
            infe_para['infe_sample_outp'],
            feed_dict={
                infe_para['infe_frame_outp_slices'] : frame_outp_slices,
                infe_para['infe_sample_inp'] : sample_input_sequences})

        #print(sample_out)
        sample_out_batch = [] # Will contain one sample for each item in batch
        for row in sample_out:
            #print(np.argmax(row))
            #print(np.amax(row))
            sample_out_batch.append(np.argmax(row))
        for (i, samp) in enumerate(sample_out_batch):
            if t % 2 == 1:
                output_samples_batch[i, t, :] = samp

    for i in range(0, net.batch_size):
        inp = output_samples_batch[i, :, :].reshape([-1,1]).tolist()
        out = sess.run(infe_para['infe_decode'],
                       feed_dict={infe_para['infe_sample_decode_inp'] : inp})
        outpath = './test_'+str(i)+'.wav'
        write_wav(out, SAMPLE_RATE, outpath)
        if i >= 10: break
    '''
    for i in range(0, net.batch_size):
        out = wb_samples[i]
        outpath = './test_'+'_'+str(i)+'.wav'
        print(out)
        write_wav(out, SAMPLE_RATE, outpath)
        if i >= 10: break
    '''
    
def main():
    args = get_args()
    logdir = os.path.join(args.logdir_root, 'train')
    coord = tf.train.Coordinator()

    with tf.name_scope('create_inputs'):
        # Reader
        reader = AudioReader(args.nb_data_dir,
                             args.wb_data_dir,
                             coord,
                             sample_rate=args.sample_rate,
                             sample_size=args.sample_size,
                             silence_threshold=args.silence_threshold)
        nb_audio_batch, wb_audio_batch = reader.dequeue(args.batch_size)
        
    net = create_model(args)

    global_step = tf.get_variable('global_step', [],
                                  initializer = tf.constant_initializer(0),
                                  trainable=False)
    optim = optimizer_factory[args.optimizer](learning_rate=args.learning_rate,
                                              momentum=args.momentum)
    
    # Set up placeholders and variables on each GPU
    tower_grads = []
    losses = []
    wb_input_batch_rnn = []
    nb_input_batch_rnn = []
    train_big_frame_state = []
    train_frame_state = []
    final_big_frame_state = []
    final_frame_state = []

    for i in xrange(args.num_gpus):
        with tf.device('/gpu:%d' % (i)):
            nb_input_batch_rnn.append(
                tf.Variable(tf.zeros([net.batch_size, net.seq_len, 1]),
                            trainable=False,
                            name='nb_input_batch_rnn',
                            dtype=tf.float32))
            wb_input_batch_rnn.append(
                tf.Variable(tf.zeros([net.batch_size, net.seq_len, 1]),
                            trainable=False,
                            name='wb_input_batch_rnn',
                            dtype=tf.float32))
            # Create initial states
            train_big_frame_state.append(
                net.big_cell.zero_state(net.batch_size, tf.float32))
            final_big_frame_state.append(
                net.big_cell.zero_state(net.batch_size, tf.float32))
            train_frame_state.append(
                net.cell.zero_state(net.batch_size, tf.float32))
            final_frame_state.append(
                net.cell.zero_state(net.batch_size, tf.float32))

    with tf.variable_scope(tf.get_variable_scope()):
        for i in xrange(args.num_gpus):
            with tf.device('/gpu:%d' % (i)):
                with tf.name_scope('TOWER_%d' % i) as scope:
                    # Create variables
                    print("Creating model on GPU:%d" % i)
                    loss, final_big_frame_state[i], final_frame_state[i] = \
                        net.loss_HRNN(nb_input_batch_rnn[i],
                                      wb_input_batch_rnn[i],
                                      train_big_frame_state[i],
                                      train_frame_state[i],
                                      l2_reg_strength=args.l2_reg_strength)
                    tf.get_variable_scope().reuse_variables()
                    losses.append(loss)
                    trainable = tf.trainable_variables()
                    gradients = optim.compute_gradients(
                        loss, trainable,
                        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
                    tower_grads.append(gradients)
    grad_vars = average_gradients(tower_grads)
    grads, vars = zip(*grad_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, 5.0)
    grad_vars = zip(grads_clipped, vars)
    
    apply_gradient_op = optim.apply_gradients(grad_vars, global_step=global_step)

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(var_list=tf.trainable_variables(),
                           max_to_keep=args.max_checkpoints)
    
    try:
        saved_global_step = load(saver, sess, logdir)
        if saved_global_step is None: saved_global_step = -1
    except:
        print("Something went wrong while restoring checkpoint.")
        raise

    # Start queue runners
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    infe_para = create_gen_wav_para(net)
    nb_inputs, wb_inputs = sess.run([nb_audio_batch, wb_audio_batch])

    generate_and_save_samples(net, infe_para, sess, nb_inputs, wb_inputs)
    

if __name__ == '__main__':
    main()
