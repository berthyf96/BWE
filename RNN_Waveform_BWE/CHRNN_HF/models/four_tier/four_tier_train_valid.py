from time import time
from datetime import datetime
print "Experiment started at:", datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M')
exp_start = time()

import os, sys, glob
sys.path.insert(1, os.getcwd())
import argparse
import itertools

import numpy
numpy.random.seed(123)
np = numpy
import random
random.seed(123)

import theano
import theano.tensor as T
import theano.ifelse
import lasagne
import scipy.io.wavfile

import lib

LEARNING_RATE = 0.001

### Parsing passed args/hyperparameters ###
def get_args():
    def t_or_f(arg):
        ua = str(arg).upper()
        if 'TRUE'.startswith(ua):
            return True
        elif 'FALSE'.startswith(ua):
            return False
        else:
           raise ValueError('Arg is neither `True` nor `False`')

    def check_non_negative(value):
        ivalue = int(value)
        if ivalue < 0:
             raise argparse.ArgumentTypeError("%s is not non-negative!" % value)
        return ivalue

    def check_positive(value):
        ivalue = int(value)
        if ivalue < 1:
             raise argparse.ArgumentTypeError("%s is not positive!" % value)
        return ivalue

    def check_unit_interval(value):
        fvalue = float(value)
        if fvalue < 0 or fvalue > 1:
             raise argparse.ArgumentTypeError("%s is not in [0, 1] interval!" % value)
        return fvalue

    # No default value here. Indicate every single arguement.
    parser = argparse.ArgumentParser(
        description='three_tier.py\nNo default value! Indicate every argument.')

    # TODO: Fix the descriptions
    # Hyperparameter arguements:
    parser.add_argument('--exp', help='Experiment name',
            type=str, required=False, default='_')
    parser.add_argument('--seq_len', help='How many samples to include in each Truncated BPTT pass', type=check_positive, required=True)
    parser.add_argument('--con_dim', help='Condition dimension',\
            type=check_positive, required=True)
    parser.add_argument('--con_frame_size', help='How many samples per condition frame',\
            type=check_positive, required=True)
    parser.add_argument('--big_frame_size', help='How many samples per big frame',\
            type=check_positive, required=True)
    parser.add_argument('--frame_size', help='How many samples per frame',\
            type=check_positive, required=True)
    parser.add_argument('--weight_norm', help='Adding learnable weight normalization to all the linear layers (except for the embedding layer)',\
            type=t_or_f, required=True)
    parser.add_argument('--emb_size', help='Size of embedding layer (> 0)',
            type=check_positive, required=True)  # different than two_tier
    parser.add_argument('--skip_conn', help='Add skip connections to RNN',
            type=t_or_f, required=True)
    parser.add_argument('--dim', help='Dimension of RNN and MLPs',\
            type=check_positive, required=True)
    parser.add_argument('--n_rnn', help='Number of layers in the stacked RNN',
            type=check_positive, choices=xrange(1,6), required=True)
    parser.add_argument('--rnn_type', help='GRU or LSTM', choices=['LSTM', 'GRU'],\
            required=True)
    parser.add_argument('--learn_h0', help='Whether to learn the initial state of RNN',\
            type=t_or_f, required=True)
    parser.add_argument('--q_levels', help='Number of bins for quantization of audio samples. Should be 256 for mu-law.',\
            type=check_positive, required=True)
    parser.add_argument('--q_type', help='Quantization in linear-scale, a-law-companding, or mu-law compandig. With mu-/a-law quantization level shoud be set as 256',\
            choices=['linear', 'a-law', 'mu-law'], required=True)
    parser.add_argument('--which_set', help='ONOM, BLIZZ, MUSIC, or HUCK',
            choices=['yp1000','ONOM', 'BLIZZ', 'MUSIC', 'HUCK','TIMIT'], required=True)
    parser.add_argument('--batch_size', help='size of mini-batch',
            type=check_positive, choices=[50,64, 128, 256], required=True)

    parser.add_argument('--resume', help='Resume the same model from the last checkpoint. Order of params are important. [for now]',\
            required=False, default=False, action='store_true')

    args = parser.parse_args()

    # NEW
    # Create tag for this experiment based on passed args
    tag='four_tier_model'
    print "Created experiment tag for these args:"
    print tag

    return args, tag

#tag:three_tier.py-expAXIS1-seq_len512-big_frame_size8-frame_size2-weight_normT-emb_size64-skip_connF-dim32-n_rnn2-rnn_typeLSTM-learn_h0F-q_levels16-q_typelinear-batch_size128-which_setMUSIC-lr0.001
args, tag = get_args()

SEQ_LEN = args.seq_len # How many samples to include in each truncated BPTT pass (512)
#print "------------------previous SEQ_LEN:", SEQ_LEN
# TODO: test incremental training
#SEQ_LEN = 512 + 256
#print "---------------------------new SEQ_LEN:", SEQ_LEN
CON_DIM=args.con_dim
CON_FRAME_SIZE=args.con_frame_size
BIG_FRAME_SIZE = args.big_frame_size # how many samples per big frame
FRAME_SIZE = args.frame_size # How many samples per frame
WEIGHT_NORM = args.weight_norm #True
EMB_SIZE = args.emb_size #(256)
SKIP_CONN = args.skip_conn #(False)
DIM = args.dim # Model dimensionality. (1024)
BIG_DIM = DIM # Dimensionality for the slowest level. (1024)
CON_TIER_DIM=DIM
N_RNN = args.n_rnn # How many RNNs to stack in the frame-level model (1)
N_BIG_RNN = N_RNN # how many RNNs to stack in the big-frame-level model (1)
N_CON_RNN=N_RNN
RNN_TYPE = args.rnn_type #GRU
H0_MULT = 2 if RNN_TYPE == 'LSTM' else 1 #(1)
LEARN_H0 = args.learn_h0 #(True)
Q_LEVELS = args.q_levels # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization #(256)
Q_TYPE = args.q_type # log- or linear-scale #(linear)
WHICH_SET = args.which_set #(MUSIC)
BATCH_SIZE = args.batch_size #(128)
RESUME = args.resume #(False)
assert SEQ_LEN % CON_FRAME_SIZE == 0,\
    'seq_len should be divisible by con_frame_size'
assert CON_FRAME_SIZE % BIG_FRAME_SIZE == 0,\
    'con_frame_size should be divisible by big_frame_size'
assert BIG_FRAME_SIZE % FRAME_SIZE == 0,\
    'big_frame_size should be divisible by frame_size'

if Q_TYPE == 'mu-law' and Q_LEVELS != 256:
    raise ValueError('For mu-law Quantization levels should be exactly 256!')

# Fixed hyperparams
GRAD_CLIP = 1 # Elementwise grad clip threshold
BITRATE = 16000

# Other constants
TRAIN_MODE = 'iters' # To use PRINT_ITERS and STOP_ITERS
#TRAIN_MODE = 'time' # To use PRINT_TIME and STOP_TIME
#TRAIN_MODE = 'time-iters'
# To use PRINT_TIME for validation,
# and (STOP_ITERS, STOP_TIME), whichever happened first, for stopping exp.
#TRAIN_MODE = 'iters-time'
# To use PRINT_ITERS for validation,
# and (STOP_ITERS, STOP_TIME), whichever happened first, for stopping exp.
PRINT_ITERS = 5000 # Print cost, generate samples, save model checkpoint every N iterations.
STOP_ITERS = 300000 # Stop after this many iterations
PRINT_TIME = 2*60 # Print cost, generate samples, save model checkpoint every N seconds.
STOP_TIME = 60*60*24*7 # Stop after this many seconds of actual training (not including time req'd to generate samples etc.)
N_SEQS = 5  # Number of samples to generate every time monitoring.
RESULTS_DIR = 'results_4t'
FOLDER_PREFIX = os.path.join(RESULTS_DIR, tag)
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude
OVERLAP = BIG_FRAME_SIZE

epoch_str = 'epoch'
iter_str = 'iter'
lowest_valid_str = 'lowest valid cost'
corresp_test_str = 'correponding test cost'
train_nll_str, valid_nll_str, test_nll_str = \
    'train NLL (bits)', 'valid NLL (bits)', 'test NLL (bits)'

### Create directories ###
#   FOLDER_PREFIX: root, contains:
#       log.txt, __note.txt, train_log.pkl, train_log.png [, model_settings.txt]
#   FOLDER_PREFIX/params: saves all checkpoint params as pkl
#   FOLDER_PREFIX/samples: keeps all checkpoint samples as wav
#   FOLDER_PREFIX/best: keeps the best parameters, samples, ...
if not os.path.exists(FOLDER_PREFIX):
    os.makedirs(FOLDER_PREFIX)
PARAMS_PATH = os.path.join(FOLDER_PREFIX, 'params')
if not os.path.exists(PARAMS_PATH):
    os.makedirs(PARAMS_PATH)
SAMPLES_PATH = os.path.join(FOLDER_PREFIX, 'samples')
if not os.path.exists(SAMPLES_PATH):
    os.makedirs(SAMPLES_PATH)
BEST_PATH = os.path.join(FOLDER_PREFIX, 'best')
if not os.path.exists(BEST_PATH):
    os.makedirs(BEST_PATH)

lib.print_model_settings(locals(), path=FOLDER_PREFIX, sys_arg=True)

### Import the data_feeder ###
# Handling WHICH_SET
if WHICH_SET == 'TIMIT':
    from datasets.dataset import TIMIT_train_feed_epoch as train_feeder
    from datasets.dataset import TIMIT_valid_feed_epoch as valid_feeder
    from datasets.dataset import TIMIT_test_feed_epoch  as test_feeder

def load_data(data_feeder):
    """
    Helper function to deal with interface of different datasets.
    `data_feeder` should be `train_feeder`, `valid_feeder`, or `test_feeder`.
    """
    return data_feeder(BATCH_SIZE,
                       SEQ_LEN,
                       CON_FRAME_SIZE,
                       CON_DIM,
                       OVERLAP,
                       Q_LEVELS,
                       Q_ZERO,
                       Q_TYPE)

### Creating computation graph ###
def con_frame_level_rnn(input_sequences, h0, reset):
    """
    input_sequences.shape: (batch size, n con frames * CON_DIM) 
    h0.shape:              (batch size, N_BIG_RNN, BIG_DIM) #N_BIG_RNN=1,BIG_DIM=1024
    reset.shape:           ()
    output[0].shape:       (batch size, n frames, DIM)
    output[1].shape:       same as h0.shape
    output[2].shape:       (batch size, seq len, Q_LEVELS)
    """

    frames = input_sequences.reshape((
        input_sequences.shape[0],
        input_sequences.shape[1] // CON_DIM,
        CON_DIM
    ))

    # Rescale frames from ints in [0, Q_LEVELS) to floats in [-2, 2]
    # (a reasonable range to pass as inputs to the RNN)
    # frames = (frames.astype('float32') / lib.floatX(Q_LEVELS/2)) - lib.floatX(1)
    # frames *= lib.floatX(2)

    # Initial state of RNNs
    learned_h0 = lib.param(
        'ConFrameLevel.h0',
        numpy.zeros((N_BIG_RNN, H0_MULT*BIG_DIM), dtype=theano.config.floatX)
    )
    # Handling LEARN_H0
    learned_h0.param = LEARN_H0 #True
    learned_h0 = T.alloc(learned_h0, h0.shape[0], N_BIG_RNN, H0_MULT*BIG_DIM) #broadcast according to batch size,H0_MULT=1
    learned_h0 = T.unbroadcast(learned_h0, 0, 1, 2)
    h0 = theano.ifelse.ifelse(reset, learned_h0, h0)   #if reset=1,h0=learned_h0; if reset=0,h0=h0

    # Handling RNN_TYPE
    # Handling SKIP_CONN
    if RNN_TYPE == 'GRU':
        rnns_out, last_hidden = lib.ops.stackedGRU('ConFrameLevel.GRU',
                                                   N_CON_RNN,
                                                   CON_DIM,
                                                   CON_TIER_DIM,
                                                   frames,
                                                   h0=h0,
                                                   weightnorm=WEIGHT_NORM,
                                                   skip_conn=SKIP_CONN)
    elif RNN_TYPE == 'LSTM':
        rnns_out, last_hidden = lib.ops.stackedLSTM('ConFrameLevel.LSTM',
                                                    N_CON_RNN,
                                                    CON_DIM,
                                                    CON_TIER_DIM,
                                                    frames,
                                                    h0=h0,
                                                    weightnorm=WEIGHT_NORM,
                                                    skip_conn=SKIP_CONN)

    output = lib.ops.Linear(       #batch*timestep*dim
        'ConFrameLevel.Output',
        CON_TIER_DIM,
        BIG_DIM * CON_FRAME_SIZE / BIG_FRAME_SIZE,  #1024*8/2
        rnns_out,
        initialization='he',
        weightnorm=WEIGHT_NORM
    )
    output = output.reshape((output.shape[0], output.shape[1] * CON_FRAME_SIZE / BIG_FRAME_SIZE, BIG_DIM))

    return (output, last_hidden) #last_hidden:#batch*1*dim

def big_frame_level_rnn(input_sequences, other_input,h0, reset):
    """
    input_sequences.shape: (batch size, n big frames * BIG_FRAME_SIZE) #BIG_FRAME_SIZE=8
    h0.shape:              (batch size, N_BIG_RNN, BIG_DIM) #N_BIG_RNN=1,BIG_DIM=1024
    reset.shape:           ()
    output[0].shape:       (batch size, n frames, DIM)
    output[1].shape:       same as h0.shape
    output[2].shape:       (batch size, seq len, Q_LEVELS)
    """
    frames = input_sequences.reshape((
        input_sequences.shape[0],
        input_sequences.shape[1] // (2*BIG_FRAME_SIZE),
        2*BIG_FRAME_SIZE
    ))

    # Rescale frames from ints in [0, Q_LEVELS) to floats in [-2, 2]
    # (a reasonable range to pass as inputs to the RNN)
    frames = (frames.astype('float32') / lib.floatX(Q_LEVELS/2)) - lib.floatX(1)
    frames *= lib.floatX(1)

    gru_input = lib.ops.Linear(
        'BigFrameLevel.InputExpand',
        2*BIG_FRAME_SIZE,
        BIG_DIM,
        frames,
        initialization='he',
        weightnorm=WEIGHT_NORM,
        ) + other_input

    # Initial state of RNNs
    learned_h0 = lib.param(
        'BigFrameLevel.h0',
        numpy.zeros((N_BIG_RNN, H0_MULT*BIG_DIM), dtype=theano.config.floatX)
    )
    # Handling LEARN_H0
    learned_h0.param = LEARN_H0 #True
    learned_h0 = T.alloc(learned_h0, h0.shape[0], N_BIG_RNN, H0_MULT*BIG_DIM) #broadcast according to batch size,H0_MULT=1
    learned_h0 = T.unbroadcast(learned_h0, 0, 1, 2)
    h0 = theano.ifelse.ifelse(reset, learned_h0, h0)   #if reset=1,h0=learned_h0; if reset=0,h0=h0

    # Handling RNN_TYPE
    # Handling SKIP_CONN
    if RNN_TYPE == 'GRU':
        rnns_out, last_hidden = lib.ops.stackedGRU('BigFrameLevel.GRU',
                                                   N_BIG_RNN,
                                                   BIG_DIM,
                                                   BIG_DIM,
                                                   gru_input,
                                                   h0=h0,
                                                   weightnorm=WEIGHT_NORM,
                                                   skip_conn=SKIP_CONN)
    elif RNN_TYPE == 'LSTM':
        rnns_out, last_hidden = lib.ops.stackedLSTM('BigFrameLevel.LSTM',
                                                    N_BIG_RNN,
                                                    BIG_DIM,
                                                    BIG_DIM,
                                                    gru_input,
                                                    h0=h0,
                                                    weightnorm=WEIGHT_NORM,
                                                    skip_conn=SKIP_CONN)

    output = lib.ops.Linear(       #batch*timestep*dim
        'BigFrameLevel.Output',
        BIG_DIM,
        DIM * BIG_FRAME_SIZE / FRAME_SIZE,  #1024*8/2
        rnns_out,
        initialization='he',
        weightnorm=WEIGHT_NORM
    )
    output = output.reshape((output.shape[0], output.shape[1] * BIG_FRAME_SIZE / FRAME_SIZE, DIM))

    return (output, last_hidden) #last_hidden:#batch*1*dim

def frame_level_rnn(input_sequences, other_input, h0, reset):
    """
    input_sequences.shape: (batch size, n frames * FRAME_SIZE) #FRAME_SIZE=2
    other_input.shape:     (batch size, n frames, DIM)
    h0.shape:              (batch size, N_RNN, DIM)
    reset.shape:           ()
    output.shape:          (batch size, n frames * FRAME_SIZE, DIM)
    """
    frames = input_sequences.reshape((
        input_sequences.shape[0],
        input_sequences.shape[1] // (2*FRAME_SIZE),
        2*FRAME_SIZE
    ))

    # Rescale frames from ints in [0, Q_LEVELS) to floats in [-2, 2]
    # (a reasonable range to pass as inputs to the RNN)
    frames = (frames.astype('float32') / lib.floatX(Q_LEVELS/2)) - lib.floatX(1)
    frames *= lib.floatX(1)

    gru_input = lib.ops.Linear(
        'FrameLevel.InputExpand',
        2*FRAME_SIZE,
        DIM,
        frames,
        initialization='he',
        weightnorm=WEIGHT_NORM,
        ) + other_input

    # Initial state of RNNs
    learned_h0 = lib.param(
        'FrameLevel.h0',
        numpy.zeros((N_RNN, H0_MULT*DIM), dtype=theano.config.floatX)
    )
    # Handling LEARN_H0
    learned_h0.param = LEARN_H0
    learned_h0 = T.alloc(learned_h0, h0.shape[0], N_RNN, H0_MULT*DIM)
    learned_h0 = T.unbroadcast(learned_h0, 0, 1, 2)
    #learned_h0 = T.patternbroadcast(learned_h0, [False] * learned_h0.ndim)
    h0 = theano.ifelse.ifelse(reset, learned_h0, h0)

    # Handling RNN_TYPE
    # Handling SKIP_CONN
    if RNN_TYPE == 'GRU':
        rnns_out, last_hidden = lib.ops.stackedGRU('FrameLevel.GRU',
                                                   N_RNN,
                                                   DIM,
                                                   DIM,
                                                   gru_input,
                                                   h0=h0,
                                                   weightnorm=WEIGHT_NORM,
                                                   skip_conn=SKIP_CONN)
    elif RNN_TYPE == 'LSTM':
        rnns_out, last_hidden = lib.ops.stackedLSTM('FrameLevel.LSTM',
                                                    N_RNN,
                                                    DIM,
                                                    DIM,
                                                    gru_input,
                                                    h0=h0,
                                                    weightnorm=WEIGHT_NORM,
                                                    skip_conn=SKIP_CONN)

    output = lib.ops.Linear(
        'FrameLevel.Output',
        DIM,
        FRAME_SIZE * DIM,
        rnns_out,
        initialization='he',
        weightnorm=WEIGHT_NORM
    )
    output = output.reshape((output.shape[0], output.shape[1] * FRAME_SIZE, DIM))

    return (output, last_hidden)

def sample_level_predictor(frame_level_outputs, prev_samples):
    """
    frame_level_outputs.shape: (batch size, DIM)
    prev_samples.shape:        (batch size, FRAME_SIZE)
    output.shape:              (batch size, Q_LEVELS)
    """
    # Handling EMB_SIZE
    if EMB_SIZE == 0:  # no support for one-hot in three_tier and one_tier.
        prev_samples = lib.ops.T_one_hot(prev_samples, Q_LEVELS)
        # (BATCH_SIZE*N_FRAMES*FRAME_SIZE, FRAME_SIZE, Q_LEVELS)
        last_out_shape = Q_LEVELS
    elif EMB_SIZE > 0:  #The embedding steps maps each of the q discrete values to a real-valued vector embedding.
        prev_samples = lib.ops.Embedding(  #after embedding, the dim is batch size*FRANME_SIZE*EMB_SIZE
            'SampleLevel.Embedding',
            Q_LEVELS,
            EMB_SIZE,
            prev_samples)
        # (BATCH_SIZE*N_FRAMES*FRAME_SIZE, FRAME_SIZE, EMB_SIZE), f32
        last_out_shape = EMB_SIZE
    else:
        raise ValueError('EMB_SIZE cannot be negative.')

    prev_samples = prev_samples.reshape((-1, FRAME_SIZE * last_out_shape)) #dim:batch size*(FRAME_SIZE*EMB_SIZE)

    out = lib.ops.Linear(
        'SampleLevel.L1_PrevSamples',
        FRAME_SIZE * last_out_shape,
        DIM,
        prev_samples,
        biases=False,
        initialization='he',
        weightnorm=WEIGHT_NORM
    )

    out += frame_level_outputs
    # out = T.nnet.relu(out)  # commented out to be similar to two_tier

    out = lib.ops.Linear('SampleLevel.L2',
                         DIM,
                         DIM,
                         out,
                         initialization='he',
                         weightnorm=WEIGHT_NORM)
    out = T.nnet.relu(out)

    # L3
    out = lib.ops.Linear('SampleLevel.L3',
                         DIM,
                         DIM,
                         out,
                         initialization='he',
                         weightnorm=WEIGHT_NORM)
    out = T.nnet.relu(out)

    # Output
    # We apply the softmax later
    out = lib.ops.Linear('SampleLevel.Output',
                         DIM,
                         Q_LEVELS,
                         out,
                         weightnorm=WEIGHT_NORM)
    return out

sequences_8k   = T.imatrix('sequences_8k') #batch size*samplenum
sequences_up   = T.imatrix('sequences_up')
condition   = T.matrix('con')
con_h0      = T.tensor3('con_h0')
h0          = T.tensor3('h0')     #(batch size, N_RNN, DIM)
big_h0      = T.tensor3('big_h0') #(batch size, N_BIG_RNN, BIG_DIM)
reset       = T.iscalar('reset')
mask        = T.matrix('mask') #batch size*samplenum
batch_size       =T.iscalar('batch_size')
lr=T.scalar('lr')

con_input_sequences = condition

big_input_sequences = sequences_8k #The last BIG_FRAME_SIZE frames do not need (tier3)
big_input_sequences=big_input_sequences.reshape((1, batch_size, 1, -1))
big_input_sequences=T.nnet.neighbours.images2neibs(big_input_sequences, (1,  2*OVERLAP), neib_step=(1, OVERLAP), mode='valid')
big_input_sequences=big_input_sequences.reshape((batch_size,-1))

input_sequences = sequences_8k[:,0:-(OVERLAP-FRAME_SIZE)]  #(tier2)
input_sequences=input_sequences.reshape((1, batch_size, 1, -1))
input_sequences=T.nnet.neighbours.images2neibs(input_sequences, (1,  2*FRAME_SIZE), neib_step=(1, FRAME_SIZE), mode='valid')
input_sequences=input_sequences.reshape((batch_size,-1))
target_sequences = sequences_up[:,0:-OVERLAP] #groundtrues

target_mask = mask[:,0:-OVERLAP]

con_frame_level_outputs, new_con_h0 = con_frame_level_rnn(con_input_sequences,con_h0,reset)

big_frame_level_outputs, new_big_h0 = big_frame_level_rnn(big_input_sequences, con_frame_level_outputs,big_h0, reset)#tier3->tier2

frame_level_outputs, new_h0 = frame_level_rnn(input_sequences, big_frame_level_outputs, h0, reset)#tier2->tier1

prev_samples = sequences_8k[:,0:-(OVERLAP-FRAME_SIZE+1)]
prev_samples = prev_samples.reshape((1, batch_size, 1, -1))
prev_samples = T.nnet.neighbours.images2neibs(prev_samples, (1,  FRAME_SIZE), neib_step=(1, 1), mode='valid') #2-dim:([[x7,x8],[x8,x9],[x9,x10],...])
prev_samples = prev_samples.reshape((batch_size * SEQ_LEN,  FRAME_SIZE))

sample_level_outputs = sample_level_predictor(
    frame_level_outputs.reshape((batch_size * SEQ_LEN, DIM)),
    prev_samples
)          #sample_level_outputs dim:(BATCH_SIZE * SEQ_LEN, Q_LEVELS) -> [[x9pre],[x10pre],...]

accuracy=T.eq(lib.ops.softmax_and_no_sample(sample_level_outputs.reshape((batch_size,SEQ_LEN,Q_LEVELS))),target_sequences)
accuracy=accuracy*target_mask
accuracy=T.sum(accuracy,axis=1)
mask_sum=T.sum(target_mask,axis=1)

cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(sample_level_outputs),  #Every row represents a distribution(256 propability)
    target_sequences.flatten()    #A list, represent the groundtruth of every row
)
cost = cost.reshape(target_sequences.shape)
cost = cost * target_mask #dim: batch*num
# Don't use these lines; could end up with NaN
# Specially at the end of audio files where mask is
# all zero for some of the shorter files in mini-batch.
#cost = cost.sum(axis=1) / target_mask.sum(axis=1)
#cost = cost.mean(axis=0)
cost_sum=T.sum(cost,axis=1)
# Use this one instead.
cost = cost.sum()
cost = cost / target_mask.sum() #cost average by samples

# By default we report cross-entropy cost in bits.
# Switch to nats by commenting out this line:
# log_2(e) = 1.44269504089
#cost = cost * lib.floatX(numpy.log2(numpy.e))

###########
all_params = lib.get_params(cost, lambda x: hasattr(x, 'param') and x.param==True) #if LEARN_H0=True,then learn_h0 is included in parmeters to train

lib.print_params_info(all_params, path=FOLDER_PREFIX)

grads = T.grad(cost, wrt=all_params, disconnected_inputs='warn')
grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

updates = lasagne.updates.adam(grads, all_params,learning_rate=lr)

# Training function(s)
train_fn = theano.function(
    [sequences_8k,sequences_up, condition, con_h0,big_h0, h0, reset, mask,batch_size,lr],
    [cost, new_con_h0,new_big_h0, new_h0],
    updates=updates,
    on_unused_input='warn'
)

# Validation and Test function, hence no updates
valid_fn = theano.function(
    [sequences_8k,sequences_up, condition,con_h0,big_h0,h0, reset, mask,batch_size],
    [cost_sum, accuracy,mask_sum,new_con_h0,new_big_h0,new_h0],
    on_unused_input='warn'
)

test_fn=theano.function(
    [sequences_8k,sequences_up, condition,con_h0,big_h0,h0, reset, mask,batch_size],
    [cost_sum,accuracy,mask_sum,lib.ops.softmax_and_no_sample(sample_level_outputs.reshape((batch_size,SEQ_LEN,Q_LEVELS))),new_con_h0,new_big_h0,new_h0],
    on_unused_input='warn'
)

def generate_and_save_samples(tag):
    def write_audio_file(name, data):
        data = data.astype('float32')
        #data -= data.min()
        #data /= data.max()
        #data -= 0.5
        #data *= 0.95
        scipy.io.wavfile.write(
                    os.path.join(SAMPLES_PATH, name+'.wav'),
                    BITRATE,
                    data)

    total_time=time()
    costs_g = []
    accuracys_g=[]
    count=0
    data_feeder = load_data(test_feeder)
    for seqs_g_8k,seqs_g_up, reset_g, end_flag_g,mask_g,con_g,batch_g,seqs_g_8k_real in data_feeder:
        if reset_g==1:
            con_h0_g=numpy.zeros((batch_g, N_CON_RNN, H0_MULT*CON_TIER_DIM), dtype='float32')
            big_h0_g = numpy.zeros((batch_g, N_BIG_RNN, H0_MULT*DIM), dtype='float32')
            h0_g = numpy.zeros((batch_g, N_RNN, H0_MULT*DIM), dtype='float32')
            cost_batch=np.zeros((batch_g,),dtype='float32')
            accuracy_batch=np.zeros((batch_g,),dtype='float32')
            mask_batch=np.zeros((batch_g,),dtype='float32')
            count+=1
        cost_g, accuracy_g,mask_sum_g,sample, con_h0_g,big_h0_g,h0_g = test_fn(seqs_g_8k,seqs_g_up, con_g,con_h0_g,big_h0_g,h0_g, reset_g, mask_g,batch_g)
        cost_batch=cost_batch+cost_g
        accuracy_batch=accuracy_batch+accuracy_g
        mask_batch=mask_batch+mask_sum_g
        if end_flag_g==1:
            costs_g.extend(list(cost_batch/mask_batch))
            accuracys_g.extend(list(accuracy_batch/mask_batch))

        if count==1:
            if reset_g==1:
                samples_low=seqs_g_8k_real[:,0:-OVERLAP]
                samples=sample
                masks_g=mask_g[:,0:-OVERLAP]
            else:
                samples_low=np.concatenate([samples_low,seqs_g_8k_real[:,0:-OVERLAP]],axis=1)
                samples=np.concatenate([samples,sample],axis=1)
                masks_g=np.concatenate([masks_g,mask_g[:,0:-OVERLAP]],axis=1)


    for i in xrange(N_SEQS):
        samples_lowi=samples_low[i]
        samplei=samples[i]
        maski=masks_g[i]
        samples_lowi=samples_lowi[0:len(np.where(maski==1)[0])]
        samplei=samplei[0:len(np.where(maski==1)[0])]
        if Q_TYPE == 'mu-law':
            from datasets.dataset import mu2linear
            samplei = mu2linear(samplei)
        write_audio_file("sample_{}_{}".format(tag, i), samplei/3+samples_lowi)

    total_time = time() - total_time
    log = "{} samples generated in {} seconds."
    log = log.format(N_SEQS, total_time)
    print log,

    return numpy.mean(costs_g),numpy.mean(accuracys_g)*100,total_time


def monitor(data_feeder):
    """
    Cost and time of test_fn on a given dataset section.
    Pass only one of `valid_feeder` or `test_feeder`.
    Don't pass `train_feed`.

    :returns:
        Mean cost over the input dataset (data_feeder)
        Total time spent
    """
    _total_time = time()
    _costs = []
    _accuracys=[]
    _data_feeder = load_data(data_feeder)
    for _seqs_8k,_seqs_up, _reset, _end_flag,_mask,_con,_batch,_seqs_8k_real in _data_feeder:
        if _reset==1:
            _con_h0=numpy.zeros((_batch, N_CON_RNN, H0_MULT*CON_TIER_DIM), dtype='float32')
            _big_h0=numpy.zeros((_batch, N_BIG_RNN, H0_MULT*DIM), dtype='float32')
            _h0 = numpy.zeros((_batch, N_RNN, H0_MULT*DIM), dtype='float32')
            _cost_batch=np.zeros((_batch,),dtype='float32')
            _accuracy_batch=np.zeros((_batch,),dtype='float32')
            _mask_batch=np.zeros((_batch,),dtype='float32')
        _cost, _accuracy,_mask_sum,_con_h0,_big_h0,_h0 = valid_fn(_seqs_8k,_seqs_up, _con,_con_h0,_big_h0,_h0, _reset, _mask,_batch)
        _cost_batch=_cost_batch+_cost
        _accuracy_batch=_accuracy_batch+_accuracy
        _mask_batch=_mask_batch+_mask_sum
        if _end_flag==1:
            _costs.extend(list(_cost_batch/_mask_batch))
            _accuracys.extend(list(_accuracy_batch/_mask_batch))


    return numpy.mean(_costs), numpy.mean(_accuracys)*100,time() - _total_time

print "Wall clock time spent before training started: {:.2f}h"\
        .format((time()-exp_start)/3600.)
print "Training!"
total_iters = 0
total_time = 0.
last_print_time = 0.
last_print_iters = 0
costs = []
lowest_valid_cost = numpy.finfo(numpy.float32).max
corresponding_test_cost = numpy.finfo(numpy.float32).max
new_lowest_cost = False
end_of_batch = False
epoch = 0
learning_rate=LEARNING_RATE

# Initial load train dataset
tr_feeder = load_data(train_feeder)

### Handling the resume option:
if RESUME:
    # Check if checkpoint from previous run is not corrupted.
    # Then overwrite some of the variables above.
    iters_to_consume, res_path, epoch, total_iters,\
        [lowest_valid_cost, corresponding_test_cost, test_cost] = \
        lib.resumable(path=FOLDER_PREFIX,
                      iter_key=iter_str,
                      epoch_key=epoch_str,
                      add_resume_counter=True,
                      other_keys=[lowest_valid_str,
                                  corresp_test_str,
                                  test_nll_str])
    # At this point we saved the pkl file.
    last_print_iters = total_iters
    print "### RESUMING JOB FROM EPOCH {}, ITER {}".format(epoch, total_iters)
    # Consumes this much iters to get to the last point in training data.
    consume_time = time()
    for i in xrange(iters_to_consume):
        tr_feeder.next()
    consume_time = time() - consume_time
    print "Train data ready in {:.2f}secs after consuming {} minibatches.".\
            format(consume_time, iters_to_consume)

    lib.load_params(res_path)
    print "Parameters from last available checkpoint loaded."

while True:
    # THIS IS ONE ITERATION
    if total_iters % 500 == 0:
        print total_iters,

    total_iters += 1

    try:
        # Take as many mini-batches as possible from train set
        mini_batch = tr_feeder.next()
    except StopIteration:
        # Mini-batches are finished. Load it again.
        # Basically, one epoch.
        tr_feeder = load_data(train_feeder)

        # and start taking new mini-batches again.
        mini_batch = tr_feeder.next()
        epoch += 1
        end_of_batch = True
        print "[Another epoch]",

    seqs_8k, seqs_up,reset, end_flag,mask,con,batch_num,seqs_8k_real = mini_batch
    if reset==1:
        con_h0=numpy.zeros((batch_num, N_CON_RNN, H0_MULT*CON_TIER_DIM), dtype='float32')
        big_h0=numpy.zeros((batch_num, N_BIG_RNN, H0_MULT*DIM), dtype='float32')
        h0 = numpy.zeros((batch_num, N_RNN, H0_MULT*DIM), dtype='float32')

    start_time = time()
    cost,con_h0,big_h0,h0 = train_fn(seqs_8k, seqs_up, con,con_h0, big_h0, h0, reset, mask,batch_num,learning_rate)
    total_time += time() - start_time
    #print "This cost:", cost, "This h0.mean()", h0.mean()

    costs.append(cost)

    # Monitoring step
    if (TRAIN_MODE=='iters' and total_iters-last_print_iters == PRINT_ITERS) or \
        (TRAIN_MODE=='time' and total_time-last_print_time >= PRINT_TIME) or \
        (TRAIN_MODE=='time-iters' and total_time-last_print_time >= PRINT_TIME) or \
        (TRAIN_MODE=='iters-time' and total_iters-last_print_iters >= PRINT_ITERS) or \
        end_of_batch:
        # 0. Validation
        print "\nValidation!",
        valid_cost, valid_accuracy,valid_time = monitor(valid_feeder)
        print "Done!"

        # 1. Test
        test_time = 0.
        # Only when the validation cost is improved get the cost for test set.
        if valid_cost < lowest_valid_cost:
            lowest_valid_cost = valid_cost
            print "\n>>> Best validation cost of {} reached."\
                    .format(valid_cost),
            #test_cost, test_time = monitor(test_feeder)
            #print "Done!"
            # Report last one which is the lowest on validation set:
            #print ">>> test cost:{}\ttotal time:{}".format(test_cost, test_time)
            #corresponding_test_cost = test_cost
            new_lowest_cost = True

        tag = "e{}_i{}_t{:.2f}_tr{:.4f}_v{:.4f}"
        tag = tag.format(epoch,
                         total_iters,
                         total_time/3600,
                         numpy.mean(cost),
                         valid_cost)
        tag += ("_best" if new_lowest_cost else "")

        print "Sampling!",
        # Generate samples
        test_cost, test_accuracy,test_time=generate_and_save_samples(tag)
        print "\n>>> test cost:{}\ttest accuracy:{}%\ttotal time:{}".format(test_cost, test_accuracy,test_time)
        if new_lowest_cost:
            corresponding_test_cost = test_cost
        print "Done!"

        # 2. Stdout the training progress
        print_info = "epoch:{}\ttotal iters:{}\twall clock time:{:.2f}h\n"
        print_info += ">>> Lowest valid cost:{}\t Corresponding test cost:{}\n"
        print_info += "\ttrain cost:{:.4f}\ttotal time:{:.2f}h\tper iter:{:.3f}s\n"
        print_info += "\tvalid cost:{:.4f}\tvalid accuracy:{:.4f}%\ttotal time:{:.2f}h\n"
        print_info += "\ttest  cost:{:.4f}\ttest accuracy:{:.4f}%\ttotal time:{:.2f}h"
        print_info = print_info.format(epoch,
                                       total_iters,
                                       (time()-exp_start)/3600,
                                       lowest_valid_cost,
                                       corresponding_test_cost,
                                       numpy.mean(costs),
                                       total_time/3600,
                                       total_time/total_iters,
                                       valid_cost,
                                       valid_accuracy,
                                       valid_time/3600,
                                       test_cost,
                                       test_accuracy,
                                       test_time/3600)
        print print_info


        # 3. Save params of model (IO bound, time consuming)
        # If saving params is not successful, there shouldn't be any trace of
        # successful monitoring step in train_log as well.
        print "Saving params!",
        lib.save_params(
                os.path.join(PARAMS_PATH, 'params_{}.pkl'.format(tag))
        )
        print "Done!"

        # 4. Save and graph training progress (fast)
        training_info = {epoch_str : epoch,
                         iter_str : total_iters,
                         train_nll_str : numpy.mean(costs),
                         valid_nll_str : valid_cost,
                         test_nll_str : test_cost,
                         lowest_valid_str : lowest_valid_cost,
                         corresp_test_str : corresponding_test_cost,
                         'train time' : total_time,
                         'valid time' : valid_time,
                         'test time' : test_time,
                         'wall clock time' : time()-exp_start}
        lib.save_training_info(training_info, FOLDER_PREFIX)
        print "Train info saved!",

        # y_axis_strs = [train_nll_str, valid_nll_str, test_nll_str]
        # lib.plot_traing_info(iter_str, y_axis_strs, FOLDER_PREFIX)
        print "And plotted!"

        if total_iters-last_print_iters == PRINT_ITERS:
                # If we are here b/c of onom_end_of_batch, we shouldn't mess
                # with costs and last_print_iters
            costs = []
            last_print_time += PRINT_TIME
            last_print_iters += PRINT_ITERS

        if epoch==6 and end_of_batch==True:
            learning_rate=0.0001
            print "\n Now learning rate is 0.0001."

        end_of_batch = False
        new_lowest_cost = False

        print "Validation Done!\nBack to Training..."

    if (TRAIN_MODE=='iters' and total_iters == STOP_ITERS) or \
       (TRAIN_MODE=='time' and total_time >= STOP_TIME) or \
       ((TRAIN_MODE=='time-iters' or TRAIN_MODE=='iters-time') and \
            (total_iters == STOP_ITERS or total_time >= STOP_TIME)):

        print "Done! Total iters:", total_iters, "Total time: ", total_time
        print "Experiment ended at:", datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M')
        print "Wall clock time spent: {:.2f}h"\
                    .format((time()-exp_start)/3600)

        sys.exit()