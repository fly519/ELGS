import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider_p2 as  provider
import tf_util
import part_dataset_all_normal


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='sem_seg_model', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log_test', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=121, help='Epoch to run [default: 121]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.8, help='Initial learning rate [default: 0.8]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.6, help='Decay rate for lr decay [default: 0.6]')


parser.add_argument('--K', type=int, default=3, help='The number of neighbors k in contextual representation [default: 3]')
parser.add_argument('--R', type=float, default=0.06, help='The farthest distance r in contextual representation [default: 0.06]')
parser.add_argument('--ckpt', default='log/best_seg_model.ckpt', help='Log dir [default: log]')
parser.add_argument('--ckpt_meta', default='log/best_seg_model.ckpt.meta', help='ckpt file [default: best_seg_model.ckpt.meta]')

FLAGS = parser.parse_args()

EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate


NUM_NEIGHBORS = FLAGS.K
FARTHEST_DISTANCE=FLAGS.R
CKPT_META = FLAGS.ckpt_meta
CKPT = FLAGS.ckpt

MODEL = importlib.import_module(FLAGS.model) # import network module

MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 13

## Load ALL data
ALL_FILES = provider.getDataFiles('../data/indoor3d_sem_seg_hdf5_data/all_files.txt')
room_filelist = [line.rstrip() for line in open('../data/indoor3d_sem_seg_hdf5_data/room_filelist.txt')]
data_batch_list = []
label_batch_list = []
for h5_filename in ALL_FILES:
    data_batch, label_batch = provider.loadDataFile(h5_filename)
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)

data_batches = np.concatenate(data_batch_list, 0)
label_batches = np.concatenate(label_batch_list, 0)
print(data_batches.shape)
print(label_batches.shape)
test_area = 'Area_5'

test_idxs = []
room_name_list=[]
for i,room_name in enumerate(room_filelist):
    if test_area in room_name:
        test_idxs.append(i)
        room_name_list.append(room_name)

lis = [] 
for i in room_name_list:                    
    if  i  not in lis:            
        lis.append(i)

test_data = data_batches[test_idxs,...]
test_label = label_batches[test_idxs]


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print "--- Get model and loss"
            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_NEIGHBORS, FARTHEST_DISTANCE, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            print "--- Get training operator"
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=0)
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        new_saver = tf.train.import_meta_graph(CKPT_META)
        new_saver.restore(sess, CKPT)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_acc = -1
        sys.stdout.flush()
        eval(sess, ops, test_writer)
            

    
def eval(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
   
    current_data=test_data[:,0:NUM_POINT,:]
    current_label=test_label
    num_batches = current_data.shape[0]/BATCH_SIZE
    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    ious = np.zeros(13)
    totalnums = np.zeros(13)

    ious = np.zeros(NUM_CLASSES)
    totalnums = np.zeros(NUM_CLASSES)

    ##
    room_name_list.append('end_mark')
    lis_i=0
    seg_gt=[]
    seg_pred=[]
    for room_i,room in enumerate(room_name_list):
        if room=='end_mark':
            break

        start_idx = room_i
        end_idx = room_i+1
        
        batch_data=current_data[start_idx:end_idx, :, :]
        batch_label=current_label[start_idx:end_idx]
        
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training,}
        
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
             ops['loss'], ops['pred']], feed_dict=feed_dict)

        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)

        intersections = np.zeros(NUM_CLASSES)
        unions = np.zeros(NUM_CLASSES)
     
        seg_gt_block=batch_label.reshape(-1)
        seg_pred_block=pred_val.reshape(-1)       
        seg_gt.extend(seg_gt_block)
        seg_pred.extend(seg_pred_block)
        
        room_i1=room_i+1
        if lis[lis_i]!=room_name_list[room_i1]:
            lis_i=lis_i+1
            un, indices = np.unique(seg_gt, return_index=True)
            
            for segid in un:
                intersect = np.sum((seg_pred == segid) & (seg_gt == segid))
                union = np.sum((seg_pred == segid) | (seg_gt == segid))
                intersections[segid] += intersect
                unions[segid] += union
                
            iou = intersections / unions
            seg_gt=[]
            seg_pred=[]

            for i_iou, iou_ in enumerate(iou):
               if not np.isnan(iou_):
                    ious[i_iou] += iou_
                    totalnums[i_iou] += 1

    IOU=ious / totalnums
   
    print 'Semantic Segmentation IoU:', IOU
    nsum = 0.0
    for i in range(len(IOU)):
        nsum += IOU[i]
    miou= nsum / (len(IOU))
    print 'Semantic Segmentation Mean IoU:', miou

    #log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('Semantic Segmentation Overall Accuracy: %f'% (total_correct / float(total_seen)))   
    

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
