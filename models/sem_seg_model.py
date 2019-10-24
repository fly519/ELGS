import os
import sys
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_gab_util import pointnet_sa_module_withgab, pointnet_fp_module


def gating_process(inputs,num_output_channels,scope,stride=1,padding='VALID',bn_decay=None,is_training=None):

  with tf.variable_scope(scope) as sc:    
    num_in_channels = inputs.get_shape()[-1].value  
    kernel_shape = [1, num_in_channels, num_output_channels]

    with tf.device("/cpu:0"):       
        kernel = tf.get_variable('weights', kernel_shape, initializer= tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        biases = tf.get_variable('biases', [num_output_channels], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
  
    df='NHWC'
    outputs = tf.nn.conv1d(inputs, kernel, stride=stride, padding=padding, data_format=df)
    outputs = tf.nn.bias_add(outputs, biases, data_format=df)
    

    outputs =tf.contrib.layers.batch_norm(outputs, 
                                      center=True, scale=True,
                                      is_training=is_training, decay=bn_decay,updates_collections=None,
                                      scope='bn',
                                      data_format=df)
    outputs = tf.nn.relu(outputs)

    return outputs


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, num_neighbors, farthest_distance, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx9, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
   
    l0_xyz = point_cloud[:, :, 0:3]
    l0_points = point_cloud[:, :, 3:9]

    """Point Enrichment  """

    new_xyz = l0_xyz # (batch_size, npoint, 3)
    idx, pts_cnt = query_ball_point(farthest_distance, num_neighbors, l0_xyz, new_xyz)

    neighbor_xyz = group_point(l0_xyz, idx) 
    neighbor_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,num_neighbors,1]) 
    
    neighbor_points = group_point(l0_points, idx) 
    neighbor_representation = tf.concat([neighbor_xyz, neighbor_points], axis=-1) 
    neighbor_representation=tf.reshape(neighbor_representation, (batch_size, num_point, -1)) 

    num_channel=neighbor_representation.get_shape()[2].value
    points= tf_util.conv1d(point_cloud, num_channel, 1, padding='VALID', bn=True, is_training=is_training, scope='points_fc', bn_decay=bn_decay)
    
    neighbor_representation_gp= gating_process(neighbor_representation, num_channel, padding='VALID',  is_training=is_training, scope='neighbor_representation_gp', bn_decay=bn_decay)
    points_gp= gating_process(points, num_channel,  padding='VALID',  is_training=is_training, scope='points_gp', bn_decay=bn_decay)

    l0_points=tf.concat([neighbor_representation_gp*points, points_gp*neighbor_representation], axis=-1) 

    # Layer 1
    l1_xyz, l1_points, l1_indices = pointnet_sa_module_withgab( l0_xyz, l0_points, npoint=1024, radius=0.1, nsample=32, mlp=[32,32,64], mlp2=[64,64], group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1',gab=True)
    l2_xyz, l2_points, l2_indices = pointnet_sa_module_withgab( l1_xyz, l1_points, npoint=256, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module_withgab( l2_xyz, l2_points, npoint=64, radius=0.4, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module_withgab(l3_xyz, l3_points, npoint=16, radius=0.8, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')
     
    # Feature Propagation layers
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer2')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer3')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)


    """Spatial-wise Attention"""

    input = net     
    output_a = tf_util.conv2d(tf.expand_dims(input, 1),64, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv_output_a', bn_decay=bn_decay) 

    output_b= tf_util.conv2d(tf.expand_dims(input, 1),64, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv_output_b', bn_decay=bn_decay)

    output_b = tf.transpose(output_b, [0,1,3,2])
    output_a = tf.squeeze(output_a)
    output_b = tf.squeeze(output_b)

    
    energy=tf.matmul(output_a,output_b)
    attention=tf.nn.softmax(energy,axis=-1)

    output_d= tf_util.conv2d(tf.expand_dims(input, 1),128, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv_output_d', bn_decay=bn_decay)

    output_d = tf.squeeze(output_d)
        
    gamma = tf_util._variable_with_weight_decay('weight_patial',
                                shape=[1],
                                use_xavier=True,
                                stddev=1e-3,
                                wd=0.0)

    output_SA=tf.matmul(attention,output_d)
    output_SA=output_SA*gamma+tf.squeeze(input)
  



    """Channel-wise Attention"""
  
    output_f = tf.transpose(input, [0,2,1])
    energy=tf.matmul(output_f,input)

    D=tf.reduce_max(energy, -1)
    D=tf.expand_dims(D, -1)   

    energy_new=tf.tile(D, multiples=[1, 1,energy.shape[2]])-energy
    attention=tf.nn.softmax(energy_new,axis=-1)

    output_CA=tf.matmul(input,attention)

    gamma2 = tf_util._variable_with_weight_decay('weightsgamma2m',
                                shape=[1],
                                use_xavier=True,
                                stddev=1e-3,
                                wd=0.0)
    output_CA=output_CA*gamma2+input

    output=output_SA+output_CA
    end_points['feats'] = output 


    net = tf_util.dropout(output, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net,13, 1, padding='VALID', activation_fn=None, scope='fc2')


    return net, end_points


def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, 
	smpw: BxN """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)
