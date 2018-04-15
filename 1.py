from __future__ import division, print_function, absolute_import
import tensorflow as tf

from config import cfg
from utils import get_batch_data
from utils import softmax
from utils import reduce_sum

import pdb

import numpy as np
num_input=31*31
num_classes=103
num_steps=10000
display_step=1000
learning_rate=0.001

X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])


epsilon = 1e-9

class CapsLayer(object):
    ''' Capsule layer.
    Args:
        input: A 4-D tensor.
        num_outputs: the number of capsule in this layer.
        vec_len: integer, the length of the output vector of a capsule.
        layer_type: string, one of 'FC' or "CONV", the type of this layer,
            fully connected or convolution, for the future expansion capability
        with_routing: boolean, this capsule is routing with the
                      lower-level layer capsule.

    Returns:
        A 4-D tensor.
    '''
    def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type

    def __call__(self, input, kernel_size=None, stride=None):
        '''
        The parameters 'kernel_size' and 'stride' will be used while 'layer_type' equal 'CONV'
        '''
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            if not self.with_routing:
                # the PrimaryCaps layer, a convolutional layer
                # input: [batch_size, 20, 20, 256]
                #assert input.get_shape() == [cfg.batch_size, 20, 20, 256]

                '''
                # version 1, computational expensive
                capsules = []
                for i in range(self.vec_len):
                    # each capsule i: [batch_size, 6, 6, 32]
                    with tf.variable_scope('ConvUnit_' + str(i)):
                        caps_i = tf.contrib.layers.conv2d(input, self.num_outputs,
                                                          self.kernel_size, self.stride,
                                                          padding="VALID", activation_fn=None)
                        caps_i = tf.reshape(caps_i, shape=(cfg.batch_size, -1, 1, 1))
                        capsules.append(caps_i)
                #assert capsules[0].get_shape() == [cfg.batch_size, 1152, 1, 1]
                capsules = tf.concat(capsules, axis=2)
                '''

                # version 2, equivalent to version 1 but higher computational
                # efficiency.
                # NOTE: I can't find out any words from the paper whether the
                # PrimaryCap convolution does a ReLU activation or not before
                # squashing function, but experiment show that using ReLU get a
                # higher test accuracy. So, which one to use will be your choice
                capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len,
                                                    self.kernel_size, self.stride, padding="VALID",
                                                    activation_fn=tf.nn.relu)
                # capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len,
                #                                    self.kernel_size, self.stride,padding="VALID",
                #                                    activation_fn=None)
                capsules = tf.reshape(capsules, (cfg.batch_size, -1, self.vec_len, 1))

                print(capsules.get_shape())
                # [batch_size, 1152, 8, 1]
                capsules = squash(capsules)
                #assert capsules.get_shape() == [cfg.batch_size, 1152, 8, 1]
                return(capsules)

        if self.layer_type == 'FC':
            if self.with_routing:
                # the DigitCaps layer, a fully connected layer
                # Reshape the input into [batch_size, 1152, 1, 8, 1]
                self.input = tf.reshape(input, shape=(cfg.batch_size, -1, 1, input.shape[-2].value, 1))

                #pdb.set_trace()
                with tf.variable_scope('routing'):
                    # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                    # about the reason of using 'batch_size', see issue #21
                    b_IJ = tf.constant(np.zeros([cfg.batch_size, input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
                    capsules = routing(self.input, b_IJ)
                    capsules = tf.squeeze(capsules, axis=1)

            return(capsules)


def routing(input, b_IJ):
    ''' The routing algorithm.

    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
    Returns:
        A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''

    # W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1]
    W = tf.get_variable('Weight', shape=(1, 1568, 16*num_classes, 8, 1), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=cfg.stddev))
    biases = tf.get_variable('bias', shape=(1, 1, num_classes, 16, 1))

    # Eq.2, calc u_hat
    # Since tf.matmul is a time-consuming op,
    # A better solution is using element-wise multiply, reduce_sum and reshape
    # ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
    # element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
    # reshape to [a, c]
    input = tf.tile(input, [1, 1, 16*num_classes, 1, 1])
    #assert input.get_shape() == [cfg.batch_size, 1152, 160, 8, 1]

    u_hat = reduce_sum(W * input, axis=3, keepdims=True)
    u_hat = tf.reshape(u_hat, shape=[-1, 1568, num_classes, 16, 1])
    #assert u_hat.get_shape() == [cfg.batch_size, 1152, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(cfg.iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            c_IJ = softmax(b_IJ, axis=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == cfg.iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                #assert s_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]

                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                #assert v_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
            elif r_iter < cfg.iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)

                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, 1568, 1, 1, 1])
                u_produce_v = reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                #assert u_produce_v.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v

    return(v_J)


def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)

class CapsNet(object):
    def __init__(self, is_training=True):
        self.X = tf.reshape(X,(cfg.batch_size,31,31,1))
        self.labels = Y
        self.Y = Y
        self.build_arch()
        self.loss()
        #self._summary()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step) 


    def build_arch(self):
        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256,
                                             kernel_size=9, stride=1,
                                             padding='VALID')
            assert conv1.get_shape() == [cfg.batch_size, 23, 23, 256]

        # Primary Capsules layer, return [batch_size, 1152, 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(conv1, kernel_size=10, stride=2)
            ##assert caps1.get_shape() == [cfg.batch_size, 1152, 8, 1]

        # DigitCaps layer, return [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs=num_classes, vec_len=16, with_routing=True, layer_type='FC')
            self.caps2 = digitCaps(caps1)

        # Decoder structure in Fig. 2
        # 1. Do masking, how:
        with tf.variable_scope('Masking'):
            # a). calc ||v_c||, then do softmax(||v_c||)
            # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
            self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps2),
                                               axis=2, keepdims=True) + epsilon)
            # self.softmax_v = softmax(self.v_length, axis=1)
            # #assert self.softmax_v.get_shape() == [cfg.batch_size, 10, 1, 1]

            # # b). pick out the index of max softmax val of the 10 caps
            # # [batch_size, 10, 1, 1] => [batch_size] (index)
            # self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
            # #assert self.argmax_idx.get_shape() == [cfg.batch_size, 1, 1]
            # self.argmax_idx = tf.reshape(self.argmax_idx, shape=(cfg.batch_size, ))

            # # Method 1.
            # if not cfg.mask_with_y:
            #     # c). indexing
            #     # It's not easy to understand the indexing process with argmax_idx
            #     # as we are 3-dim animal
            #     masked_v = []
            #     for batch_size in range(cfg.batch_size):
            #         v = self.caps2[batch_size][self.argmax_idx[batch_size], :]
            #         masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

            #     self.masked_v = tf.concat(masked_v, axis=0)
            #     #assert self.masked_v.get_shape() == [cfg.batch_size, 1, 16, 1]
            # # Method 2. masking with true label, default mode
            # else:
            #     # self.masked_v = tf.matmul(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)), transpose_a=True)
            #     self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)))
            #     self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps2), axis=2, keepdims=True) + epsilon)

        # 2. Reconstructe the MNIST images with 3 FC layers
        # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
        # self.decode=np.zero(cfg.batch_size,961)
        def deconding_network(input_v):
            fc1 = tf.contrib.layers.fully_connected(input_v, num_outputs=512)
            #assert fc1.get_shape() == [cfg.batch_size, 512]
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            #assert fc2.get_shape() == [cfg.batch_size, 1024]
            return tf.contrib.layers.fully_connected(fc2, num_outputs=961, activation_fn=tf.sigmoid)
        
        self.caps2=tf.squeeze(self.caps2)
        self.Y=tf.reshape(self.Y,(-1,num_classes))
        with tf.variable_scope('Decoder',reuse=tf.AUTO_REUSE):
            self.decoded=deconding_network(tf.reshape(tf.multiply(self.caps2[:,0,:],tf.expand_dims(self.Y[:,0],-1)), shape=(cfg.batch_size, 16)))
            for i in range(1,9):
                self.decoded += deconding_network(tf.reshape(tf.multiply(self.caps2[:,i,:],tf.expand_dims(self.Y[:,i],-1)), shape=(cfg.batch_size, 16)))


    def loss(self):
        # 1. The margin loss

        # [batch_size, 10, 1, 1]
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))
        #assert max_l.get_shape() == [cfg.batch_size, 10, 1, 1]

        # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

        # calc T_c: [batch_size, 10]
        # T_c = Y, is my understanding correct? Try it.
        T_c = self.Y
        # [batch_size, 10], element-wise multiply
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 2. The reconstruction loss
        orgin = tf.reshape(self.X, shape=(cfg.batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        # The paper uses sum of squared error as reconstruction error, but we
        # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # mean squared error. In order to keep in line with the paper,the
        # regularization scale should be 0.0005*784=0.392
        self.total_loss = self.margin_loss + cfg.regularization_scale * self.reconstruction_err

    # Summary
    #def _summary(self):
        # train_summary = []
        # train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        # train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_err))
        # train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        # recon_img = tf.reshape(self.decoded, shape=(cfg.batch_size, 28, 28, 1))
        # train_summary.append(tf.summary.image('reconstruction_img', recon_img))
        # self.train_summary = tf.summary.merge(train_summary)

        #correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        #self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))






# Start training
p=np.load('./V1_data/region1/training_inputs.npy')
p2=np.load('./V1_data/region1/training_set.npy')
data=tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(p*1000000/255, dtype=tf.float32),
                                        tf.convert_to_tensor(p2/(1+p2), dtype=tf.float32)))
data=data.batch(2)
iterator = data.make_one_shot_iterator()

model=CapsNet()

init = tf.global_variables_initializer()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    #tf.stop_gradient(pretrained_weights)

    for step in range(1, num_steps+1):
        # print(step)
        batch_x, batch_y =  sess.run(iterator.get_next())
        #pdb.set_trace()
        # Run optimization op (backprop)
        _, loss= sess.run([model.train_op, model.total_loss], feed_dict={X: batch_x, Y: batch_y})
        sess.run([model.train_op], feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss= sess.run([model.total_loss], feed_dict={X: batch_x, Y: batch_y})
            print(loss)
            

    print("Optimization Finished!")

    # x_test=mnist.test.images
    # y_test=mnist.test.labels
    # print(x_test.shape,y_test.shape)

    # p=0
    # for i in range(9):
    #     p=p+sess.run(model.accuracy, feed_dict={X: x_test[i*batch_size:(i+1)*batch_size,:],
    #                                     Y: y_test[i*batch_size:(i+1)*batch_size,:]})
    # print(p/9)