import tensorflow as tf
import numpy as np
import math
import argparse
import logging


def set_parser():
    '''
    コマンドライン引数をセット
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--minibatch_size", default=32)
    parser.add_argument("--iteration_num", default=100)
    parser.add_argument("--data_num", default=10000)
    parser.add_argument("--learning_rate", default=0.001)
    args = parser.parse_args()
    return args

def read_data(picks_nums, input_shape):
    '''
    データを読み込み，学習用のミニバッジを返す
    '''
    # tmp
    X = []
    Y = []
    for i in picks_nums:
        file_name = "narrays/array{:08d}.csv".format(i)
        with open(file_name, "r", encoding="utf-8") as f:
            f.readline()
            input_shape = tuple(map(int, f.readline().strip().split(",")))
            y = np.array(list(map(int, f.readline().strip().split(","))))
            x = np.array(list(f.readline().strip()))
        x = x.reshape(input_shape)
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def generate_mini_batch(N: int, mini_size: int, input_dims: tuple):
    '''
    N: データ数
    mini_size: ミニバッジのサイズ
    '''
    random_lists = [i for i in range(N)]
    np.random.shuffle(random_lists)
    iter_num = math.ceil(N/mini_size)
    for i in range(iter_num):
        picks_nums = random_lists[i * mini_size: min((i + 1) * mini_size, N)]
        X, Y = read_data(picks_nums, input_dims)
        yield X, Y


def create_placeholder(n0_h, n0_w, n0_c, output_dim):
    '''
    for tensorflow
    '''
    # create placeholder of input Matrix
    X = tf.placeholder(dtype=tf.float32, name="X", shape=(None, n0_h, n0_w, n0_c))
    Y = tf.placeholder(dtype=tf.float32, name="Y", shape=(None, output_dim))
    return X, Y


def initialize_parameter(hparameters, n0_c):
    '''
    for tensorflow
    '''
    conv_f1 = hparameters["conv_f1"]
    conv_f2 = hparameters["conv_f2"]
    n1_c = hparameters["n1_c"]
    n2_c = hparameters["n2_c"]
    W1 = tf.get_variable(dtype=tf.float32, name="W1", initializer=tf.contrib.layers.xavier_initializer(), shape=(conv_f1, conv_f1, n0_c, n1_c))
    W2 = tf.get_variable(dtype=tf.float32, name="W2", initializer=tf.contrib.layers.xavier_initializer(), shape=(conv_f2, conv_f2, n1_c, n2_c))
    b1 = tf.get_variable(dtype=tf.float32, name="b1", initializer=tf.zeros_initializer(), shape=(1, 1 ,1, n1_c))
    b2 = tf.get_variable(dtype=tf.float32, name="b2", initializer=tf.zeros_initializer(), shape=(1, 1, 1, n2_c))
    return W1, W2, b1, b2


def forward_propagation(hparameters, X, W1, W2, b1, b2, output_dim):
    '''
    for tensorflow
    '''
    conv_s1 = hparameters["conv_s1"]
    pool_s1 = hparameters["pool_s1"]
    pool_f1 =  hparameters["pool_f1"]

    conv_s2 = hparameters["conv_s2"]
    pool_s2 = hparameters["pool_s2"]
    pool_f2 =  hparameters["pool_f2"]

    Z1 = tf.add(tf.nn.conv2d(filter=W1, input=X, name="Z1", strides=[1, conv_s1, conv_s1, 1], padding="SAME"), b1)
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=(1, pool_f1, pool_f1, 1), strides=[1, pool_s1, pool_s1, 1], padding="SAME")

    Z2 = tf.add(tf.nn.conv2d(filter=W2, input=P1, name="Z2", strides=[1, conv_s2, conv_s2, 1], padding="SAME"), b2)
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=(1, pool_f2, pool_f2, 1), strides=[1, pool_s2, pool_s2, 1], padding="SAME")

    P2 = tf.contrib.layers.flatten(inputs=P2)
    S = tf.contrib.layers.fully_connected(inputs=P2, num_outputs=output_dim)

    return S

def define_cost_tensor(mini_batch_size, output_dim, S, Y):
    '''
    for tensorflow
    '''
    cost = 1 / (mini_batch_size * output_dim) * tf.reduce_mean(tf.abs(tf.square(S - Y)))
    return cost

def main(args, _logger, print_cost=True):
    N = args.data_num
    mini_batch_size = args.minibatch_size
    iteration_num = args.iteration_num
    learning_rate = args.learning_rate
    input_dims, output_dim = read_dimensions()
    hparameters = {"conv_f1": 2, "conv_s1":1,"pool_f1":2, "pool_s1":1, "n1_c":8,
                   "conv_f2": 4,"conv_s2":2,"pool_f2":4, "pool_s2":2, "n2_c":16}
    
    n0_h, n0_w, n0_c = input_dims
    tf.reset_default_graph()
    X, Y = create_placeholder(n0_h, n0_w, n0_c, output_dim)
    W1, W2, b1, b2 = initialize_parameter(hparameters, n0_c)
    S = forward_propagation(hparameters, X, W1, W2, b1, b2, output_dim)
    cost = define_cost_tensor(mini_batch_size, output_dim, S, Y)
    optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1, iteration_num + 1):
            costs = []
            for train_X, train_Y in generate_mini_batch(N, mini_batch_size, input_dims):
                _, temp_cost = sess.run([optimize, cost], feed_dict={X:train_X, Y:train_Y})
                costs.append(temp_cost)
            if epoch % 10 == 0 and print_cost:
                _logger.info("EPOCH {0} COST: {1}".format(epoch,len(costs)))
        write_predict(sess, S)

def read_dimensions():
    file_name = "narrays/array00000000.csv"
    with open(file_name, "r", encoding='utf-8') as f:
        input_dims = tuple(map(int, f.readline().strip().split(",")))
        output_dim = len(list(f.readline().strip().split(",")))
    return input_dims, output_dim

def set_logging():
    _logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s"
    )
    return _logger

def generate_test_data():
    test_data_size = 1
    for i in range(test_data_size):
        file_name = "narrays/t_array{:08d}.csv".format(i)
        with open(file_name, "r", encoding="utf-8") as f:
            input_shape = tuple(map(int, f.readline().strip().split(",")))
            x = np.array(list(f.readline().strip()))
            x = x.reshape(input_shape)
        yield x
    
def write_predict(sess, S):
    for x in generate_test_data():
        Y_hat = sess.run(S, feed_dict={X: x})
        # tmp
        file_name = "predict_y.csv"
        with open(file_name, "a", encoding="utf-8") as f:
            f.write(",".join([str(i) for i in Y_hat.tolist()]))
            f.write("\n")


if __name__ == "__main__":
    args=set_parser()
    _logger = set_logging()
    main(args, _logger)
