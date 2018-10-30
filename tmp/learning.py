import tensorflow as tf
import numpy as np
import math


def read_data(picks_nums: np.ndarray):
    '''
    データを読み込み，学習用のミニバッジを返す
    '''
    # tmp
    X = []
    Y = []
    for i in picks_nums:
        file_name = "array{:08d}.csv".format(i)
        with open(file_name, "r", encoding="utf-8") as f:
            input_shape = tuple(map(int, f.readline().strip().split(",")))
            y = np.array(list(map(int, f.readline().strip().split(","))))
            x = np.array(list(f.readline().strip()))
        x = x.reshape(input_shape)
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def generate_mini_batch(N: int, mini_size: int):
    '''
    N: データ数
    mini_size: ミニバッジのサイズ
    '''
    random_lists = [i for i in range(N)]
    np.random.shuffle(random_lists)
    iter_num = math.ceil(N/mini_size)
    for i in range(iter_num):
        picks_nums = random_lists[i * mini_size: min((i + 1) * mini_size, N)]
        X, Y = read_data(picks_nums)
        yield X, Y


def create_placeholder(n0_h, n0_w, n0_c, output_dim):
    # create placeholder of input Matrix
    X = tf.placeholder(dtype=tf.float32, name="X", shape=(None, n0_h, n0_w, n0_c))
    Y = tf.placeholder(dtype=tf.float32, name="Y", shape=(None, output_dim))
    return X, Y


# initialize parameter
def initialize_parameter(hparameters):
    conv_f1 = hparameters["conv_f1"]
    conv_f2 = hparameters["conv_f2"]
    n1_c = hparameters["n1_c"]
    n2_c = hparameters["n2_c"]
    W1 = tf.get_variable(dtype=tf.float32, name="W1", initializer=tf.contrib.layers.xavier_initializer(), shape=(conv_f1, conv_f1, n0_c, n1_c))
    W2 = tf.get_variable(dtype=tf.float32, name="W2", initializer=tf.contrib.layers.xavier_initializer(), shape=(conv_f2, conv_f2, n1_c, n2_c))
    b1 = tf.get_variable(dtype=tf.float32, name="b1", initializer=tf.zeros_initializer(), shape=(1, 1 ,1, n1_c))
    b2 = tf.get_variable(dtype=tf.float32, name="b2", initializer=tf.zeros_initializer(), shape=(1, 1, 1, n2_c))
    return W1, W2, b1, b2


def forward_propagation(hparameters, X, W1, W2, output_dim):
    # forward propagation

    # convolution
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

    # flatten
    P2 = tf.contrib.layers.flatten(inputs=P2)

    S = tf.contrib.layers.fully_connected(inputs=P2, num_outputs=output_dim)

    return S


def main():
    # paramter setting
    input_dims = (114, 92, 1)
    output_dim = 4
    # データ数
    N = 10
    mini_batch_size = 3

    n0_h, n0_w, n0_c = input_dims
    hparameters = {"conv_f1": 2, "conv_s1":1,"pool_f1":2, "pool_s1":1, "n1_c":8,
                   "conv_f2": 4,"conv_s2":2,"pool_f2":4, "pool_s2":2, "n2_c":16}
    iteration_num = 1000

    # reset the graph
    tf.reset_default_graph()

    # create placeholder for input data
    X, Y = create_placeholder(n0_h, n0_w, n0_c, output_dim)

    # initialize parameter
    W1, W2, b1, b2 = initialize_parameter(hparameters)

    # forward propagation
    S = forward_propagation(hparameters, X, W1, W2, output_dim)

    # compute cost
    cost = tf.reduce_mean(1 / N * tf.square(S - Y))

    # backpropagation
    optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    # initialize
    init = tf.global_variables_initializer()

    costs = []
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(iteration_num):
            for train_X, train_Y in generate_mini_batch(N, mini_batch_size):
                _, temp_cost = sess.run([optimize, cost], feed_dict={X:train_X, Y:train_Y})
                if iteration_num % 10 == 0:
                    print(temp_cost)
        Y_hat = predict()


if __name__ == "__main__":
    main()
