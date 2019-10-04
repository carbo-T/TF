# -*- codeing: utf-8 -*-
import numpy as np
import tensorflow as tf
# 以下两行用于解决错误： _tkinter.TclError: no display name and no $DISPLAY environment variable
# import matplotlib as mpl
# mpl.use("Agg")
import matplotlib.pyplot as plt


def rnn_forward_prop():
    # define input
    x = [1, 2]
    state = [0.0, 0.0]

    # define inner weight and bias
    w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    w_cell_input = np.asarray([0.5, 0.6])
    b_cell = np.asarray([0.1, -0.1])

    # define output weight and bias
    w_output = np.asarray([1.0, 2.0])
    b_outoput = 0.1

    for i in range(len(x)):
        before_activation = np.dot(state, w_cell_state) + x[i] * w_cell_input + b_cell
        # tanh as activation function
        state = np.tanh(before_activation)

        final_output = np.dot(state, w_output) + b_outoput

        print("before activation: ", before_activation)
        print("state: ", state)
        print("output: ", final_output, "\n")


def lstm_structure():
    lstm_hidden_size = 1
    batch_size = 10
    num_steps = 10  # the length of data
    num_of_layers = 5
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)
    # 带dropout的多层循环神经网络
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.DropoutWrapper(lstm(lstm_hidden_size)) for _ in range(num_of_layers)]
    )
    # state is a LSTEMStateTuple instance with two Tensors, state.c and state.h
    state = lstm.zero_state(batch_size, tf.float32)
    loss = 0.0
    for i in range(num_steps):
        # assign variables in the first time step, and reuse variables in the following steps
        if i > 0:
            tf.get_variable_scope().reuse_variables()

        # current_input represent xt, input prev state ht-1 and ct-1
        # lstm_output can be sent to other layers, state can be used for the next time step
        lstm_output, state = lstm(current_input, state)

        # connect to a fc layer to generate the final output
        final_output = fully_connected(lstm_output)

        # calc current loss
        loss += calc_loss(final_output, expected_output)


HIDDEN_SIZE = 30  # No. of hidden node
NUM_LAYERS = 2

TIMESTEPS = 10
TRAINING_STEPS = 10000
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000
TEST_EXAMPLES = 1000
SAMPlE_GAP = 0.01  # 采样间隔


def generate_data(seq):
    x = []
    y = []
    # input from i to i+TIMESTEPS-1
    # output i+TIMESTEPS, which uses TIMESTEPS samples to predict the TIMESTEPSth result
    for i in range(len(seq) - TIMESTEPS):
        x.append([seq[i:i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def lstm_model(x, y, is_training):
    with tf.name_scope("rnn"):
        cell = tf.nn.rnn_cell.MultiRNNCell([
            tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)
        ])
        # output for every timestep, shape=[batch_size, time, HIDDEN_SIZE]
        outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        output = outputs[:, -1, :]
        # add a fc layer for output
        predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)

    if not is_training:
        return predictions, None, None

    with tf.name_scope("loss"):
        loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    # create model optimizer
    with tf.name_scope("train"):
        train_op = tf.contrib.layers.optimize_loss(
            loss, tf.train.get_global_step(),
            optimizer="Adagrad", learning_rate=0.1
        )

    return predictions, loss, train_op


def train(sess, train_x, train_y, writer):
    # generate dataset
    dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    dataset = dataset.repeat().shuffle(1000).batch(BATCH_SIZE)
    x, y = dataset.make_one_shot_iterator().get_next()

    with tf.variable_scope("model"):
        predictions, loss, train_op = lstm_model(x, y, True)

    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    for i in range(TRAINING_STEPS):

        if i % 500 == 0:
            # config necessary info when training
            run_options = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE
            )
            # record proto when training
            run_metadata = tf.RunMetadata()

            summary, _, l = sess.run([merged, train_op, loss], options=run_options, run_metadata=run_metadata)
            writer.add_run_metadata(run_metadata, 'step%03d' % i)
            writer.add_summary(summary, i)
            print("train step: " + str(i) + ", loss: " + str(l))
        else:
            _, l = sess.run([train_op, loss])


def run_eval(sess, test_x, test_y):
    # generate dataset
    dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    dataset = dataset.batch(1)
    x, y = dataset.make_one_shot_iterator().get_next()

    with tf.variable_scope("model", reuse=True):
        # unnecessary to input real y value
        prediction, _, _ = lstm_model(x, [0.0], False)

    predictions = []
    labels = []
    for i in range(TEST_EXAMPLES):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    # root mean square error
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("MSE: %.4f" % rmse)

    # draw fitting lines
    plt.figure(1)
    plt.plot(predictions, label="predictions")
    plt.plot(labels, label="real_sin")
    plt.legend()
    plt.show()


def sample_generator(data):
    amps = [0.6, 0.2, 0.8]
    phases = [0.1, 0.4, 0.7]
    return amps[0] * np.sin(data + np.pi * phases[0]) + amps[1] * np.sin(data + np.pi * phases[1]) + amps[2] * np.sin(
        data + np.pi * phases[2])  # + np.random.rand(len(data))*0.02


def rnn_example():
    test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPlE_GAP
    test_end = test_start * 2
    train_x, train_y = generate_data(
        sample_generator(np.linspace(0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
    test_x, test_y = generate_data(
        sample_generator(np.linspace(test_start, test_end, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))

    # writer = tf.summary.FileWriter("log", tf.get_default_graph())
    # with tf.device("/gpu:0"):
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("log", sess.graph)
        print("train")
        train(sess, train_x, train_y, writer)
        print("eval")
        run_eval(sess, test_x, test_y)
        writer.close()


def main():
    # rnn_forward_prop()
    rnn_example()


if __name__ == "__main__":
    main()
