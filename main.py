import os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from captcha.image import ImageCaptcha
import numpy as np
import random

ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

char_set = ALPHABET  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(char_set)
IMAGE_HEIGHT = 30
IMAGE_WIDTH = 80
MAX_CAPTCHA = 4


def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):
        k = ord(c) - ord('A')

        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        char_code = char_idx + ord('A')

        text.append(chr(char_code))
    return "".join(text)


def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img


def get_bin_table(threshold=135):
    table = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    return table


image_data = []
text_data = []
file_count = 0


def noise(img):
    for i in range(1, IMAGE_HEIGHT - 1):
        for j in range(1, IMAGE_WIDTH - 1):
            if img[i, j] == 0 and img[i - 1, j] == 1 and img[i + 1, j] == 1:
                img[i, j] = 1
    for i in range(1, IMAGE_HEIGHT - 1):
        for j in range(1, IMAGE_WIDTH - 1):
            if img[i, j] == 0 and img[i, j - 1] == 1 and img[i, j + 1] == 1:
                img[i, j] = 1
    for i in range(1, IMAGE_HEIGHT - 1):
        for j in range(1, IMAGE_WIDTH - 1):
            if img[i, j] == 0 and img[i - 1, j] == 1 and img[i + 1, j] == 1:
                img[i, j] = 1
    img[0:1, :] = 1
    img[IMAGE_HEIGHT - 1:IMAGE_HEIGHT, :] = 1
    img[:, 0:1] = 1
    img[:, IMAGE_WIDTH - 1:IMAGE_WIDTH] = 1
    return img


def read_img(rootdir):
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    random.shuffle(list)
    for i in range(0, len(list)):
        if list[i] != ".DS_Store":
            path = os.path.join(rootdir, list[i])
            text = list[i].split(".")[0]  # 取文件名-验证码结果
            text_data.append(text)

            img = Image.open(path)
            # img = img.resize((200, 80))

            img = img.convert('L')

            # table = get_bin_table()
            # img = img.point(table, '1')

            img = np.array(img)

            img = img[5:35, 10:90]  # 裁剪

            # img = noise(img)

            # img = convert2gray(img)  # 转换为灰度

            # plt.imshow(img)
            # plt.show()

            image_data.append(img)

    global file_count
    file_count = len(list) - 1


def get_next_batch(batch_size=128, type='train'):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    if type == 'train':
        list = range(file_count - 2000)
    else:
        list = range(file_count - 2000 + 1, file_count)

    list = random.sample(list, batch_size)

    count = 0
    for i in list:
        image = np.array(image_data[i])
        text = text_data[i]

        batch_x[count, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[count, :] = text2vec(text)
        count += 1
    return batch_x, batch_y


X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout


def cnn_structure(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    wc1 = tf.get_variable(name='wc1', shape=[3, 3, 1, 32], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    # wc1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    bc1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='SAME'), bc1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    wc2 = tf.get_variable(name='wc2', shape=[3, 3, 32, 64], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    # wc2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    bc2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='SAME'), bc2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    wc3 = tf.get_variable(name='wc3', shape=[3, 3, 64, 128], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    # wc3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 128]))
    bc3 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, wc3, strides=[1, 1, 1, 1], padding='SAME'), bc3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    wd1 = tf.get_variable(name='wd1', shape=[8 * 20 * 32, 1024], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    # wd1 = tf.Variable(w_alpha * tf.random_normal([7*20*128,1024]))
    bd1 = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, wd1.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
    dense = tf.nn.dropout(dense, keep_prob)

    wout = tf.get_variable('name', shape=[1024, MAX_CAPTCHA * CHAR_SET_LEN], dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer())
    # wout = tf.Variable(w_alpha * tf.random_normal([1024, max_captcha * char_set_len]))
    bout = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, wout), bout)
    return out


def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 16]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([16]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 16, 32]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 32]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 8, 512]))
    b_d = tf.Variable(b_alpha * tf.random_normal([512]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([512, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return out


def train_crack_captcha_cnn():
    # output = crack_captcha_cnn()
    output = cnn_structure()
    # loss
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(100)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print(step, loss_)

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(200, type="valid")
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                if acc > 0.98:
                    saver.save(sess, "crack_capcha.model", global_step=step)
                    break

            step += 1


read_img("train")
train_crack_captcha_cnn()
