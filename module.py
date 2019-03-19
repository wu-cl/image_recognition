import tensorflow as tf
import numpy as np

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


output = cnn_structure()
saver = tf.train.Saver(max_to_keep=1)
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint('.'))
predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)


def crack_captcha(captcha_image):
    text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
    text = text_list[0].tolist()
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    i = 0
    for n in text:
        vector[i * CHAR_SET_LEN + n] = 1
        i += 1
    return vec2text(vector)


def image_recognition(img):
    img = img.convert('L')

    img = np.array(img)

    img = img[5:35, 10:90]  # 裁剪



    img = img.flatten() / 255

    predict_text = crack_captcha(img)
    return predict_text
