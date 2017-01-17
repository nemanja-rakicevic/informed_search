
import PIL
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

DWN_RATE = 4
# Load background image
img_background = PIL.Image.open('pic_background.png', 'r')
img_bd = img_background.resize((img_background.size[0]/DWN_RATE, img_background.size[1]/DWN_RATE), PIL.Image.ANTIALIAS)

# Load puck image
img_puck = PIL.Image.open('pic_puck.png', 'r')
img_pd = img_puck.resize((img_puck.size[0]/DWN_RATE, img_puck.size[1]/DWN_RATE), PIL.Image.ANTIALIAS)

# Load end and end2 image (use either depending on the y coordinate)
img_goal = PIL.Image.open('pic_goalp1.png', 'r')
img_gd = img_goal.resize((img_goal.size[0]/DWN_RATE, img_goal.size[1]/DWN_RATE), PIL.Image.ANTIALIAS)


# Convert camera coordinate to robot coordinate
def px2cm(in_px_y, in_px_x, img):
    w, h = img.size
    px_y = 77-in_px_y
    px_x = w-in_px_y
    # output is (X, Y) in robot coordinate system
    return 0.05*np.exp(0.05231*px_y)*100, 1.2*px_x


# Generate image (60k)
def generateImage(samples, img_bd, img_gd, img_pd):
    batch_img = []
    batch_labels = []
    # Generate goal position within the ellipse
    r = img_bd.size[0]/3.8      # width
    h = img_bd.size[0]/2.47     # x-coord
    k = img_bd.size[1]/3        # y-coord
    ratio = 0.55
    # Loop and generate batch
    for s in range(samples):
        img = img_bd.copy()
        img_gmask = img_gd.copy()
        img_pmask = img_pd.copy()
        # Generate parameters
        th = np.random.choice(np.linspace(0,2*np.pi,100))
        phi = np.random.choice(np.linspace(0,1,100))
        x_g = np.sqrt(phi) * np.cos(th)
        y_g = np.sqrt(phi) * np.sin(th)
        x_g = int(round(h + x_g * r))
        y_g = int(round(k + y_g * ratio*r))
        t1 = abs(img.size[0]/2-x_g)/10
        t2 = abs(img.size[1]/2-y_g)/10
        img_gmask = img_gmask.resize( (img_gmask.size[0]-t1, img_gmask.size[1]-t2), PIL.Image.ANTIALIAS)
        img.paste(img_gmask, (x_g, y_g), img_gmask)
        # Generate puck position near the sticke
        x_p = np.random.randint(img.size[0]/2-0.05*img.size[0],img.size[0]/2+0.05*img.size[0])
        y_p = int(-0.4*x_p+99)
        img.paste(img_pmask, (x_p, y_p), img_pmask)
        # Convert coordinates
        rob_x_p, rob_y_p = px2cm(y_p, x_p, img)
        rob_x_g, rob_y_g = px2cm(y_g, x_g, img)
        # Calculate angle and distance
        dx = rob_x_g - rob_x_p
        dy = rob_y_g - rob_y_p
        # print dx, dy
        angle = np.arctan2(dy, dx) * 180 / np.pi
        dist = np.sqrt(dx**2 + dy**2)
        # Save image and labels
        batch_img.append(np.asarray(img).flatten()/255.)
        batch_labels.append(np.array([angle, dist]))

    #return the images and labels
    return (np.array(batch_img), np.array(batch_labels))


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#### LEARNING PART
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 120*160*3]) #19200
y_ = tf.placeholder(tf.float32, [None, 2])

# DEFINING THE LAYERS
x_image = tf.reshape(x, [-1,120,160,3])
# 1st convolutional
W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# 2nd convolutional
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# 3rd convolutional
W_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)
# fully connected
W_fc1 = weight_variable([15 * 20 * 128, 1024])
b_fc1 = bias_variable([1024])
h_pool1_flat = tf.reshape(h_pool1, [-1, 15 * 20 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
# Regularisation
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# Output - fully connected
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# TRAINING
batch_size = 100
cost_function = tf.reduce_sum(tf.pow(y_conv - y_, 2))/(2 * batch_size)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost_function)

correct_prediction = tf.equal(y_conv, y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

for i in range(1):
    batch = generateImage(batch_size, img_bd, img_gd, img_pd)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))




######################################################
### HELPER CODE ###
######################################################

# ### ELIPSE of positions (downsampled 4x)
# img = img_bd.copy()
# draw = PIL.ImageDraw.Draw(img)
# th = np.linspace(-np.pi,np.pi,100)
# r = 40
# h = 65
# k = 40
# x = h + r*np.cos(th)
# y = k - 0.55 * r*np.sin(th)
# draw.point(zip(x,y),fill=128)
# plt.imshow(img)
# plt.show()

# ### CALIBRATION
# cal = PIL.Image.open('calibration.png', 'r')
# img = img_bd.copy()
# draw = PIL.ImageDraw.Draw(img)
# m = img.size[0]/2
# n = img.size[1]
# for n in range(0, img.size[1], 5):
#     w=1
#     if not n%100:
#         w=2
#     draw.line([m, n, m+10, n],fill=255, width=w)

# plt.imshow(img)
# plt.show()

# ### CONVERSION
# def px2m_original(px):
#     return 0.0628*np.exp(0.0122*px)