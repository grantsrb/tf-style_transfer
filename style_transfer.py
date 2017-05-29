import numpy as np
import tensorflow as tf
from scipy.optimize import fmin_l_bfgs_b
import utils.vgg16_avg as vgg
from PIL import Image
import time
import sys

# May 28th, 2017
# Satchel Grant

# Style transfer using Vgg16

# The style of the style-image is transferred to the specified image
# Pass the arguments in the command line or follow the prompt

try:
    img_path = str(sys.argv[1])
    style_path = str(sys.argv[2])
    img_size = str(sys.argv[3]).split(',')
    img_size = (int(img_size[0]), int(img_size[1]))
    eval_conv = float(sys.argv[3])
    styleweights = sys.argv[4]
    if '[' in styleweights or ']' in styleweights:
        styleweights = styleweights[1:-1]
    styleweights = [float(x) for x in styleweights.split(',')]
    img_weight = float(sys.argv[5])
    style_weight = float(sys.argv[6])
    n_iterations = int(sys.argv[7])
    save_path = str(sys.argv[8])
except:
    img_path = input("Image path: ")
    style_path = input("Style path: ")
    img_h = input("Image height [300]: ")
    if img_h == '':
        img_h = 300
    else:
        img_h = int(img_h)
    img_w = input("Image width [300]: ")
    if img_w == '':
        img_w = 300
    else:
        img_w = int(img_w)
    eval_conv = input("Vgg16 evaluation layer [4.1]: ")
    if eval_conv == '':
        eval_conv = 4.1
    else:
        eval_conv = float(eval_conv)
    styleweights = input("Style layer weights [0,.5,1,1]: ")
    if styleweights == '':
        styleweights = [0.,.5,1.,1.]
    else:
        styleweights = [float(x) for x in styleweights]
    img_weight = input("Image cost weight [1]: ")
    if img_weight == '':
        img_weight = 1.
    else:
        img_weight = float(img_weight)
    style_weight = input("Style cost weight [1]: ")
    if style_weight == '':
        style_weight = 1.
    else:
        style_weight = float(style_weight)
    n_iterations = input('Number of iterations [10]: ')
    if n_iterations == '':
        n_iterations = 10
    else:
        n_iterations = int(n_iterations)
    save_path = input("Save name [combined.jpg]: ")
    if save_path == '':
        save_path = 'combined.jpg'


def loss_grads(gen_img, sess):
    cost, grads = sess.run([combined_loss, combined_grads], feed_dict={tfimg:gen_img})
    return cost, np.asarray(grads)

class Evaluator(object):
    def __init__(self, fxn, shape, sess):
        self.fxn = fxn
        self.shape = shape
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def loss(self, x):
        loss_, self.grad_vals = self.fxn(x.reshape(self.shape), self.sess)
        return loss_.astype(np.float64)

    def grads(self, x):
        return self.grad_vals.flatten().astype(np.float64)

def solve_image(eval_obj, n_iters, gen_img):
    for i in range(n_iters):
        gen_img, min_val, info = fmin_l_bfgs_b(eval_obj.loss, gen_img.flatten(),
                                         fprime=eval_obj.grads, maxfun=20)
        gen_img = np.clip(gen_img, 0,1)
        print('Iteration:',i,'- Loss:', min_val, end='\r')
    eval_obj.sess.close()
    return gen_img

def gram_matrix(activs):
    reshape1 = tf.transpose(activs,perm=[0,3,1,2])
    reshape2 = tf.transpose(activs,perm=[0,3,2,1])
    output = tf.matmul(reshape1,reshape2)
    return tf.reshape(output,[-1])/(224*224)

def style_loss(loss1, loss2):
    return tf.reduce_mean(tf.squared_difference(gram_matrix(loss1), gram_matrix(loss2)))

img = Image.open(img_path)
img = img.resize((224,224))
img = np.asarray(img).astype(np.float32)
img = img / 255 # scaled for vgg16

style = Image.open(style_path)
style = style.resize((img.shape[1],img.shape[0]))
style = np.asarray(style) / 255 # scaled for vgg16

gen_img = np.random.uniform(0, .5, img.shape)

vmodel = vgg.Vgg16()


#### Image
tfimg = tf.placeholder(tf.float32, [None]+[s for s in img.shape], name='tfimg')
trgt = vmodel.run_convs(tfimg, eval_conv)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    target_img = sess.run(trgt, feed_dict={tfimg:[img]})


trgt_const = tf.convert_to_tensor(target_img, dtype=tf.float32)
img_output = vmodel.run_convs(tfimg, eval_conv)


#### Style
tfstyle = tf.placeholder(tf.float32, [1]+[s for s in style.shape], name="tfstyle")
styleconvs = [vmodel.run_convs(tfstyle, i+.2) for i in range(1,4)]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    target_styles = sess.run(styleconvs, feed_dict={tfstyle:[style]})

stylelayers = [tf.convert_to_tensor(s, dtype=tf.float32) for s in target_styles]
imglayers = [vmodel.run_convs(tfimg, i+.2) for i in range(1,4)]


#### Combined
combined_loss = img_weight*tf.reduce_mean(tf.squared_difference(img_output, trgt_const)) + style_weight*tf.reduce_sum([w*style_loss(l1,l2) for l1,l2,w in zip(stylelayers, imglayers, styleweights)])

combined_grads = tf.gradients(combined_loss, tfimg)

with tf.Session() as sess:
    evaluator = Evaluator(loss_grads, [1]+[s for s in gen_img.shape], sess)
    combined_img = solve_image(evaluator, n_iterations,
                            np.reshape(gen_img, [1]+[s for s in gen_img.shape]))

combined_img = combined_img.reshape(gen_img.shape)
save_img = Image.fromarray((combined_img*255).astype(np.uint8))
save_img.save(save_path)

outimg = np.zeros([combined_img.shape[0],combined_img.shape[1]*3,3])
outimg[:,:outimg.shape[1]//3,:] = img[:,:,:]
outimg[:,outimg.shape[1]//3:2*outimg.shape[1]//3,:] = combined_img[:,:,:]
outimg[:,2*outimg.shape[1]//3:,:] = style[:,:,:]

save_img = Image.fromarray((outimg*255).astype(np.uint8))
save_img.save(save_path[:-4]+'_combined.jpg')
