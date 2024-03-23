import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras.layers import Flatten
import tensorflow_probability as tfp
tfd = tfp.distributions
kl = tf.keras.losses.KLDivergence()

LARGE_NUM = 1e9


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def compute_logits_and_labels(hidden, temperature=0.5, hidden_norm=True):
    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)

    hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
    logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature
    return logits_ab, logits_aa, logits_ba, logits_bb, labels


class NTXentLoss(tf.keras.losses.Loss):

    def __init__(self, hidden_norm=True, temperature=1.0, batch_size = 4, *args, **kwargs):
        super().__init__(name='contrastive_loss')
        self.hidden_norm = hidden_norm
        self.temperature = temperature
        self.batch_size = batch_size
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def call(self, y_true, hidden):
#         hidden = hidden[self.batch_size*2:]
        
        logits_ab, logits_aa, logits_ba, logits_bb, labels = compute_logits_and_labels(hidden,
                                                                                       temperature=self.temperature,
                                                                                       hidden_norm=self.hidden_norm)
        loss_a = self.cross_entropy_loss(labels, tf.concat([logits_ab, logits_aa], 1))
        loss_b = self.cross_entropy_loss(labels, tf.concat([logits_ba, logits_bb], 1))
        loss = loss_a + loss_b
        return loss

# class ConsistencyLoss(tf.keras.losses.Loss):
#     
#     def __init__(self, batch_size = 4, output_inpaint_model=None, *args, **kwargs):
#         super().__init__(name='consistency_loss')
#         self.batch_size = batch_size
#         self.output_inpaint_model = output_inpaint_model
#     
#     def call(self, y_true, hidden):
#         inputs_img = hidden[0][self.batch_size*2:]
#         inputs_mask = hidden[1][self.batch_size*2:]
#         id_pairs = hidden[2]
#         
#         # the encoder outputs with projection head
#         projected_representation = hidden[3][self.batch_size*2:]
#         
#         # the imputed outputs
#         x = (inputs_img, inputs_mask)
#         imputed_img = self.output_inpaint_model(x)
#         
#         # imputed img thickness density distribution
#         imputed_img_gray = rgb2gray(imputed_img)
#         imputed_img_flatten = Flatten()(imputed_img_gray) / 255.
#         imputed_img_flatten = tf.sort(imputed_img_flatten, axis=-1,direction='ASCENDING',name=None)
#         imputed_img_mean = tf.math.reduce_mean(imputed_img_flatten)
#         imputed_img_std = tf.math.reduce_std(imputed_img_flatten)
#         imputed_img_density_dist = tfd.Normal(loc=imputed_img_mean, scale=imputed_img_std)
#         imputed_imgs = tf.split(imputed_img_density_dist, 2, 0)
#         imputed_dist1 = imputed_imgs[0]
#         imputed_dist2 = imputed_imgs[1]
#         
#         # projected img thickness density distribution
#         projected_img_gray = rgb2gray(projected_representation)
#         projected_img_flatten = Flatten()(projected_img_gray) / 255.
#         projected_img_flatten = tf.sort(projected_img_flatten, axis=-1,direction='ASCENDING',name=None)
#         projected_img_mean = tf.math.reduce_mean(projected_img_flatten)
#         projected_img_std = tf.math.reduce_std(projected_img_flatten)
#         projected_img_density_dist = tfd.Normal(loc=projected_img_mean, scale=projected_img_std)
#         projected_imgs = tf.split(projected_img_density_dist, 2, 0)
#         projected_dist1 = projected_imgs[0]
#         projected_dist2 = projected_imgs[1]
#         
#         # local proximities
#         imputed_pro = 1.0 / (1 + (tf.math.exp(-tf.reduce_sum( tf.multiply(imputed_dist1, imputed_dist2), 1, keep_dims=True) * 1.0)))
#         projected_pro = 1.0 / (1 + (tf.math.exp(-tf.reduce_sum( tf.multiply(projected_dist1, projected_dist2), 1, keep_dims=True) * 1.0)))
#         
#         
#         kl_loss = kl(projected_pro, imputed_pro)
#         
#         return kl_loss
