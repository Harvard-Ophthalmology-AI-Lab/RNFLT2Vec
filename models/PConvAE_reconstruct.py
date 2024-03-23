import os
import sys
import numpy as np
from datetime import datetime

import tensorflow as tf
# from keras.models import Model
# from keras.models import load_model
# from keras.optimizers import Adam
# from keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, BatchNormalization, Activation, Lambda, Layer, GlobalAveragePooling2D, Flatten
# from keras.layers.merge import Concatenate
# from keras.applications import VGG16
# from keras import backend as K
# from keras.utils.multi_gpu_utils import multi_gpu_model

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, BatchNormalization, Activation, Lambda, GlobalAveragePooling2D, Flatten, Layer, Multiply
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.applications import VGG16
import tensorflow.keras.backend as K
# from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from enum import Enum, unique
from utils import model_util
import tensorflow_probability as tfp
from models import loss
tfd = tfp.distributions
kl = tf.keras.losses.KLDivergence()

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from models.PartialConv import PConv2D

class BaseEnum(Enum):
    @classmethod
    def values(cls):
        list(map(lambda x: x.value, cls))

@unique
class EncoderReduction(BaseEnum):
    """
    Define various methods for reducing the encoder output of shape (w, h, f) to
    """
    GA_POOLING = 'ga_pooling'
    FLATTEN = 'flatten'

class PConvUnet_Reconstruct(object):

    def __init__(self, img_rows=256, img_cols=256, reconstruct_w=1.0, contrastive_w=0.001, consistency_w=0.001,
                 batch_size=4, vgg_weights="imagenet", 
                 inference_only=False, net_name='default', gpus=1, vgg_device=None,
                 encoder_reduction=EncoderReduction.GA_POOLING,
                         projection_dim=100,
                         projection_head_layers=2):
        """Create the PConvUnet. If variable image size, set img_rows and img_cols to None
        
        Args:
            img_rows (int): image height.
            img_cols (int): image width.
            vgg_weights (str): which weights to pass to the vgg network.
            inference_only (bool): initialize BN layers for inference.
            net_name (str): Name of this network (used in logging).
            gpus (int): How many GPUs to use for training.
            vgg_device (str): In case of training with multiple GPUs, specify which device to run VGG inference on.
                e.g. if training on 8 GPUs, vgg inference could be off-loaded exclusively to one GPU, instead of
                running on one of the GPUs which is also training the UNet.
        """
        
        # Settings
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.batch_size = batch_size
        self.img_overlap = 30
        self.inference_only = inference_only
        self.net_name = net_name
        self.w1 = reconstruct_w
        self.w2 = contrastive_w
        self.w3 = consistency_w
        self.gpus = gpus
        self.vgg_device = vgg_device
        self.encoder_reduction = encoder_reduction
        self.projection_dim = projection_dim
        self.projection_head_layers = projection_head_layers
        

        # Scaling for VGG input
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Assertions
        assert self.img_rows >= 256, 'Height must be >256 pixels'
        assert self.img_cols >= 256, 'Width must be >256 pixels'

        # Set current epoch
        self.current_epoch = 0
        
        # VGG layers to extract features from (first maxpooling layers, see pp. 7 of paper)
        self.vgg_layers = [3, 6, 10]

        # Instantiate the vgg network
        if self.vgg_device:
            with tf.device(self.vgg_device):
                self.vgg = self.build_vgg(vgg_weights)
        else:
            self.vgg = self.build_vgg(vgg_weights)
        
        # Create UNet-like model
        if self.gpus >= 1:
            self.model, inputs_mask, con_loss = self.build_pconv_unet()
#             self.reconstruct_loss = self.ae_loss_total(inputs_mask)
#             self.consistency_loss = self.consistencyLoss(inputs_img, inputs_mask, id_pairs, output_inpaint_model)
            self.compile_pconv_unet(self.model, inputs_mask, con_loss)            
        else:
            with tf.device("/cpu:0"):
                self.model, inputs_mask, con_loss = self.build_pconv_unet()
            self.model = multi_gpu_model(self.model, gpus=self.gpus)
#             self.reconstruct_loss = self.ae_loss_total(inputs_mask)
#             self.consistency_loss = self.consistencyLoss(inputs_img, inputs_mask, id_pairs, output_inpaint_model)
            self.compile_pconv_unet(self.model, inputs_mask, con_loss)
        
    def build_vgg(self, weights="imagenet"):
        """
        Load pre-trained VGG16 from keras applications
        Extract features to be used in loss function from last conv layer, see architecture at:
        https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py
        """        
            
        # Input image to extract features from
        img = Input(shape=(self.img_rows, self.img_cols, 3))

        # Mean center and rescale by variance as in PyTorch
        processed = Lambda(lambda x: (x-self.mean) / self.std)(img)
        
        # If inference only, just return empty model        
        if self.inference_only:
            model = Model(inputs=img, outputs=[img for _ in range(len(self.vgg_layers))])
            model.trainable = False
            model.compile(loss='mse', optimizer='adam')
            return model
                
        # Get the vgg network from Keras applications
        if weights in ['imagenet', None]:
            vgg = VGG16(weights=weights, include_top=False)
        else:
            vgg = VGG16(weights=None, include_top=False)
            vgg.load_weights(weights, by_name=True)

        # Output the first three pooling layers
        outputs = [vgg.layers[i](processed) for i in self.vgg_layers]        
        
        # Create model and compile
        model = Model(img, outputs)
        model.trainable = False
        model.compile(loss='mse', optimizer='adam')

        return model
        
    def build_pconv_unet(self, train_bn=True):      

        # INPUTS
        inputs_img = Input((self.img_rows, self.img_cols, 3), name='inputs_img')
        inputs_mask = Input((self.img_rows, self.img_cols, 3), name='inputs_mask')
        inputs_img_cc = Input((self.img_rows, self.img_cols, 3), name='inputs_img_cc')
        inputs_mask_cc = Input((self.img_rows, self.img_cols, 3), name='inputs_mask_cc')
        id_pairs = Input((1,), dtype=tf.int32, name='id_pairs')
        
        # ENCODER
        def encoder_layer(img_in, mask_in, filters, kernel_size, bn=True):
            conv, mask = PConv2D(kernel_size=kernel_size, n_channels=3, mono=False, filters=filters,
                                 strides=2, padding='same')([img_in, mask_in])
            if bn:
                conv = BatchNormalization(name='EncBN'+str(encoder_layer.counter))(conv, training=train_bn)
            conv = Activation('relu')(conv)
            encoder_layer.counter += 1
            return conv, mask
        encoder_layer.counter = 0
        
        e_conv1, e_mask1 = encoder_layer(inputs_img, inputs_mask, 64, 7, bn=False)
        e_conv2, e_mask2 = encoder_layer(e_conv1, e_mask1, 128, 5)
        e_conv3, e_mask3 = encoder_layer(e_conv2, e_mask2, 256, 5)
        e_conv4, e_mask4 = encoder_layer(e_conv3, e_mask3, 512, 3)
        e_conv5, e_mask5 = encoder_layer(e_conv4, e_mask4, 512, 3)
        e_conv6, e_mask6 = encoder_layer(e_conv5, e_mask5, 512, 3)
        e_conv7, e_mask7 = encoder_layer(e_conv6, e_mask6, 512, 3)
        e_conv8, e_mask8 = encoder_layer(e_conv7, e_mask7, 512, 3)
        e_conv8 = Layer(name='embeds')(e_conv8)
        print(self.encoder_reduction)
        
        # LATENT REPRSENTATION
        reduced = self.reduce_encoder_output(encoder_output=e_conv8, encoder_reduction=self.encoder_reduction)
        latent_representation = Layer(name='encoder_output')(reduced) # final RNFLT representations
        projected_representation = self.add_contrastive_output(input=latent_representation, projection_dim=self.projection_dim,
                                        projection_head_layers=self.projection_head_layers)
        
        # DECODER
        def decoder_layer(img_in, mask_in, e_conv, e_mask, filters, kernel_size, bn=True):
            up_img = UpSampling2D(size=(2,2))(img_in)
            up_mask = UpSampling2D(size=(2,2))(mask_in)
            concat_img = Concatenate(axis=3)([e_conv,up_img])
            concat_mask = Concatenate(axis=3)([e_mask,up_mask])
            conv, mask = PConv2D(kernel_size=kernel_size, padding='same', filters=filters)([concat_img, concat_mask])
            if bn:
                conv = BatchNormalization()(conv)
            conv = LeakyReLU(alpha=0.2)(conv)
            return conv, mask
            
        d_conv9, d_mask9 = decoder_layer(e_conv8, e_mask8, e_conv7, e_mask7, 512, 3)
        d_conv10, d_mask10 = decoder_layer(d_conv9, d_mask9, e_conv6, e_mask6, 512, 3)
        d_conv11, d_mask11 = decoder_layer(d_conv10, d_mask10, e_conv5, e_mask5, 512, 3)
        d_conv12, d_mask12 = decoder_layer(d_conv11, d_mask11, e_conv4, e_mask4, 512, 3)
        d_conv13, d_mask13 = decoder_layer(d_conv12, d_mask12, e_conv3, e_mask3, 256, 3)
        d_conv14, d_mask14 = decoder_layer(d_conv13, d_mask13, e_conv2, e_mask2, 128, 3)
        d_conv15, d_mask15 = decoder_layer(d_conv14, d_mask14, e_conv1, e_mask1, 64, 3)
        d_conv16, d_mask16 = decoder_layer(d_conv15, d_mask15, inputs_img, inputs_mask, 3, 3, bn=False)
        outputs = Conv2D(3, 1, activation = 'sigmoid', name='inpaint')(d_conv16)
        
        # output from the projection layer
        output_projection_model = Model(inputs=[inputs_img, inputs_mask], outputs=projected_representation, name='embed_model')
        output_projection_model.compile(loss='mse', optimizer='adam')
        projection_out = output_projection_model([inputs_img_cc, inputs_mask_cc], training=True)
        projection_out = Layer(name='contrast')(projection_out)
        
        # output from the imputed layer
        output_inpaint_model = Model(inputs=[inputs_img, inputs_mask], outputs=outputs, name='inpaint_model')
#         output_inpaint_model.compile(loss='mse', optimizer='adam')
#         output_inpaint_model.trainable = False
        output_inpaint = output_inpaint_model([inputs_img_cc, inputs_mask_cc])
#         output_inpaint = output_inpaint_model.predict([inputs_img_cc, inputs_mask_cc])
#         imputed_out = Layer(name='consist', name='consist')(output_inpaint)
        imputed_out = K.stop_gradient(output_inpaint)
        imputed_out = Layer(name='consist')(imputed_out)
        
        print('112:', projection_out.shape, imputed_out.shape)
        
#         con_loss = Lambda(lambda x: self.consistencyLossLayer(*x), name="con_loss")([projection_out, imputed_out, id_pairs])
        
        # Setup the model inputs / outputs
        model = Model(inputs=[inputs_img, inputs_mask, inputs_img_cc, inputs_mask_cc, id_pairs], 
                      outputs=[outputs])

        return model, inputs_mask, None

    def consistencyLossLayer(self, prejection_out, imputed_out, id_pairs):

        imputed_img = tf.image.rgb_to_grayscale(imputed_out)
        imputed_img = tf.squeeze(imputed_img, axis=-1)
        imputed_img = tf.reshape(imputed_img, (-1, self.img_rows*self.img_cols))
        
        print('imputed_img:', imputed_img.shape)
        
        imputed_img_gray = K.gather(imputed_img, id_pairs)
        imputed_img_flatten =tf.reshape(imputed_img_gray,(-1,self.img_rows*self.img_cols))
        imputed_img_flatten = imputed_img_flatten
        imputed_img_flatten = tf.sort(imputed_img_flatten, axis=-1,direction='ASCENDING',name=None)
        imputed_img_mean =  tf.expand_dims(tf.math.reduce_mean(imputed_img_flatten, axis=1), -1)
        imputed_img_std = tf.expand_dims(tf.math.reduce_std(imputed_img_flatten, axis=1), -1)
#             imputed_img_density_dist = tfd.Normal(loc=imputed_img_mean, scale=imputed_img_std).log_prob(imputed_img_flatten)
        print('imputed_img_mean:', imputed_img_mean.shape)
        print('imputed_img_std:', imputed_img_std.shape)
        imputed_img_density_dist = tf.concat([imputed_img_mean, imputed_img_std], axis=-1)
        print('imputed_img_density_dist:', imputed_img_density_dist.shape)
        
        imputed_dist1, imputed_dist2 = tf.split(imputed_img_density_dist, 2, 0)
#          = imputed_imgs[0]
#          = imputed_imgs[1]
#         imputed_dist1 = K.gather(imputed_imgs[0], id_pairs[0,:])
#         imputed_dist2 = K.gather(imputed_imgs[1], id_pairs[1,:])
        
        # projected img thickness density distribution
        print('prejection_out:', prejection_out.shape)
        projected_representation = tf.squeeze(K.gather(prejection_out, id_pairs), 1)
        projected_img_gray = projected_representation
        print('projected_img_gray:', projected_img_gray.shape)
#         projected_img_flatten = tf.keras.utils.normalize(projected_img_gray, axis=1)
        projected_img_flatten = projected_img_gray
        projected_img_flatten = tf.sort(projected_img_flatten, axis=-1,direction='ASCENDING',name=None)
        projected_img_mean = tf.expand_dims(tf.math.reduce_mean(projected_img_flatten, axis=1), -1)
        projected_img_std = tf.expand_dims(tf.math.reduce_std(projected_img_flatten, axis=1), -1)
        
        projected_img_density_dist = tf.concat([projected_img_mean, projected_img_std], axis=-1)
        print('projected_img_density_dist:', projected_img_density_dist.shape)
#             projected_img_density_dist = tfd.Normal(loc=projected_img_mean, scale=projected_img_std)
        projected_imgs = tf.split(projected_img_density_dist, 2, 0)
        projected_dist1 = projected_imgs[0]
        projected_dist2 = projected_imgs[1]
#         projected_dist1 = K.gather(projected_imgs[0], id_pairs[0,:]) 
#         projected_dist2 = K.gather(projected_imgs[1], id_pairs[1,:])
        
        # local proximities
        imputed_pro = K.softmax(1.0 / (1 + tf.math.exp(-tf.math.multiply(imputed_dist1, imputed_dist2))))
        projected_pro = K.softmax(1.0 / (1 + tf.math.exp(-tf.math.multiply(projected_dist1, projected_dist2))))
#         imputed_pro = tf.squeeze(imputed_pro, -1)
#         projected_pro = tf.squeeze(projected_pro, -1)
        
        print('111:', imputed_pro.shape, projected_pro.shape)
        
        kl_loss = kl(imputed_pro, projected_pro)

        return kl_loss

    def consistencyLoss(self, con_loss):
        
        def loss(y_true, y_pred):
            
            return con_loss
        
        return loss

    def compile_pconv_unet(self, model, inputs_mask, con_loss, lr=0.0002):
        
        # reconstruction loss
        reconstruct_loss = self.ae_loss_total(inputs_mask)
        # contrastive loss
#         contrastive_loss = loss.NTXentLoss(temperature=0.5,  
#                                            batch_size = self.batch_size)
        # consistency loss
#         consistency_loss = self.consistencyLoss(con_loss)
        
#         weights = [self.w1, self.w2, self.w3]
#         weights = [1., 0.001, 100000.]
#        print('weight', weights)
        
        model.compile(
                optimizer = Adam(lr=lr),
                loss=reconstruct_loss,
#                loss_weights=weights,
                metrics=['mse']
#                 metrics=[self.PSNR]
                )
        
        
    def reduce_encoder_output(self, encoder_output, encoder_reduction):
        if encoder_reduction == EncoderReduction.GA_POOLING:
            reduced = GlobalAveragePooling2D()(encoder_output)
        elif encoder_reduction == EncoderReduction.FLATTEN:
            reduced = Flatten()(encoder_output)
        else:
            raise ValueError()
        return reduced
    
    def add_contrastive_output(self, input, projection_head_layers, projection_dim):
        mid_dim = int(input.shape[-1])
        
        ph_layers = []
        for _ in range(projection_head_layers - 1):
            ph_layers.append(mid_dim)
        if projection_head_layers > 0:
            ph_layers.append(projection_dim)
        contrast_head = model_util.projection_head(input, ph_layers=ph_layers)
        print(contrast_head.shape)
        con_output = Flatten(name='con_output')(contrast_head)
#         con_output = Layer(name='con_output')(contrast_head)
        return con_output
    
    def ae_loss_total(self, mask):
        """
        Creates a loss function which sums all the loss components 
        and multiplies by their weights. See paper eq. 7.
        """
        def loss(y_true, y_pred):
            
            # Compute predicted image with non-hole pixels set to ground truth
            y_comp = Multiply()([mask,y_true]) + Multiply()([1-mask, y_pred])
#             y_comp = mask * y_true + (1-mask) * y_pred
            # Compute the vgg features. 
            if self.vgg_device:
                with tf.device(self.vgg_device):
                    vgg_out = self.vgg(y_pred)
                    vgg_gt = self.vgg(y_true)
                    vgg_comp = self.vgg(y_comp)
            else:
                vgg_out = self.vgg(y_pred)
                vgg_gt = self.vgg(y_true)
                
#                 print(y_true.shape, y_pred.shape, y_comp.shape)
                
                vgg_comp = self.vgg(y_comp)
            
#             print('y_comp:', y_comp.shape)
            # Compute loss components
            l1 = self.loss_valid(mask, y_true, y_pred)
            l2 = self.loss_hole(mask, y_true, y_pred)
            l3 = self.loss_perceptual(vgg_out, vgg_gt, vgg_comp)
            l4 = self.loss_style(vgg_out, vgg_gt)
            l5 = self.loss_style(vgg_comp, vgg_gt)
            l6 = self.loss_tv(mask, y_comp)

            # Return loss function
            return l1 + 6*l2 + 0.05*l3 + 1*(l4+l5) + 0.1*l6

        return loss

    def loss_hole(self, mask, y_true, y_pred):
        """Pixel L1 loss within the hole / mask"""
        return self.l1((1-mask) * y_true, (1-mask) * y_pred)
    
    def loss_valid(self, mask, y_true, y_pred):
        """Pixel L1 loss outside the hole / mask"""
        return self.l1(mask * y_true, mask * y_pred)
    
    def loss_perceptual(self, vgg_out, vgg_gt, vgg_comp): 
        """Perceptual loss based on VGG16, see. eq. 3 in paper"""       
        loss = 0
        for o, c, g in zip(vgg_out, vgg_comp, vgg_gt):
            loss += self.l1(o, g) + self.l1(c, g)
        return loss
        
    def loss_style(self, output, vgg_gt):
        """Style loss based on output/computation, used for both eq. 4 & 5 in paper"""
        loss = 0
        for o, g in zip(output, vgg_gt):
            loss += self.l1(self.gram_matrix(o), self.gram_matrix(g))
        return loss
    
    def loss_tv(self, mask, y_comp):
        """Total variation loss, used for smoothing the hole region, see. eq. 6"""

        # Create dilated hole region using a 3x3 kernel of all 1s.
        kernel = K.ones(shape=(3, 3, mask.shape[3], mask.shape[3]))
        dilated_mask = K.conv2d(1-mask, kernel, data_format='channels_last', padding='same')

        # Cast values to be [0., 1.], and compute dilated hole region of y_comp
        dilated_mask = K.cast(K.greater(dilated_mask, 0), 'float32')
        P = dilated_mask * y_comp

        # Calculate total variation loss
        a = self.l1(P[:,1:,:,:], P[:,:-1,:,:])
        b = self.l1(P[:,:,1:,:], P[:,:,:-1,:])        
        return a+b

    def fit_generator(self, generator, *args, **kwargs):
        """Fit the U-Net to a (images, targets) generator

        Args:
            generator (generator): generator supplying input image & mask, as well as targets.
            *args: arguments to be passed to fit_generator
            **kwargs: keyword arguments to be passed to fit_generator
        """
        self.model.fit_generator(
            generator,
            *args, **kwargs
        )
        
    def fit(self, generator, *args, **kwargs):
        """Fit the U-Net to a (images, targets) generator

        Args:
            generator (generator): generator supplying input image & mask, as well as targets.
            *args: arguments to be passed to fit_generator
            **kwargs: keyword arguments to be passed to fit_generator
        """
        self.model.fit(
            generator,
            *args, **kwargs
        )
        
    def summary(self):
        """Get summary of the UNet model"""
        print(self.model.summary())

    def load(self, filepath, train_bn=True, lr=0.0002):

        # Create UNet-like model
        self.model, inputs_mask, con_loss = self.build_pconv_unet(train_bn)
#         self.compile_pconv_unet(self.model, inputs_mask, lr) 
        self.compile_pconv_unet(self.model, inputs_mask, con_loss, lr)  

        # Load weights into model
        epoch = int(os.path.basename(filepath).split('.')[1].split('-')[0])
        assert epoch > 0, "Could not parse weight file. Should include the epoch"
        self.current_epoch = epoch
        self.model.load_weights(filepath)
        
        print("pretrained model loaded")

    @staticmethod
    def PSNR(y_true, y_pred):
        """
        PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        The equation is:
        PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
        
        Our input is scaled with be within the range -2.11 to 2.64 (imagenet value scaling). We use the difference between these
        two values (4.75) as MAX_I        
        """        
        #return 20 * K.log(4.75) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0) 
        return - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0) 

    @staticmethod
    def current_timestamp():
        return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    @staticmethod
    def l1(y_true, y_pred):
        """Calculate the L1 loss used in all loss calculations"""
        if K.ndim(y_true) == 4:
            return K.mean(K.abs(y_pred - y_true), axis=[1,2,3])
        elif K.ndim(y_true) == 3:
            return K.mean(K.abs(y_pred - y_true), axis=[1,2])
        else:
            raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")
    
    @staticmethod
    def gram_matrix(x, norm_by_channels=False):
        """Calculate gram matrix used in style loss"""
        
        # Assertions on input
        assert K.ndim(x) == 4, 'Input tensor should be a 4d (B, H, W, C) tensor'
        assert K.image_data_format() == 'channels_last', "Please use channels-last format"        
        
        # Permute channels and get resulting shape
        x = K.permute_dimensions(x, (0, 3, 1, 2))
        shape = K.shape(x)
        B, C, H, W = shape[0], shape[1], shape[2], shape[3]
        
        # Reshape x and do batch dot product
        features = K.reshape(x, K.stack([B, C, H*W]))
        gram = K.batch_dot(features, features, axes=2)
        
        # Normalize with channels, height and width
        gram = gram /  K.cast(C * H * W, x.dtype)
        
        return gram
    
    # Prediction functions
    ######################
    def predict(self, sample, **kwargs):
        """Run prediction using this model"""
        return self.model.predict(sample, **kwargs)
