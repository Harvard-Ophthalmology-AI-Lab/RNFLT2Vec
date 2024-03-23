import tensorflow as tf
import argparse
from models import rnflt2vec
from utils import data_process
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback
# import tensorflow_addons as tfa
# from keras_tqdm import TQDMNotebookCallback
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_addons as tfa
import numpy as np
# tqdm_callback = tfa.callbacks.TQDMProgressBar()

import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'


parser = argparse.ArgumentParser(description='Tensorflow RNFLT2Vec Training')
    
rnflt2vec.add_model_args(parser)

parser.add_argument('--model-name', default='RNFLT2vec')
parser.add_argument('--runmodel', default='RNFLT2vec')
parser.add_argument('--img-rows', type=int, default=256)
parser.add_argument('--img-cols', type=int, default=256)
#parser.add_argument('--encoder-reduction', default=EncoderReduction.GA_POOLING)
parser.add_argument('--projection-dim', type=int, default=128,
                    help='Dimension at the output of the projection head')
parser.add_argument('--embed-dim', type=int, default=512)
parser.add_argument('--projection-layers', type=int, default=3, help='Number of projection layers')
parser.add_argument('--temperature', type=float, default=0.5)

parser.add_argument('--data-path', help='path to the original dataset')
parser.add_argument('--pretrained', help='path to the pretrained model')
parser.add_argument('--gpu', default='1', help="The id of the gpu device used")
parser.add_argument('--batch-size', default=4, type=int, help='mini-batch size (default: 4)')
parser.add_argument('-vgg-weights', default='imagenet', help='pretraind vgg model weights')
parser.add_argument('--lr', default=0.0002, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--reconstruct_w', default=1.0, type=float, help='reconstruct_w')
parser.add_argument('--contrastive_w', default=0.001, type=float, help='contrastive_w')
parser.add_argument('--consistency_w', default=0.04, type=float, help='consistency_w')
parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
parser.add_argument('--steps-per-epoch', type=int, default=5000, help='number of running steps per epoch')
parser.add_argument('-f')
args = parser.parse_args()

#     args.vgg_weights = 'imagenet'
args.data_path = '/shared/hdds_20T/ms1233/pyspace/RNFLT2Vec/dataset/'
args.vgg_weights = '/shared/hdds_20T/ms1233/pyspace/RNFLT2Vec/dataset/pytorch_to_keras_vgg16.h5'
args.local_data_path = '/shared/hdds_20T/ms1233/pyspace/RNFLT2Vec_v2/dataset/'
args.pretrained = False


if args.pretrained:
    model = rnflt2vec.construct_model_from_args(args)
    model.load(
        r"/shared/hdds_20T/ms1233/pyspace/RNFLT2Vec/checkpoint/combined_rnflt2vec_weights_10_0001_0001.30-0.10.h5",
        train_bn=False,
        lr=0.00005
    )
#     model = tf.keras.models.load_model(args.pretrained, compile=False)
else:
    model = rnflt2vec.construct_model_from_args(args)
    
    

train_generator, val_generator, train_size, val_size = data_process.load_dataset(args.data_path,
                                                                                   args.local_data_path, 
                                                                                   batch_size=args.batch_size)


(masked, mask, masked_cc, mask_cc, id_pairs), (ori1, ori_cc, ori_cc) = next(val_generator)
# (masked, mask, masked_cc, mask_cc, id_pairs), ori1 = next(test_generator)
def plot_callback(model):
    """Called at the end of each epoch, displaying our previous test images,
    as well as their masked predictions and saving them to disk"""

    # Get samples & Display them, outputs, projection_out, imputed_out       
    pred_img, _, impute_img = model.predict([masked, mask, masked_cc, mask_cc, id_pairs])
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    for i in range(len(ori1)):
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].imshow(masked_cc[i,:,:,:])
        axes[1].imshow(impute_img[i,:,:,:] * 1.)
        axes[2].imshow(ori_cc[i,:,:,:])
        axes[0].set_title('Masked Image')
        axes[1].set_title('Predicted Image')
        axes[2].set_title('Original Image')

        plt.savefig(r'/shared/hdds_20T/ms1233/pyspace/RNFLT2Vec_v2/imgs/{}_img_{}_{}.png'.format(args.model_name, i, 
                                                                                                      pred_time))
        plt.close()

        
FOLDER = '/shared/hdds_20T/ms1233/pyspace/RNFLT2Vec_v2/checkpoint/'

train_weight_str = str(args.reconstruct_w).replace('.','')+'_'+str(args.contrastive_w).replace('.','')+'_'+str(args.consistency_w).replace('.','')

start_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

# Run training for certain amount of epochs
history = model.fit(
    train_generator, 
    steps_per_epoch=int(train_size/args.batch_size),
    validation_data=val_generator,
    validation_steps=int(val_size/args.batch_size),
    epochs=100,  
    verbose=2,
    callbacks=[
        TensorBoard(
            log_dir=FOLDER, 
            write_graph=False
        ),
        ModelCheckpoint(
            FOLDER+'combined_rnflt2vec_weights_512_128_'+train_weight_str+'.{epoch:02d}-{loss:.2f}.h5',
            monitor='val_loss', 
            save_best_only=True,
            save_weights_only=True
        ),
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: plot_callback(model)
        )
# #             tqdm_callback
#         tqdm_callback
    ]
)

end_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

print(start_time, end_time)

np.save(FOLDER+'history_512_128_'+train_weight_str, history.history)

print('Training finished!')