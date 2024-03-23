from models.PConvAE import PConvUnet, EncoderReduction
from models.PConvAE_delContrast import PConvUnet_DelContrast
from models.PConvAE_delConsist import PConvUnet_DelConsist
from models.PConvAE_contrast import PConvUnet_Contrast
from models.PConvAE_reconstruct import PConvUnet_Reconstruct
from models import loss
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def add_model_args(parser):
    # Unet options
#     parser.add_argument('-f', '--filters', type=int, default=64, help="How many base filters should be used")
    parser.add_argument('--encoder-reduction', default=EncoderReduction.GA_POOLING)
    
    return parser


def construct_model_from_args(args):
    
    print('Running model:', args.runmodel)
    
    rnfl2vec_model = PConvUnet(img_rows = args.img_rows,
                      img_cols = args.img_cols,
                      reconstruct_w=args.reconstruct_w, 
                      contrastive_w=args.contrastive_w, 
                      consistency_w=args.consistency_w,
                      batch_size = args.batch_size,
                      vgg_weights = args.vgg_weights,
                      #encoder_reduction = args.encoder_reduction,
                      projection_dim = args.projection_dim,
                      embed_dim = args.embed_dim,
                      projection_head_layers = args.projection_layers)
    
    return rnfl2vec_model


