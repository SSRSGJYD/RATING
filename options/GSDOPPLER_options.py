import config

class GSDOPPLEROptions(object):
    @staticmethod
    def modify_commandline_options(parser):
        # basic parameters
        parser.add_argument('--name', type=str, default='GSDOPPLER', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--split', type=int, default=1)
        parser.add_argument('--expid', type=int, default=1)
        parser.add_argument('--checkpoints_dir', type=str, default=config.GSDOPPLER_checkpoints, help='models are saved here')
        parser.add_argument('--load_dir', type=str, default=None, help='model paths to be loaded')
        parser.add_argument('--vis_dir', default='./vis', help='visualized heatmap are saved here')
        
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='GSDOPPLER', help='chooses train dataset')
        parser.add_argument('--dataset', type=object, default=None, help='created dataset')
        parser.add_argument('--dataset_id', type=int, default=0, help='index of train dataset')
        parser.add_argument('--v_dataset_mode', type=str, default='GSDOPPLER', help='chooses valid dataset')
        parser.add_argument('--v_dataset', type=object, default=None, help='created v_dataset')
        parser.add_argument('--v_dataset_id', type=int, default=0, help='index of valid and test dataset')
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--v_batch_size', type=int, default=64, help='valid input batch size')
        parser.add_argument('--drop_last', type=int, default=0)
        parser.add_argument('--serial_batches',type=bool, default=False, help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        
        parser.add_argument('--train_datasets', type=list, default=[["'./dataset_files/train_split{}.json'.format(opt.split)"]])
        parser.add_argument('--valid_datasets', type=list, default=[["'./dataset_files/val_split{}.json'.format(opt.split)"]])
        parser.add_argument('--test_datasets', type=list, default=[])

        parser.add_argument('--classes', type=str, default='01,23-01,23')
        parser.add_argument('--label', type=str, default='GS', choices=('GS', 'DOPPLER', 'combined'))

        parser.add_argument('--GS_policy', type=str, default='roi_random')
        parser.add_argument('--GS_norm_type', type=str, default='imagenet')
        parser.add_argument('--GS_input_h', type=int, default=224)
        parser.add_argument('--GS_input_w', type=int, default=224)

        parser.add_argument('--DOPPLER_policy', type=str, default='roi_random')
        parser.add_argument('--DOPPLER_norm_type', type=str, default='imagenet')
        parser.add_argument('--DOPPLER_input_h', type=int, default=224)
        parser.add_argument('--DOPPLER_input_w', type=int, default=224)

        parser.add_argument('--sample_strategy', type=str, default='resample')
        return parser

    def __init__(self):
        pass