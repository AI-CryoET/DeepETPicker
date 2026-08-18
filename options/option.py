import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Parameters for 3D particle picking')

        # dataloader parameters
        self.parser.add_argument('--block_size', help='block size', type=int, default=72)
        self.parser.add_argument('--val_block_size', help='block size', type=int, default=0)
        self.parser.add_argument('--random_num', help='random number', type=int, default=0)
        self.parser.add_argument('--num_classes', help='number of classes', type=int, default=13)
        self.parser.add_argument('--use_bg', type=str2bool, help='whether use batch generator', default=False)
        self.parser.add_argument('--test_use_pad', type=str2bool, help='whether use coord conv', default=False)
        self.parser.add_argument('--pad_size', nargs='+', type=int, default=[12])
        self.parser.add_argument('--data_split', nargs='+', type=int, default=[0, 1, 0, 1, 0, 1])
        self.parser.add_argument('--configs', type=str, default='')
        self.parser.add_argument('--pre_configs', type=str, default='')
        self.parser.add_argument('--train_configs', type=str, default='')
        self.parser.add_argument('--val_configs', type=str, default='')
        self.parser.add_argument('--loader_type', type=str, default='dataloader_DynamicLoad', help="whether use DynamicLoad",
                                 # choices=["dataloader", "dataloader_DynamicLoad",
                                 #          'dataloader_DynamicLoad_CellSeg',
                                 #          "dataloader_DynamicLoad_Semi"]
                                 )
        self.parser.add_argument('--sel_train_num', nargs='+', type=int)
        self.parser.add_argument('--input_cat', type=str2bool, help='whether use input cat', default=False)
        self.parser.add_argument('--input_cat_items', nargs='+', type=str, default='None')

        # model parameters
        self.parser.add_argument('--network', help='network type', type=str, default='ResUNet',
                                 # choices=['unet', 'UMC', 'ResUnet', 'DoubleUnet', 'MFNet', 'DMFNet', 'DMFNet_down3',
                                 #          'NestUnet', 'VoxResNet', 'HighRes3DNet', 'HRNetv1']
                                 )
        self.parser.add_argument('--in_channels', help='input channels of the network', type=int, default=1)
        self.parser.add_argument('--f_maps', nargs='+', type=int, help="Feature numbers of ResUnet")
        self.parser.add_argument('--use_IP', type=str2bool, help='whether use image pyramid', default=False)
        self.parser.add_argument('--use_coord', type=str2bool, help='whether use coord conv', default=False)
        self.parser.add_argument('--use_softmax', type=str2bool, help='whether use softmax', default=False)
        self.parser.add_argument('--use_sigmoid', type=str2bool, help='whether use sigmoid', default=False)
        self.parser.add_argument('--use_tanh', type=str2bool, help='whether use tanh', default=False)
        self.parser.add_argument('--use_softpool', type=str2bool, help='whether use softpool', default=False)
        self.parser.add_argument('--use_aspp', type=str2bool, help='whether use aspp', default=False)
        self.parser.add_argument('--use_se_loss', type=str2bool, help='whether use SE loss', default=False)
        self.parser.add_argument('--use_att', type=str2bool, help='whether use aspp', default=False)
        self.parser.add_argument('--norm', help='type of normalization', type=str, default='bn',
                                 choices=['bn', 'gn', 'in', 'sync_bn'])
        self.parser.add_argument('--act', help='type of activation function', type=str, default='relu',
                                 choices=['relu', 'lrelu', 'elu', 'gelu'])
        self.parser.add_argument('--use_paf', type=str2bool, help='PostFusion_orit: whether use part affinity field',
                                 default=False)
        self.parser.add_argument('--paf_sigmoid', type=str2bool, help='whether use sigmoid for the branch of '
                                                                      'part affinity field', default=False)
        self.parser.add_argument('--pif_sigmoid', type=str2bool, help='whether use sigmoid for the branch of '
                                                                      'part intensity field', default=False)
        self.parser.add_argument('--final_double', type=str2bool, help='whether use sigmoid for the branch of '
                                                                       'part affinity field', default=False)
        self.parser.add_argument('--use_uncert', type=str2bool, help='whether use uncert for loss weights',
                                 default=False)
        self.parser.add_argument('--use_lw', type=str2bool, help='whether use lightweight', default=False)
        self.parser.add_argument('--lw_kernel', type=int, default=3)

        # training hyper-parameters
        self.parser.add_argument('--learning_rate', type=float, default=5e-5)
        self.parser.add_argument('--batch_size', help='batch size', type=int, default=32)
        self.parser.add_argument('--val_batch_size', help='batch size', type=int, default=0)
        self.parser.add_argument('--max_epoch', help='number of epochs', type=int, default=100)
        self.parser.add_argument('--loss_func_seg', help='seg loss function type', type=str, default='Dice')
        self.parser.add_argument('--threshold', type=float, default=0.5, help="calculate seg_metrics")
        self.parser.add_argument('--others', help='others', type=str, default='')
        self.parser.add_argument('--dset_name', type=str, help="the name of dataset")
        self.parser.add_argument('--train_mode', type=str, default='train', help='train mode')
        self.parser.add_argument('--gpu_id', nargs='+', type=int, default=[0, 1, 2, 3], help='gpu id')
        self.parser.add_argument('--prf1_alpha', type=float, default=3, help="calculate seg_metrics")

        self.parser.add_argument('--checkpoints', type=str, help='Checkpoint directory',
                                 default=None)
        self.parser.add_argument('--dir_name', type=str, help='Directory name',
                                 default=None)
        self.parser.add_argument('--seg_tau', type=float, default=0.95, help='Segmentation threshold')
        self.parser.add_argument('--use_mask', type=str2bool, help='use mask to cal loss for SSL',
                                 default=False)
        self.parser.add_argument('--coord_path', type=str, help='Coordiate path name',
                                 default=None)

        # test_parameters
        self.parser.add_argument('--test_idxs', nargs='+', type=int, default=[0])
        self.parser.add_argument('--save_pred', type=str2bool, help='whether use segmentation', default=False)
        self.parser.add_argument('--de_duplication', type=str2bool, default=False, help='Whether use dilation')
        self.parser.add_argument('--test_mode', type=str, default='test_val', help='test mode')
        self.parser.add_argument('--save_mrc', type=str2bool, default=False, help='Whether save .mrc file')
        self.parser.add_argument('--use_cluster', type=str2bool, default=False, help='Whether use clustering')
        self.parser.add_argument('--skip_4v94', type=bool, default=False,
                                 help='Whether to skip 4V94 evaluation or not. True in SHREC Cryo-ET 2021 results.')
        self.parser.add_argument('--skip_vesicles', type=bool, default=False,
                                 help='Whether to skip vesicles or not. True in SHREC Cryo-ET 2021 results.')
        self.parser.add_argument('--out_name', type=str, default='TestRes',
                                 help='file name for saving the predicted coordinates')

        self.parser.add_argument('--train_set_ids', type=str, default="0")
        self.parser.add_argument('--val_set_ids', type=str, default="0")
        self.parser.add_argument('--cfg_save_path', type=str, default=".")
        # optim parameters
        self.parser.add_argument('--optim', type=str, default='AdamW')
        self.parser.add_argument('--scheduler', type=str, default='OneCycleLR')
        self.parser.add_argument('--weight_decay', type=float, default=0.01, help="torch.optim: weight decay")
        self.parser.add_argument('--use_seg', type=str2bool, default=False, help='Whether use dilation')
        self.parser.add_argument('--use_eval', type=str2bool, default=False, help='Whether use dilation')

        # loss parameters
        self.parser.add_argument('--gamma', type=float, default=0.75, help="Focal Tversky Loss: focal gamma")

        # eval parameters
        self.parser.add_argument('--de_dup_fmt', type=str, default='fmt4', help='de-duplication format')
        self.parser.add_argument('--mini_dist', type=int, default=10, help='Minimum volume')
        self.parser.add_argument('--meanPool_NMS', type=str2bool, default=False, help='mean_pool NMS')
        self.parser.add_argument('--meanPool_kernel', type=int, default=5, help='mean_pool NMS')

    def gather_options(self):
        args = self.parser.parse_args()
        return args
