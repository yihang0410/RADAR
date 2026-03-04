
##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys
 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..") 
sys.path.insert(0, "../..") 
print(sys.path)

##########################################################################################
# import

import logging

from utils.utils import create_logger, copy_all_src
from ACVRPTester import ACVRPTeseter as Tester

problem_cnt = 100
test_batch_size = {
    100: 1000,
    200: 400,
    500: 50,
    1000: 3   
}

head_num = 8
embedding_dim = 256
qkv_dim = embedding_dim // head_num
##########################################################################################
# parameters

env_params = {
    'node_cnt':problem_cnt,
    'pomo_size': problem_cnt  # same as node_cnt
}

model_params = {
    'embedding_dim': embedding_dim,
    'sqrt_embedding_dim': embedding_dim**(1/2),
    'encoder_layer_num': 5,
    'qkv_dim': qkv_dim,
    'sqrt_qkv_dim': qkv_dim**(1/2),
    'head_num': head_num,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1/2)**(1/2),
    'ms_layer2_init': (1/16)**(1/2),
    'eval_type': 'softma',
    'one_hot_seed_cnt': problem_cnt,  # must be >= node_cnt
}


tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 1,
    'train_episodes': 100*1000,
    'test_batch_size': test_batch_size[problem_cnt],
    'npz_path': f'/common/home/users/h/hang.yi.2024/Code/MatNet-Distance/ACVRP/ACVRP/acvrp_{problem_cnt}_instances.npz',
    'model_load': {
        'path': '/common/home/users/h/hang.yi.2024/Code/final_radar/code/acvrp/result/radar_official_checkpoint',
        'epoch': 2100, 
    }
}

logger_params = {
    'log_file': {
        'desc': '',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Tester(env_params=env_params,
                      model_params=model_params,
                      tester_params=tester_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():

    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2
    trainer_params['validate_episodes'] = 4
    trainer_params['validate_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
