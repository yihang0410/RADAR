DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
print(sys.path)

from utils.utils import create_logger, copy_all_src
from ATSPTester import ATSPTester as Tester



problem_cnt = 100

test_batch_size = {
    100: 1000,
    200: 400,
    500: 50,
    1000: 3   
}

aug_batch_size = {
    100: 20,
    200: 2
}

head_num = 8
embedding_dim = 256
qkv_dim = embedding_dim // head_num
##########################################################################################
# parameters

env_params = {
    'node_cnt':problem_cnt,
    'problem_gen_params': {
        'int_min': 0,
        'int_max': 1000*1000,
        'scaler': 1000*1000
    },
    'pomo_size': problem_cnt  # same as node_cnt
}

model_params = {
    'embedding_dim': embedding_dim,
    'sqrt_embedding_dim': embedding_dim**(1/2),
    'encoder_layer_num': 5,
    'qkv_dim': qkv_dim,
    'sqrt_qkv_dim': qkv_dim**(1/2),
    'head_num': head_num,
    'logit_clipping': 50,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1/2)**(1/2),
    'ms_layer2_init': (1/16)**(1/2),
    'eval_type': 'softmax',
    'one_hot_seed_cnt': problem_cnt,  # must be >= node_cnt
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': 'result/radar_official_checkpoint',
        'epoch': 2100,
    },
    'npz_file': 'dataset/ATSP'+str(problem_cnt)+'.npz',
    'test_batch_size': test_batch_size[problem_cnt],
    'augmentation_enable': False,
    'aug_factor': 128,
    'aug_batch_size': 10,
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'atsp_radar_test',
        'filename': 'log.txt'
    }
}

def main():
    if DEBUG_MODE:
        tester_params['aug_factor'] = 10
        tester_params['test_batch_size'] = 10

    create_logger(**logger_params)
    tester = Tester(env_params, model_params, tester_params)
    copy_all_src(tester.result_folder)
    tester.run()

if __name__ == "__main__":
    main()
