
import torch
from logging import getLogger

from ACVRPEnv import ACVRPEnv as Env
from ACVRPModel import ACVRPModel as Model

from utils.utils import *


class ACVRPTeseter:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        
        self.env = Env(**self.env_params)

        self.model = Model(**self.model_params)

        self.start_epoch = 1
        model_load = self.tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

        npz_path = self.tester_params['npz_path']
        npz_data = np.load(npz_path)
        self.all_dists = torch.tensor(npz_data["dist"], dtype=torch.float32) / 1000000
        demand = npz_data["demand"][:, 1:]
        node_cnt = demand.shape[1]
        demand_scaler = {20: 30, 50: 40, 100: 50, 200: 80, 500: 100, 1000: 250}[node_cnt]
        demand = demand / demand_scaler
        self.all_demands = torch.tensor(demand, dtype=torch.float32)
        self.batch_size = self.tester_params['test_batch_size']
    

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        print("TESTING")
        for epoch in range(self.start_epoch, self.tester_params['epochs']+1):
            self.logger.info('=================================================================')

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.tester_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.tester_params['epochs'], elapsed_time_str, remain_time_str))


    def run(self):
        self.time_estimator.reset()
        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        no_aug_scores = []
        aug_scores = []

        episode = 0
        total = self.all_dists.size(0)
        batch_size = self.batch_size

        while episode < total:
            bs = min(batch_size, total - episode)
            score, aug_score, no_aug_list, aug_list = self._test_one_batch(episode, episode + bs)
            score_AM.update(score, bs)
            aug_score_AM.update(aug_score, bs)

            no_aug_scores.append(no_aug_list.cpu().numpy())
            aug_scores.append(aug_list.cpu().numpy())

            episode += bs
            elapsed, remain = self.time_estimator.get_est_string(episode, total)
            self.logger.info(f"Episode {episode}/{total}, Elapsed[{elapsed}], Remain[{remain}], "
                             f"Score: {score:.4f}, Aug_Score: {aug_score:.4f}")

        all_no_aug = np.concatenate(no_aug_scores)
        all_aug = np.concatenate(aug_scores)

        no_aug_std = np.std(all_no_aug, ddof=1)
        aug_std = np.std(all_aug, ddof=1)

        self.logger.info("*** Test Done ***")
        self.logger.info(f"NO-AUG SCORE: {score_AM.avg:.4f} ± {no_aug_std:.4f}")
        self.logger.info(f"AUGMENTED SCORE: {aug_score_AM.avg:.4f} ± {aug_std:.4f}")

    def _test_one_batch(self, idx_start, idx_end):
        dist = self.all_dists[idx_start:idx_end]
        demand = self.all_demands[idx_start:idx_end]
        batch_size = dist.size(0)

        
        aug_factor = 1

        self.model.eval()
        with torch.no_grad():
            self.env.load_problems_manual(dist, demand)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            prob_list = torch.zeros(size=(dist.size(0), self.env.pomo_size, 0))
            state, reward, done = self.env.pre_step()

            while not done:
                selected, prob = self.model(state)
                state, reward, done = self.env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

            reward = reward.view(aug_factor, batch_size, self.env.pomo_size)
            max_pomo_reward, _ = reward.max(dim=2)

            no_aug_list = -max_pomo_reward[0].float()             # (batch,)
            aug_list = -max_pomo_reward.max(dim=0)[0].float()     # (batch,)

            no_aug_score = no_aug_list.mean()
            aug_score = aug_list.mean()

        return no_aug_score.item(), aug_score.item(), no_aug_list, aug_list

