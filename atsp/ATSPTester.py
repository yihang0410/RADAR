import torch
import numpy as np
from logging import getLogger

from ATSPEnv import ATSPEnv as Env
from ATSPModel import ATSPModel as Model
from utils.utils import get_result_folder, AverageMeter, TimeEstimator

class ATSPTester:
    def __init__(self, env_params, model_params, tester_params):
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        if tester_params['use_cuda']:
            torch.cuda.set_device(tester_params['cuda_device_num'])
            self.device = torch.device('cuda', tester_params['cuda_device_num'])
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        checkpoint = torch.load(
            f"{tester_params['model_load']['path']}/checkpoint-{tester_params['model_load']['epoch']}.pt",
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.time_estimator = TimeEstimator()

        # Load npz data
        npz_path = tester_params['npz_file']
        data = np.load(npz_path)['data']  # shape: (batch, node_cnt, node_cnt)
        data = data / (1000 * 1000)
        self.all_problems = torch.tensor(data, dtype=torch.float32)

    def run(self):
        self.time_estimator.reset()
        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()
        score_std_AM = AverageMeter()
        aug_score_std_AM = AverageMeter()

        episode = 0
        total = self.all_problems.size(0)
        batch_size = self.tester_params['test_batch_size']

        while episode < total:
            bs = min(batch_size, total - episode)
            score, aug_score, score_std, aug_score_std = self._test_one_batch(episode, episode + bs)

            score_AM.update(score, bs)
            aug_score_AM.update(aug_score, bs)
            score_std_AM.update(score_std, bs)
            aug_score_std_AM.update(aug_score_std, bs)
            episode += bs

            elapsed, remain = self.time_estimator.get_est_string(episode, total)
            self.logger.info(
                f"Episode {episode}/{total}, Elapsed[{elapsed}], Remain[{remain}], "
                f"Score: {score:.3f}±{score_std:.3f}, Aug: {aug_score:.3f}±{aug_score_std:.3f}"
            )

        self.logger.info("*** Test Done ***")
        self.logger.info(f"NO-AUG SCORE: {score_AM.avg:.4f} ± {score_std_AM.avg:.4f}")
        self.logger.info(f"AUGMENTED SCORE: {aug_score_AM.avg:.4f} ± {aug_score_std_AM.avg:.4f}")

    def _test_one_batch(self, idx_start, idx_end):
        problems = self.all_problems[idx_start:idx_end]  # shape: (batch, node, node)
        aug_factor = self.tester_params['aug_factor'] if self.tester_params['augmentation_enable'] else 1
        batch_size = 3

        all_rewards = []

        self.model.eval()
        with torch.no_grad():
            base = problems.size(0)

            num_batches = (aug_factor + batch_size - 1) // batch_size
            for i in range(num_batches):
                current_batch = min(batch_size, aug_factor - i * batch_size)

                problems_aug = problems.repeat(current_batch, 1, 1)  
                self.env.load_problems_manual(problems_aug)
                reset_state, _, _ = self.env.reset()
                self.model.pre_forward(reset_state)

                state, reward, done = self.env.pre_step()
                while not done:
                    selected, prob = self.model(state)
                    state, reward, done = self.env.step(selected)

                reward = reward.view(current_batch, base, self.env.pomo_size)
                max_pomo = reward.max(dim=2)[0] 
                all_rewards.append(max_pomo)  

            all_rewards = torch.cat(all_rewards, dim=0)  

            no_aug = -all_rewards[0].float().mean()
            no_aug_std = all_rewards[0].float().std()

            aug = -all_rewards.max(dim=0)[0].float().mean()
            aug_std = all_rewards.max(dim=0)[0].float().std()

            return no_aug.item(), aug.item(), no_aug_std.item(), aug_std.item()