
from dataclasses import dataclass
import torch
import numpy as np
import math

from ATSProblemDef import get_random_problems


@dataclass
class Reset_State:
    problems: torch.Tensor
    log_scale: float = None
    # shape: (batch, node, node)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class ATSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.node_cnt = env_params['node_cnt']
        self.pomo_size = env_params['pomo_size']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # STEP-State
        ####################################
        self.step_state = None

    def load_problems(self, batch_size):
        self.batch_size = batch_size
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        problem_gen_params = self.env_params['problem_gen_params']
        self.problems = get_random_problems(batch_size, self.node_cnt, problem_gen_params)
        # shape: (batch, node, node)

    def load_problems_manual(self, problems):
        # problems.shape: (batch, node, node)

        self.batch_size = problems.size(0)
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.problems = problems
        # shape: (batch, node, node)
    
    def load_problems_from_npz(self, npz_path):
        data = np.load(npz_path)["data"] 
        self.load_problems_manual(torch.tensor(data, dtype=torch.float32))

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.empty((self.batch_size, self.pomo_size, 0), dtype=torch.long)

        self.log_scale = math.log2(self.pomo_size)
        # shape: (batch, pomo, 0~)

        self._create_step_state()

        reward = None
        done = False
        return Reset_State(problems=self.problems, log_scale=self.log_scale), reward, done

    def _create_step_state(self):
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.node_cnt))
        # shape: (batch, pomo, node)

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, node_idx):
        # node_idx.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = node_idx
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # print(self.selected_node_list[0][0])
        # shape: (batch, pomo, 0~node)

        self._update_step_state()
        
        # returning values
        done = (self.selected_count == self.node_cnt)
        if done:
            reward = -self._get_total_distance()  # Note the MINUS Sign ==> We MAXIMIZE reward
            # shape: (batch, pomo)
        else:    
            reward = None
        return self.step_state, reward, done

    def _update_step_state(self):
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

    def _get_total_distance(self):

        node_from = self.selected_node_list
        # shape: (batch, pomo, node)
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)
        # shape: (batch, pomo, node)
        batch_index = self.BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.node_cnt)
        # shape: (batch, pomo, node)

        selected_cost = self.problems[batch_index, node_from, node_to]
        # shape: (batch, pomo, node)
        total_distance = selected_cost.sum(2)
        # shape: (batch, pomo)

        return total_distance
    
    def get_local_feature(self):
        if self.current_node is None:
            return None
        current_node = self.current_node[:, :, None, None].expand(self.batch_size, self.pomo_size, 1, self.pomo_size)
        cur_dist = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.pomo_size, self.pomo_size).gather(2, current_node).squeeze(2)
        return cur_dist
