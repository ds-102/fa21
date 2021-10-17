import numpy as np
from copy import deepcopy


class Platform():

    def __init__(self,means_mat,sd_mat, UCB_preference_func, GS_matching_func):

        self.num_arms, self.num_players = means_mat.shape
        assert means_mat.shape == sd_mat.shape
        self.means_mat = means_mat
        self.sd_mat = sd_mat
        self.UCB_preference = UCB_preference_func
        self.GS_matching = GS_matching_func
        self.arm_preference_dict = {k: get_preference_list(self.means_mat[k,:]) for k in range(self.num_arms)}
        self.player_preference_dict = {k: get_preference_list(self.means_mat[:,k]) for k in range(self.num_players)}
        self.true_matching = self.GS_matching(self.arm_preference_dict, self.player_preference_dict)
        self.t = 0
        empty_list_of_lists = [[] for i in range(self.num_arms)]
        self.rewards = {k:deepcopy(empty_list_of_lists) for k in range(self.num_players)}
        self.times_pulled = {k:list(np.zeros(self.num_arms)) for k in range(self.num_players)}

        self.regret = {k:[0] for k in range(self.num_players)}
        self.stable_count = [0]
        self.current_matching={k:k for k in range(self.num_players)}
        return

    def make_current_matching(self, curr_player_preference_dict):
        self.current_matching = self.GS_matching(self.arm_preference_dict, curr_player_preference_dict)
        return

    def is_stable(self):
        return self.current_matching == self.true_matching

    def pull_arms(self):
        current_matching = self.current_matching
        for player, arm in current_matching.items():
            mu=self.means_mat[arm, player]
            sigma=self.sd_mat[arm, player]
            reward=np.random.normal(mu,sigma**2)
            self.rewards[player][arm].append(reward)
            best_arm = self.true_matching[player]
            self.regret[player].append(self.regret[player][-1]+self.means_mat[best_arm, player]-self.means_mat[arm, player])
            self.times_pulled[player][arm]+=1
        self.t+=1
        self.stable_count.append(self.stable_count[-1]+int(self.is_stable()))

        # get new preferences of players over arms
        curr_player_preference_dict = {k : self.UCB_preference(self.t, self.sd_mat[:,k], self.times_pulled[k], self.rewards[k]) for k in range(self.num_players)}
        self.make_current_matching(curr_player_preference_dict)
        return

    def run(self, T):
        for _ in range(1, T+1):
            self.pull_arms()
        return

def get_preference_list(reward_list):
    reward_list = np.array(reward_list)
    preference_list = np.argsort(reward_list)
    preference_list = preference_list[::-1]
    return preference_list