from os import path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from enmspring.vmddraw_bb_st_hb import DrawAgent
from enmspring.graphs_bigtraj import StackMeanModeAgent, BackboneMeanModeAgent, HBMeanModeAgent


class BackboneAgent:
    interval_time = 500
    meanmode_obj = BackboneMeanModeAgent

    def __init__(self, host, big_traj_folder):
        self.host = host
        self.big_traj_folder = big_traj_folder

        self.mean_mode_agent = None

    def get_sum_array(self, k_criteria):
        sum_array = np.zeros(self.mean_mode_agent.n_window)
        for window_id in range(self.mean_mode_agent.n_window):
            time_key = self.mean_mode_agent.time_list[window_id]
            small_agent = self.mean_mode_agent.d_smallagents[time_key]
            sum_array[window_id] = self.get_sum_from_laplacian(small_agent.laplacian_mat, k_criteria)
        return sum_array

    def ini_mean_mode_agent(self):
        if self.mean_mode_agent is None:
            self.mean_mode_agent = self.meanmode_obj(self.host, self.big_traj_folder, self.interval_time)
            self.mean_mode_agent.preprocess_all_small_agents()
            self.mean_mode_agent.initialize_all_maps()
            self.mean_mode_agent.set_d_idx_and_inverse()

    def get_sum_from_laplacian(self, laplacian_mat, k_criteria):
        tri_upper = np.triu(laplacian_mat, 1)
        idx_i_array, idx_j_array = np.where(tri_upper > k_criteria)
        fraying_lst = [self.determine_fraying(idx_i, idx_j) for idx_i, idx_j in zip(idx_i_array, idx_j_array)]
        value = 0.
        for idx_i, idx_j, fraying in zip(idx_i_array, idx_j_array, fraying_lst):
            if fraying:
                continue
            value += tri_upper[idx_i, idx_j]
        return value

    def determine_fraying(self, idx_i, idx_j):
        resid_i = self.get_resid_by_idx(idx_i)
        resid_j = self.get_resid_by_idx(idx_j)
        fraying_resid_lst = [1, 2, 3, 19, 20, 21]
        if (resid_i in fraying_resid_lst) or (resid_j in fraying_resid_lst):
            return True
        else:
            return False

    def get_resid_by_idx(self, idx):
        cgname = self.mean_mode_agent.d_idx_inverse[idx]
        return self.mean_mode_agent.resid_map[cgname]

class StackAgent(BackboneAgent):
    meanmode_obj = StackMeanModeAgent

class HBAgent(BackboneAgent):
    meanmode_obj = HBMeanModeAgent

    def determine_fraying(self, idx_i, idx_j):
        resid_i = self.get_resid_by_idx(idx_i)
        resid_j = self.get_resid_by_idx(idx_j)
        fraying_resid_lst = [1, 2, 3, 19, 20, 21]
        if (resid_i in fraying_resid_lst) and (resid_j in fraying_resid_lst):
            return True
        else:
            return False 

class ThreeBar(DrawAgent):
    interaction_lst = ['HB', 'ST', 'BB']
    host_lst = ['a_tract_21mer', 'g_tract_21mer', 'atat_21mer', 'gcgc_21mer']

    d_x_host = {'a_tract_21mer': 1, 'g_tract_21mer': 2, 'atat_21mer': 4, 'gcgc_21mer': 5}
    d_color_host = {'a_tract_21mer': '#5C8ECB', 'g_tract_21mer': '#EA6051', 'atat_21mer': '#8CF8D5', 'gcgc_21mer': '#E75F93'}
    width = 0.4
    tickfz = 4
    lbfz = 5
    d_k_criteria = {'HB': 0.5, 'ST': 1., 'BB': 1.}
    d_y_ticks = {'HB': np.arange(0, 21, 10), 'ST': np.arange(0, 31, 10), 'BB': np.arange(0, 81, 20)}
    d_y_labels = {'HB': 'HB', 'ST': 'Stack', 'BB': 'Backbone'}

    def __init__(self, big_traj_folder, data_folder):
        self.big_traj_folder = big_traj_folder
        self.data_folder = data_folder

        self.d_df = dict()

        self.d_b_agent = None
        self.d_s_agent = None
        self.d_h_agent = None

    def make_df_for_all_host(self):
        for host in self.host_lst:
            self.make_df(host)

    def read_df_for_all_host(self):
        for host in self.host_lst:
            self.read_df(host)

    def plot_main(self, figsize, hspace):
        fig = plt.figure(figsize=figsize, facecolor='white')
        d_axes = self.get_d_axes(fig, hspace)
        for interaction in self.interaction_lst:
            self.bar_plot_interaction(d_axes, interaction)
        self.remove_xticks(d_axes)
        self.set_xticks(d_axes)
        self.set_yticks(d_axes)
        self.set_ylabels(d_axes)
        return fig, d_axes

    def bar_plot_interaction(self, d_axes, interaction):
        ax = d_axes[interaction]
        for host in self.host_lst:
            mean, std = self.get_k_mean_std(host, interaction)
            color = self.d_color_host[host]
            ax.bar(self.d_x_host[host], mean, color=color, yerr=std, ecolor='black')

    def get_k_mean_std(self, host, interaction):
        k_array = self.d_df[host][interaction] / 15
        return k_array.mean(), k_array.std()

    def get_d_axes(self, fig, hspace):
        d_axes = dict()
        grid = gridspec.GridSpec(3, 1, hspace=hspace)
        for idx, interaction in enumerate(self.interaction_lst):
            d_axes[interaction] = fig.add_subplot(grid[idx])
        return d_axes

    def remove_xticks(self, d_axes):
        d_axes['HB'].tick_params(axis='x', bottom=False, top=False, labelbottom=False)
        d_axes['ST'].tick_params(axis='x', bottom=False, top=False, labelbottom=False)

    def set_xticks(self, d_axes):
        xticklabels = ['A-tract', 'G-tract', 'TATA', 'CpG']
        d_axes['BB'].set_xticks([1,2,4,5])
        d_axes['BB'].set_xticklabels(xticklabels)
        d_axes['BB'].tick_params(axis='x', labelsize=self.tickfz, length=1.5, pad=1)

    def set_ylabels(self, d_axes):
        for interaction in self.interaction_lst:
            d_axes[interaction].set_ylabel(self.d_y_labels[interaction], fontsize=self.lbfz, labelpad=1)

    def set_yticks(self, d_axes):
        for interaction in self.interaction_lst:
            d_axes[interaction].set_yticks(self.d_y_ticks[interaction])
            d_axes[interaction].tick_params(axis='y', labelsize=self.tickfz, length=1.5, pad=1)
            #for hline in self.d_y_ticks[interaction]:
            #   d_axes[interaction].axhline(hline, color='grey', alpha=0.2, linewidth=0.5)

    def make_df(self, host):
        d_result = dict()
        d_agents = {'HB': self.d_h_agent[host], 'ST': self.d_s_agent[host], 'BB': self.d_b_agent[host]}
        for interaction in self.interaction_lst:
            agent = d_agents[interaction]
            k_criteria = self.d_k_criteria[interaction]
            d_result[interaction] = agent.get_sum_array(k_criteria)
        df = pd.DataFrame(d_result)
        f_out = self.get_f_df(host)
        df.to_csv(f_out, index=False)
        self.d_df[host] = df

    def read_df(self, host):
        f_in = self.get_f_df(host)
        self.d_df[host] = pd.read_csv(f_in)

    def get_f_df(self, host):
        return path.join(self.data_folder, f'{host}_bb_st_hb.csv')

    def ini_b_agent(self):
        self.d_b_agent = {host: None for host in self.host_lst}
        for host in self.host_lst:
            if self.d_b_agent[host] is None:
                self.d_b_agent[host] = BackboneAgent(host, self.big_traj_folder)
                self.d_b_agent[host].ini_mean_mode_agent()

    def ini_s_agent(self):
        self.d_s_agent = {host: None for host in self.host_lst}
        for host in self.host_lst:
            if self.d_s_agent[host] is None:
                self.d_s_agent[host] = StackAgent(host, self.big_traj_folder)
                self.d_s_agent[host].ini_mean_mode_agent()

    def ini_h_agent(self):
        self.d_h_agent = {host: None for host in self.host_lst}
        for host in self.host_lst:
            if self.d_h_agent[host] is None:
                self.d_h_agent[host] = HBAgent(host, self.big_traj_folder)
                self.d_h_agent[host].ini_mean_mode_agent()

