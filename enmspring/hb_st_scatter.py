from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enmspring.correlation_window import HBAgent, BackboneAgent, StackAgent
from enmspring.na_seq import sequences

class HBAgentv0(HBAgent):
    n_window = 19

    def get_hb_array(self):
        data_array = np.zeros(self.n_window)
        for idx in range(self.n_window):
            data_array[idx] = self.df.iloc[idx].sum()
        return data_array

class BackboneAgentv0(BackboneAgent):

    def __init__(self, host, bigtraj_folder, corr_folder):
        self.host = host
        self.bigtraj_folder = bigtraj_folder
        self.bigtraj_intv_folder = path.join(self.bigtraj_folder, f'{self.interval_time}ns')
        self.corr_folder = corr_folder

        self.corr_input_folder = path.join(self.corr_folder, 'input')
        self.f_df = self.get_f_df()
        self.df = None

        self.mean_mode_agent = None
        self.time_list = None
        self.n_window = None
        self.d_small_agents = None # Laplacian: self.d_small_agents[(time1,time2)].laplacian_mat
        self.first_small_agent = None

        self.d_seq = {'STRAND1': sequences[host]['guide'], 'STRAND2': sequences[host]['target']}

    def get_backbone_array(self):
        return np.array(self.df['k'])

    def get_f_df(self):
        return path.join(self.corr_input_folder, f'{self.host}_backbone_whole.csv')

    def make_df(self):
        self.set_laplacian_from_small_agents()
        d_result = {'k': np.zeros(self.n_window)}
        for window_id in range(self.n_window):
            key = self.time_list[window_id]
            d_result['k'][window_id] = self.d_small_agents[key].adjacency_mat.sum() / 2
        self.df = pd.DataFrame(d_result)
        self.df.to_csv(self.f_df, index=False)
        print(f'write df into {self.f_df}')

    def read_df(self):
        self.df = pd.read_csv(self.f_df)
        print(f'read df from {self.f_df}')

class StackAgentv0(StackAgent):

    def __init__(self, host, bigtraj_folder, corr_folder):
        self.host = host
        self.bigtraj_folder = bigtraj_folder
        self.bigtraj_intv_folder = path.join(self.bigtraj_folder, f'{self.interval_time}ns')
        self.corr_folder = corr_folder

        self.corr_input_folder = path.join(self.corr_folder, 'input')
        self.f_df = self.get_f_df()
        self.df = None

        self.mean_mode_agent = None
        self.time_list = None
        self.n_window = None
        self.d_small_agents = None # Laplacian: self.d_small_agents[(time1,time2)].laplacian_mat
        self.first_small_agent = None

        self.d_seq = {'STRAND1': sequences[host]['guide'], 'STRAND2': sequences[host]['target']}

    def get_stack_array(self):
        return np.array(self.df['k'])

    def get_f_df(self):
        return path.join(self.corr_input_folder, f'{self.host}_basestack_whole.csv')

    def make_df(self):
        self.set_laplacian_from_small_agents()
        d_result = {'k': np.zeros(self.n_window)}
        for window_id in range(self.n_window):
            key = self.time_list[window_id]
            d_result['k'][window_id] = self.d_small_agents[key].adjacency_mat.sum() / 2
        self.df = pd.DataFrame(d_result)
        self.df.to_csv(self.f_df, index=False)
        print(f'write df into {self.f_df}')

    def read_df(self):
        self.df = pd.read_csv(self.f_df)
        print(f'read df from {self.f_df}')


class Scatterv0:
    host_lst = ['a_tract_21mer', 'g_tract_21mer', 'atat_21mer', 'gcgc_21mer']
    d_abbr_host = {'a_tract_21mer': 'A-tract', 'g_tract_21mer': 'G-tract', 'atat_21mer': 'TATA', 'gcgc_21mer': 'CpG'}
    d_color = {'a_tract_21mer': '#00aad4', 'g_tract_21mer': '#ff5555', 'atat_21mer': '#80ffb3', 'gcgc_21mer': '#ff7f2a'}

    lgfz = 5
    lbfz = 6
    tickfz = 4

    def __init__(self, bigtraj_folder, corr_folder):
        self.bigtraj_folder = bigtraj_folder
        self.corr_folder = corr_folder
        
        self.d_hb_agent = None
        self.ini_d_hb_agent()

        self.d_stack_agent = None
        self.ini_d_stack_agent()

        self.d_backbone_agent = None
        self.ini_d_backbone_agent()

    def scatter_main_2d(self, figsize, s):
        fig, axes = plt.subplots(nrows=3, figsize=figsize, facecolor='white')
        self.scatter_2d_hosts(axes[0], s, 'HB', 'Stack')
        self.scatter_2d_hosts(axes[1], s, 'HB', 'Backbone')
        self.scatter_2d_hosts(axes[2], s, 'Stack', 'Backbone')
        axes[0].legend(fontsize=self.lgfz, frameon=False)
        return fig, axes

    def scatter_2d_hosts(self, ax, s, interaction_x, interaction_y):
        for host in self.host_lst:
            d_function = self.get_d_function(host)
            x_array = d_function[interaction_x]()
            y_array = d_function[interaction_y]()
            label = self.d_abbr_host[host]
            ax.scatter(x_array, y_array, s=s, label=label, color=self.d_color[host])
        ax.set_xlabel(interaction_x, fontsize=self.lbfz)
        ax.set_ylabel(interaction_y, fontsize=self.lbfz)
        ax.tick_params(axis='both', labelsize=self.tickfz, length=1.5, pad=1)

    def get_d_function(self, host):
        return {'Stack': self.d_stack_agent[host].get_stack_array, 'HB': self.d_hb_agent[host].get_hb_array,
                'Backbone': self.d_backbone_agent[host].get_backbone_array}

    def scatter_main_3d(self, figsize, s):
        fig = plt.figure(figsize=figsize, facecolor='white')
        ax = fig.add_subplot(projection='3d')
        for host in self.host_lst:
            x_array = self.d_hb_agent[host].get_hb_array()
            y_array = self.d_stack_agent[host].get_stack_array()
            z_array = self.d_backbone_agent[host].get_backbone_array()
            label = self.d_abbr_host[host]
            ax.scatter(x_array, y_array, z_array, s=s, label=label)
        ax.legend()
        return fig, ax

    def ini_d_hb_agent(self):
        self.d_hb_agent = dict()
        for host in self.host_lst:
            self.d_hb_agent[host] = HBAgentv0(host, self.bigtraj_folder, self.corr_folder)
            self.d_hb_agent[host].read_df()

    def ini_d_stack_agent(self):
        self.d_stack_agent = dict()
        for host in self.host_lst:
            self.d_stack_agent[host] = StackAgentv0(host, self.bigtraj_folder, self.corr_folder)
            self.d_stack_agent[host].read_df()

    def ini_d_backbone_agent(self):
        self.d_backbone_agent = dict()
        for host in self.host_lst:
            self.d_backbone_agent[host] = BackboneAgentv0(host, self.bigtraj_folder, self.corr_folder)
            self.d_backbone_agent[host].read_df()