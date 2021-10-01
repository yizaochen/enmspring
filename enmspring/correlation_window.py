from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enmspring.hb_util import HBAgentBigTraj
from enmspring.graphs_bigtraj import BackboneMeanModeAgent, StackMeanModeAgent
from enmspring.na_seq import sequences

class HBAgent:
    n_bp = 21
    only_central = False
    split_5 = False
    one_big_window = False
    interval_time = 500

    resid_lst = list(range(4, 19))
    type_lst = ['type1', 'type2', 'type3']
    d_abbr_hb = {'HB1': 'type1', 'HB2': 'type2', 'HB3': 'type3'}

    def __init__(self, host, bigtraj_folder, corr_folder):
        self.host = host
        self.bigtraj_folder = bigtraj_folder
        self.corr_folder = corr_folder

        self.corr_input_folder = path.join(self.corr_folder, 'input')
        self.f_df = path.join(self.corr_input_folder, f'{host}_hb.csv')
        self.df = None

        self.hb_reader = None

    def get_hb_array(self, interaction, strand_id):
        typename = self.d_abbr_hb[interaction]
        data_lst = list()
        if strand_id == 'STRAND1':
            resid_lst = list(range(4, 19))
        else:
            resid_lst = list(range(18, 3, -1))
        for resid in resid_lst:
            key = f'{resid}-{typename}'
            data_lst += self.df[key].tolist()
        return np.array(data_lst)

    def set_hb_reader(self):
        self.hb_reader = HBAgentBigTraj(self.host, self.bigtraj_folder, self.n_bp, self.only_central, self.split_5, self.one_big_window, self.interval_time)
        self.hb_reader.initialize_basepair()        

    def make_df(self):
        self.set_hb_reader()
        k_container = self.hb_reader.get_k_container()
        d_result = dict()
        for resid in self.resid_lst:
            for typename in self.type_lst:
                key = f'{resid}-{typename}'
                d_result[key] = k_container[resid][typename]
        self.df = pd.DataFrame(d_result)
        self.df.to_csv(self.f_df, index=False)
        print(f'write df into {self.f_df}')

    def read_df(self):
        self.df = pd.read_csv(self.f_df)
        print(f'read df from {self.f_df}')

class BackbonePair:
    d_pairs = {
    "C2'(i)-P(i+1)": {
        'A': {'atomname_i': "C2'", 'atomname_j': "P"},
        'T': {'atomname_i': "C2'", 'atomname_j': "P"},
        'G': {'atomname_i': "C2'", 'atomname_j': "P"},
        'C': {'atomname_i': "C2'", 'atomname_j': "P"}
        },
    "C2'(i)-O1P(i+1)": {
        'A': {'atomname_i': "C2'", 'atomname_j': "O1P"},
        'T': {'atomname_i': "C2'", 'atomname_j': "O1P"},
        'G': {'atomname_i': "C2'", 'atomname_j': "O1P"},
        'C': {'atomname_i': "C2'", 'atomname_j': "O1P"}
        },
    "C3'(i)-O2P(i+1)": {
        'A': {'atomname_i': "C3'", 'atomname_j': "O2P"},
        'T': {'atomname_i': "C3'", 'atomname_j': "O2P"},
        'G': {'atomname_i': "C3'", 'atomname_j': "O2P"},
        'C': {'atomname_i': "C3'", 'atomname_j': "O2P"}
        },
    "C1'-N3/C1'-O2": {
        'A': {'atomname_i': "C1'", 'atomname_j': "N3"},
        'T': {'atomname_i': "C1'", 'atomname_j': "O2"},
        'G': {'atomname_i': "C1'", 'atomname_j': "N3"},
        'C': {'atomname_i': "C1'", 'atomname_j': "O2"}
    }, 
    "O4'-O5'": {
        'A': {'atomname_i': "O4'", 'atomname_j': "O5'"},
        'T': {'atomname_i': "O4'", 'atomname_j': "O5'"},
        'G': {'atomname_i': "O4'", 'atomname_j': "O5'"},
        'C': {'atomname_i': "O4'", 'atomname_j': "O5'"}
    }
    }

class BackboneAgent:
    pair_lst = ["C2'(i)-P(i+1)", "C3'(i)-O2P(i+1)", "C2'(i)-O1P(i+1)", "C1'-N3/C1'-O2", "O4'-O5'"]
    d_pair_abbr = {'BB1': "C2'(i)-P(i+1)", 'BB2': "C3'(i)-O2P(i+1)", 'BB3': "C2'(i)-O1P(i+1)", 'BB4': "C1'-N3/C1'-O2", 'BB5': "O4'-O5'"}
    pair_abbr_lst = ['BB1', 'BB2', 'BB3', 'BB4', 'BB5']

    resid_lst = list(range(3, 19)) # Notice: Different from HB
    interval_time = 500

    mean_mode_obj = BackboneMeanModeAgent

    def __init__(self, host, strand_id, bigtraj_folder, corr_folder):
        self.host = host
        self.strand_id = strand_id
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

    def get_bb1tobb3_array(self, interaction, upper_or_lower):
        if upper_or_lower == 'upper':
            resid_lst = range(3, 18)
        else:
            resid_lst = range(4, 19)
        data_lst = list()
        for resid_i in resid_lst:
            resid_j = resid_i + 1
            key = f'{resid_i}-{resid_j}-{interaction}'
            data_lst += self.df[key].tolist()
        return np.array(data_lst)

    def get_bb4tobb5_array(self, interaction):
        resid_lst = list(range(4, 19))
        data_lst = list()
        for resid_i in resid_lst:
            resid_j = resid_i
            key = f'{resid_i}-{resid_j}-{interaction}'
            data_lst += self.df[key].tolist()
        return np.array(data_lst)

    def make_df(self):
        self.set_laplacian_from_small_agents()
        d_result = dict()
        for resid_i in self.resid_lst:
            for pair_abbr in self.pair_abbr_lst:
                resid_j = self.get_resid_j_by_resid_i_pair_abbr(resid_i, pair_abbr)
                key = f'{resid_i}-{resid_j}-{pair_abbr}'
                d_result[key] = self.get_k_array_by_resid_i_pair_abbr(resid_i, pair_abbr)
        self.df = pd.DataFrame(d_result)
        self.df.to_csv(self.f_df, index=False)
        print(f'write df into {self.f_df}')

    def read_df(self):
        self.df = pd.read_csv(self.f_df)
        print(f'read df from {self.f_df}')

    def get_f_df(self):
        return path.join(self.corr_input_folder, f'{self.host}_{self.strand_id}_backbone.csv')

    def set_laplacian_from_small_agents(self):
        self.mean_mode_agent = self.mean_mode_obj(self.host, self.bigtraj_intv_folder, self.interval_time)
        self.time_list = self.mean_mode_agent.time_list
        self.n_window = len(self.time_list)
        self.d_small_agents = self.mean_mode_agent.get_all_small_agents()
        for time1, time2 in self.time_list:
            self.d_small_agents[(time1,time2)].pre_process()
        self.first_small_agent = self.d_small_agents[self.time_list[0]]

    def get_k_array_by_resid_i_pair_abbr(self, resid_i, pair_abbr):
        k_array = np.zeros(self.n_window)
        for window_id in range(self.n_window):
            key = self.time_list[window_id]
            idx_i, idx_j = self.get_idx_ij_by_resid_i_pair_abbr(resid_i, pair_abbr)
            k_array[window_id] = self.d_small_agents[key].laplacian_mat[idx_i, idx_j]
        return k_array

    def get_idx_ij_by_resid_i_pair_abbr(self, resid_i, pair_abbr):
        resid_j = self.get_resid_j_by_resid_i_pair_abbr(resid_i, pair_abbr)
        idx_i = self.get_idx_by_resid_pair_abbr(resid_i, pair_abbr, 'atomname_i')
        idx_j = self.get_idx_by_resid_pair_abbr(resid_j, pair_abbr, 'atomname_j')
        return idx_i, idx_j

    def get_idx_by_resid_pair_abbr(self, resid, pair_abbr, i_or_j):
        resname = self.d_seq[self.strand_id][resid-1]
        atomname = BackbonePair.d_pairs[self.d_pair_abbr[pair_abbr]][resname][i_or_j]
        selection = f"segid {self.strand_id} and resid {resid} and name {atomname}"
        cgname = self.first_small_agent.map[selection]
        return self.first_small_agent.d_idx[cgname]

    def get_resid_j_by_resid_i_pair_abbr(self, resid_i, pair_abbr):
        if pair_abbr in ['BB1', 'BB2', 'BB3']:
            return resid_i + 1
        else:
            return resid_i


class StackPair:
    d_pairs = {
    'ST1': {
        'a_tract_21mer': {
            'A': {'atomname_i': "N1", 'atomname_j': "C6"},
            'T': {'atomname_i': "N3", 'atomname_j': "C4"}
        },
        'g_tract_21mer': {
            'G': {'atomname_i': "N1", 'atomname_j': "C6"},
            'C': {'atomname_i': "N3", 'atomname_j': "C4"}
        },
        'atat_21mer': {
            'A': {'atomname_i': "C4", 'atomname_j': "C5"},
            'T': {'atomname_i': "C4", 'atomname_j': "C5"}
        },
        'gcgc_21mer': {
            'G': {'atomname_i': "C4", 'atomname_j': "C4"},
            'C': {'atomname_i': "C4", 'atomname_j': "C4"}
        }
        }
    }

class StackAgent(BackboneAgent):
    pair_lst = ['ST1']
    d_pair_abbr = {'ST1': 'ST1'}
    pair_abbr_lst = ['ST1']

    mean_mode_obj = StackMeanModeAgent

    def get_st1_array(self, interaction, upper_or_lower):
        if upper_or_lower == 'upper':
            resid_lst = range(3, 18)
        else:
            resid_lst = range(4, 19)
        data_lst = list()
        for resid_i in resid_lst:
            resid_j = resid_i + 1
            key = f'{resid_i}-{resid_j}-{interaction}'
            data_lst += self.df[key].tolist()
        return np.array(data_lst)

    def get_f_df(self):
        return path.join(self.corr_input_folder, f'{self.host}_{self.strand_id}_basestack.csv')

    def get_resid_j_by_resid_i_pair_abbr(self, resid_i, pair_abbr):
        return resid_i + 1

    def get_idx_by_resid_pair_abbr(self, resid, pair_abbr, i_or_j):
        resname = self.d_seq[self.strand_id][resid-1]
        atomname = StackPair.d_pairs[pair_abbr][self.host][resname][i_or_j]
        selection = f"segid {self.strand_id} and resid {resid} and name {atomname}"
        cgname = self.first_small_agent.map[selection]
        return self.first_small_agent.d_idx[cgname]


class CorrelationScatterv0:
    d_abbr_host = {'a_tract_21mer': 'A-tract', 'g_tract_21mer': 'G-tract', 'atat_21mer': 'TATA', 'gcgc_21mer': 'CpG'}
    strand_lst = ['STRAND1', 'STRAND2']
    lbfz = 12
    ttfz = 14

    def __init__(self, host, bigtraj_folder, corr_folder):
        self.host = host
        self.bigtraj_folder = bigtraj_folder
        self.corr_folder = corr_folder

        self.hb_agent = None
        self.ini_hb_agent()

        self.d_stack_agent = None
        self.ini_d_stack_agent()
        self.d_backbone_agent = None
        self.ini_d_backbone_agent()

    def scatter_single(self, strand_id, interaction_x, interaction_y, figsize, upper_or_lower_bb='upper', upper_or_lower_st='upper'):
        """
        interaction_x, interaction_y: HB1, HB2, HB3, BB1-BB5, ST1
        """
        fig, ax = plt.subplots(figsize=figsize)
        x_array = self.get_data_array_by_interaction(interaction_x, strand_id, upper_or_lower_bb, upper_or_lower_st)
        y_array = self.get_data_array_by_interaction(interaction_y, strand_id, upper_or_lower_bb, upper_or_lower_st)
        ax.scatter(x_array, y_array)
        ax.set_xlabel(f'{interaction_x} (kcal/mol/Å$^2$)', fontsize=self.lbfz)
        ax.set_ylabel(f'{interaction_y} (kcal/mol/Å$^2$)', fontsize=self.lbfz)
        host_abbr = self.d_abbr_host[self.host]
        ax.set_title(f'{host_abbr} {strand_id}', fontsize=self.ttfz)
        return fig, ax

    def get_data_array_by_interaction(self, interaction, strand_id, upper_or_lower_bb, upper_or_lower_st):
        if interaction in ['HB1', 'HB2', 'HB3']:
            data_array = self.hb_agent.get_hb_array(interaction, strand_id)
        elif interaction in ['BB1', 'BB2', 'BB3']:
            data_array = self.d_backbone_agent[strand_id].get_bb1tobb3_array(interaction, upper_or_lower_bb)
        elif interaction in ['BB4', 'BB5']:
            data_array = self.d_backbone_agent[strand_id].get_bb4tobb5_array(interaction)
        else:
            data_array = self.d_stack_agent[strand_id].get_st1_array(interaction, upper_or_lower_st)
        return data_array

    def ini_hb_agent(self):
        self.hb_agent = HBAgent(self.host, self.bigtraj_folder, self.corr_folder)
        self.hb_agent.read_df()

    def ini_d_stack_agent(self):
        self.d_stack_agent = dict()
        for strand_id in self.strand_lst:
            self.d_stack_agent[strand_id] = StackAgent(self.host, strand_id, self.bigtraj_folder, self.corr_folder)
            self.d_stack_agent[strand_id].read_df()

    def ini_d_backbone_agent(self):
        self.d_backbone_agent = dict()
        for strand_id in self.strand_lst:
            self.d_backbone_agent[strand_id] = BackboneAgent(self.host, strand_id, self.bigtraj_folder, self.corr_folder)
            self.d_backbone_agent[strand_id].read_df()

class BigScatterv0:
    d_abbr_host = {'a_tract_21mer': 'A-tract', 'g_tract_21mer': 'G-tract', 'atat_21mer': 'TATA', 'gcgc_21mer': 'CpG'}
    host_lst = ['a_tract_21mer', 'g_tract_21mer', 'atat_21mer', 'gcgc_21mer']
    strand_lst = ['STRAND1', 'STRAND2']
    lgfz = 12
    lbfz = 12
    ttfz = 14

    def __init__(self, bigtraj_folder, corr_folder):
        self.bigtraj_folder = bigtraj_folder
        self.corr_folder = corr_folder

        self.d_corr_agent = None
        self.ini_d_corr_agent()

    def ini_d_corr_agent(self):
        self.d_corr_agent = dict()
        for host in self.host_lst:
            self.d_corr_agent[host] = CorrelationScatterv0(host, self.bigtraj_folder, self.corr_folder)

    def scatter_single(self, interaction_x, interaction_y, figsize, upper_or_lower_bb='upper', upper_or_lower_st='upper'):
        """
        interaction_x, interaction_y: HB1, HB2, HB3, BB1-BB5, ST1
        """
        fig, ax = plt.subplots(figsize=figsize)
        for host in self.host_lst:
            agent = self.d_corr_agent[host]
            host_abbr = self.d_abbr_host[host]
            for strand_id in self.strand_lst:
                x_array = agent.get_data_array_by_interaction(interaction_x, strand_id, upper_or_lower_bb, upper_or_lower_st)
                y_array = agent.get_data_array_by_interaction(interaction_y, strand_id, upper_or_lower_bb, upper_or_lower_st)
                label = f'{host_abbr} {strand_id}'
                ax.scatter(x_array, y_array, label=label)
        ax.set_xlabel(f'{interaction_x} (kcal/mol/Å$^2$)', fontsize=self.lbfz)
        ax.set_ylabel(f'{interaction_y} (kcal/mol/Å$^2$)', fontsize=self.lbfz)
        ax.legend(fontsize=self.lgfz)
        return fig, ax
