from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enmspring.graphs_bigtraj import BackboneMeanModeAgent
from enmspring import pairtype
from enmspring.k_b0_util import get_df_by_filter_PP, get_df_by_filter_R, get_df_by_filter_RB

class BackboneRiboseK:
    interval_time = 500
    column_lst = ['PairType', 'Big_Category', 'Strand_i', 'Resid_i',  
                  'Atomname_i', 'Strand_j', 'Resid_j', 'Atomname_j', 'k-mean', 'k-std', 'b0', 'b0-std']
    n_bp = 21
    strand_lst = ['STRAND1', 'STRAND2']
    resid_lst = list(range(4, 19))

    def __init__(self, host, big_traj_folder, df_folder):
        self.host = host
        self.big_traj_folder = big_traj_folder
        self.df_folder = df_folder

        self.mean_mode_agent = None
        self.n_node = None
        self.d_idx_inverse = None

        self.f_df = path.join(self.df_folder, f'{self.host}.csv')
        self.df = None

    def plot_kmean_by_resid(self, figsize, category, ylim=None):
        df_filter = self.get_df_category(category)
        fig, axes = plt.subplots(nrows=2, figsize=figsize)
        for idx, strand_id in enumerate(self.strand_lst):
            ax = axes[idx]
            yarray, y_std_array = self.get_yarray_ystd_array(df_filter, strand_id)
            ax.errorbar(self.resid_lst, yarray, yerr=y_std_array)
            ax.set_xticks(self.resid_lst)
            ax.set_xlabel('Resid')
            ax.set_ylabel(f'{category} k (kcal/mol/Å$^2$)')
            ax.set_title(f'{strand_id}')
            if ylim is not None:
                ax.set_ylim(ylim)
        return fig, axes

    def get_df_category(self, category):
        if category in ['PP0', 'PP1', 'PP2']:
            return get_df_by_filter_PP(self.df, category)
        elif category in ['R0', 'R1']:
            return get_df_by_filter_R(self.df, category)
        elif category in ['RB0', 'RB1', 'RB2']:
            return get_df_by_filter_RB(self.df, category)
        else:
            print('Only PP0 PP1 PP2 R0 R1 RB0 RB1 RB2 are allowed here')
            return None

    def get_yarray_ystd_array(self, df, strand_id):
        yarray = np.zeros(len(self.resid_lst))
        y_std_array = np.zeros(len(self.resid_lst))
        for idx, resid in enumerate(self.resid_lst):
            mask = (df['Resid_i'] == resid) & (df['Resid_j'] == resid) & (df['Strand_i'] == strand_id)
            df_sele = df[mask]
            if df_sele.shape[0] == 0:
                continue
            yarray[idx] = df_sele['k-mean'].mean()
            y_std_array[idx] = df_sele['k-mean'].std()
        return yarray, y_std_array

    def initialize_mean_mode_agent(self):
        self.mean_mode_agent = BackboneMeanModeAgent(self.host, self.big_traj_folder, self.interval_time)
        self.mean_mode_agent.load_mean_mode_laplacian_from_npy()
        self.mean_mode_agent.load_mean_mode_std_laplacian_from_npy()
        self.mean_mode_agent.load_b0_mean_std_from_npy()
        self.mean_mode_agent.process_first_small_agent()
        self.mean_mode_agent.initialize_all_maps()
        self.n_node = len(self.mean_mode_agent.node_list)
        self.d_idx_inverse = {y:x for x,y in self.mean_mode_agent.d_idx.items()}

    def get_df_filter(self):
        d_result = {column_key: list() for column_key in ['i', 'j', 'k-mean', 'k-std', 'b0-mean', 'b0-std']}
        for i in range(self.n_node):
            for j in range(self.n_node):
                if i == j:
                    continue
                d_result['i'].append(i)
                d_result['j'].append(j)
                d_result['k-mean'].append(self.mean_mode_agent.laplacian_mat[i,j])
                d_result['k-std'].append(self.mean_mode_agent.laplacian_std_mat[i,j])
                d_result['b0-mean'].append(self.mean_mode_agent.b0_mean_mat[i,j])
                d_result['b0-std'].append(self.mean_mode_agent.b0_std_mat[i,j])
        df_result = pd.DataFrame(d_result)
        mask = df_result['k-mean'] > 1e-1
        return df_result[mask]

    def make_df(self):
        df_filter = self.get_df_filter()
        d_result = {column_key: list() for column_key in self.column_lst}
        for i, j in zip(df_filter['i'], df_filter['j']):
            cgname_i = self.d_idx_inverse[i]
            cgname_j = self.d_idx_inverse[j]
            strandid1 = self.mean_mode_agent.strandid_map[cgname_i]
            strandid2 = self.mean_mode_agent.strandid_map[cgname_j]
            resid1 = self.mean_mode_agent.resid_map[cgname_i]
            resid2 = self.mean_mode_agent.resid_map[cgname_j]
            atomname1 = self.mean_mode_agent.atomname_map[cgname_i]
            atomname2 = self.mean_mode_agent.atomname_map[cgname_j]
            temp = pairtype.Pair(strandid1, resid1, atomname1, strandid2, resid2, atomname2, n_bp=self.n_bp)
            d_result['PairType'].append(temp.pair_type)
            d_result['Big_Category'].append(temp.big_category)
            d_result['Strand_i'].append(strandid1)
            d_result['Resid_i'].append(resid1)
            d_result['Atomname_i'].append(atomname1)
            d_result['Strand_j'].append(strandid2)
            d_result['Resid_j'].append(resid2)
            d_result['Atomname_j'].append(atomname2)
        d_result['k-mean'] = df_filter['k-mean'].tolist()
        d_result['k-std'] = df_filter['k-std'].tolist()
        d_result['b0'] = df_filter['b0-mean'].tolist()
        d_result['b0-std'] = df_filter['b0-std'].tolist()
        df_result = pd.DataFrame(d_result)
        df_result.to_csv(self.f_df, index=False)
        print(f'Write DataFrame to {self.f_df}')

    def read_df(self):
        self.df = pd.read_csv(self.f_df)
        print(f'Read DataFrame from {self.f_df}')

class BackboneRiboseResid:
    hosts = ['a_tract_21mer', 'atat_21mer', 'g_tract_21mer', 'gcgc_21mer']

    def __init__(self):
        pass

    def plot_all_hosts(self):
        pass
