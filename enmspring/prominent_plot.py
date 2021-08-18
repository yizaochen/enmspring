import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from enmspring.graphs_bigtraj import ProminentModes
from enmspring.abbr import Abbreviation

class ScatterTwoStrand:
    d_strand_id = {False: 'STRAND1', True: 'STRAND2'}
    strandid_list = ['STRAND1', 'STRAND2']
    colors = ['blue', 'gray']

    def __init__(self, host, rootfolder, interval_time, figsize):
        self.host = host
        self.rootfolder = rootfolder
        self.interval_time = interval_time

        self.p_agent = ProminentModes(host, rootfolder, interval_time)
        self.ini_p_agent()

        self.df_all_modes = self.make_df_all_modes()
        self.d_df_by_strand = self.make_d_df_by_strand()

        self.figsize = figsize

    def plot_main(self, d_criteria_mean_mode_lambda, xlims, ylims, addtext):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=self.figsize, facecolor="white")
        for ax_id, strand_id in enumerate(self.strandid_list):
            ax = axes[ax_id]
            data_group = self.get_groups_by_criteria_mean_mode_lambda(ax, d_criteria_mean_mode_lambda, strand_id)
            for group_id, df_group in enumerate(data_group):
                ax.scatter(df_group['Mean-Mode-lambda'], df_group['Mean-r-alpha'], s=6, color=self.colors[group_id], alpha=0.9)
            self.set_xy_label_lims(ax, xlims, ylims, strand_id)
            if addtext:
                self.add_texts(ax, data_group[0])
        return fig, axes

    def plot_main_by_pair_criteria(self, n_sele_group, d_criteria_mean_mode_lambda, d_criteria_mean_r_alpha, xlims, ylims, addtext):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=self.figsize, facecolor="white")
        for ax_id, strand_id in enumerate(self.strandid_list):
            ax = axes[ax_id]
            df_strand = self.get_df_by_strand(strand_id)
            ax.scatter(df_strand['Mean-Mode-lambda'], df_strand['Mean-r-alpha'], s=6, color='gray', alpha=0.9)
            data_group = self.get_groups_by_two_criteria(ax, n_sele_group, d_criteria_mean_mode_lambda, d_criteria_mean_r_alpha, strand_id)
            for df_group in data_group:
                ax.scatter(df_group['Mean-Mode-lambda'], df_group['Mean-r-alpha'], s=6, color='blue', alpha=0.9)
            self.set_xy_label_lims(ax, xlims, ylims, strand_id)
            if addtext:
                for df_group in data_group:
                    self.add_texts(ax, df_group)
        return fig, axes

    def add_texts(self, ax, df_sele_group):
        offset_x = 0.05
        offset_y = 0.001
        text_x_list = df_sele_group['Mean-Mode-lambda'].tolist()
        text_y_list = df_sele_group['Mean-r-alpha'].tolist()
        text_list = df_sele_group['Mode-ID'].tolist()
        for text, x, y in zip(text_list, text_x_list, text_y_list):
            ax.text(x+offset_x, y+offset_y, text)

    def set_xy_label_lims(self, ax, xlims, ylims, strand_id):
        ax.set_xlabel(r'$\lambda_{\alpha}$', fontsize=14)
        ax.set_ylabel(r'$\left< r_{\alpha}\right>$', fontsize=14)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        host_abbr = Abbreviation.get_abbreviation(self.host)
        title = f'{host_abbr} {strand_id}'
        ax.set_title(title,fontsize=16)

    def get_groups_by_criteria_mean_mode_lambda(self, ax, d_criteria_mean_mode_lambda, strand_id):
        criteria_mean_mode_lambda = d_criteria_mean_mode_lambda[strand_id]
        df_strand = self.get_df_by_strand(strand_id)
        if criteria_mean_mode_lambda is None:
            return [df_strand]
        mask = df_strand['Mean-Mode-lambda'] >= criteria_mean_mode_lambda
        ax.axvline(criteria_mean_mode_lambda, color='red', alpha=0.2)
        return [df_strand[mask], df_strand[~mask]]

    def get_groups_by_two_criteria(self, ax, n_sele_group, d_criteria_mean_mode_lambda, d_criteria_mean_r_alpha, strand_id):
        df_lst = list()
        df_strand = self.get_df_by_strand(strand_id)
        for group_idx in range(n_sele_group):
            min_lambda, max_lambda = d_criteria_mean_mode_lambda[strand_id][group_idx]
            min_r_alpha, max_r_alpha = d_criteria_mean_r_alpha[strand_id][group_idx]
            mask = (df_strand['Mean-Mode-lambda'] >= min_lambda) & (df_strand['Mean-Mode-lambda'] < max_lambda)
            df1 = df_strand[mask]
            mask = (df1['Mean-r-alpha'] >= min_r_alpha) & (df1['Mean-r-alpha'] < max_r_alpha)
            df_lst.append(df1[mask])
            
            ax.axvline(min_lambda, color='red', alpha=0.2)
            ax.axvline(max_lambda, color='red', alpha=0.2)
            ax.axhline(min_r_alpha, color='red', alpha=0.2)
            ax.axhline(max_r_alpha, color='red', alpha=0.2)
        return df_lst

    def make_df_all_modes(self):
        x = self.p_agent.mean_modes_w
        y = self.p_agent.mean_r_alpha_array
        mode_id_list = list(range(1, len(x)+1))
        strand_id = [self.d_strand_id[self.p_agent.s_agent.decide_eigenvector_strand(mode_id)] for mode_id in mode_id_list]
        d_result = {'Mode-ID': mode_id_list, 'Mean-r-alpha': y, 'Mean-Mode-lambda': x, 'Strand-ID': strand_id}
        return pd.DataFrame(d_result)

    def get_df_all_modes(self):
        return self.df_all_modes

    def make_d_df_by_strand(self):
        return {strand_id: self.df_all_modes[self.df_all_modes['Strand-ID'] == strand_id] for strand_id in self.strandid_list}

    def get_df_by_strand(self, strand_id):
        return self.d_df_by_strand[strand_id]

    def ini_p_agent(self):
        self.p_agent.load_mean_r_alpha_array()
        self.p_agent.s_agent.process_first_small_agent()
        self.p_agent.s_agent.initialize_nodes_information()
        self.p_agent.s_agent.set_benchmark_array()
        self.p_agent.s_agent.set_strand_array()

class BoxScatterKmean(ScatterTwoStrand):
    ttfz = 16
    lbfz = 14

    def __init__(self, host, rootfolder, interval_time):
        self.host = host
        self.rootfolder = rootfolder
        self.interval_time = interval_time

        self.p_agent = ProminentModes(host, rootfolder, interval_time)
        self.ini_p_agent()

        self.df_all_modes = self.make_df_all_modes()
        self.d_df_by_strand = self.make_d_df_by_strand()

    def plot_main_by_kmean(self, strandid, n_clusters, figsize):
        df = self.d_df_by_strand[strandid]
        df_top5_percent = self.get_df_top5percent(df)
        fig = plt.figure(constrained_layout=True, figsize=figsize, facecolor='white')
        ax1, ax2, ax3 = self.get_ax123(fig)

        self.box_plot(ax1, df)
        self.set_title(ax1, strandid)
        
        self.scatter_plot(ax2, df, df_top5_percent)
        self.set_xylabel(ax2)
        xlim_ax2 = ax2.get_xlim()
        ax1.set_xlim(xlim_ax2)

        df_top5_percent = self.get_kmean_df(df_top5_percent, n_clusters)
        self.scatter_kmeans(ax3, df_top5_percent, n_clusters)
        ax3.set_title(f'Number of Group: {n_clusters}')
        self.set_xylabel(ax3)

    def plot_main_by_dmap(self, strandid, figsize):
        df = self.d_df_by_strand[strandid]
        df_top5_percent = self.get_df_top5percent(df)
        fig = plt.figure(constrained_layout=True, figsize=figsize, facecolor='white')
        ax1, ax2, ax3 = self.get_ax123(fig)

        self.box_plot(ax1, df)
        self.set_title(ax1, strandid)
        
        self.scatter_plot(ax2, df, df_top5_percent)
        self.set_xylabel(ax2)
        xlim_ax2 = ax2.get_xlim()
        ax1.set_xlim(xlim_ax2)

        df_top5_percent = self.get_label_df_by_dmap(df_top5_percent, strandid)
        n_clusters = len(list(set(df_top5_percent['kmeans_label'].tolist())))
        self.scatter_kmeans(ax3, df_top5_percent, n_clusters)
        ax3.set_title(f'Number of Group: {n_clusters}')
        self.set_xylabel(ax3)

    def get_ax123(self, fig):
        gs = fig.add_gridspec(12, 1)
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax2 = fig.add_subplot(gs[2:7, 0])
        ax3 = fig.add_subplot(gs[7:, 0])
        return ax1, ax2, ax3

    def get_df_top5percent(self, df):
        mask = df['Mean-Mode-lambda'] >= df['Mean-Mode-lambda'].quantile(0.95)
        return df[mask]

    def get_kmean_df(self, df_top5_percent, n_clusters):
        n_selection = df_top5_percent.shape[0]
        eigv_mat = np.zeros((n_selection, self.p_agent.s_agent.n_node))
        for eigv_id in range(1, n_selection+1):
            eigv_mat[eigv_id-1,:] = np.abs(self.p_agent.s_agent.get_eigenvector_by_id(eigv_id))
        labels = pd.Series(KMeans(n_clusters=n_clusters, random_state=0).fit(eigv_mat).labels_).values
        return df_top5_percent.assign(kmeans_label=labels)

    def get_label_df_by_dmap(self, df_top5_percent, strandid):
        d_map = {'a_tract_21mer': {'STRAND1': {}, 
                                   'STRAND2': {}},
                 'g_tract_21mer': {'STRAND1': {1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 7: 1, 8: 2, 10: 2, 11: 1, 13: 1, 14: 0, 15: 1}, 
                                   'STRAND2': {}}
                }
        sub_d_map = d_map[self.host][strandid]
        modeid_lst = df_top5_percent['Mode-ID'].tolist()
        labels = [sub_d_map[mode_id] for mode_id in modeid_lst]
        labels = pd.Series(labels).values
        return df_top5_percent.assign(kmeans_label=labels)

    def box_plot(self, ax, df):
        ax.boxplot([df['Mean-Mode-lambda']], vert=False, widths=4, whis=(5,95))
        ax.set_axis_off()

    def set_title(self, ax, strand_id):
        host_abbr = Abbreviation.get_abbreviation(self.host)
        title = f'{host_abbr} {strand_id}'
        ax.set_title(title,fontsize=self.ttfz)

    def scatter_plot(self, ax, df, df_top5_percent):
        ax.scatter(df['Mean-Mode-lambda'], df['Mean-r-alpha'], s=6, color='grey')
        ax.scatter(df_top5_percent['Mean-Mode-lambda'], df_top5_percent['Mean-r-alpha'], s=6, color='blue')
        
    def scatter_kmeans(self, ax, df_top5_percent, n_clusters):
        offset_x = 0.05
        offset_y = 0.001
        for kmean_label in range(n_clusters):
            mask = df_top5_percent['kmeans_label'] == kmean_label
            df_sele = df_top5_percent[mask]
            ax.scatter(df_sele['Mean-Mode-lambda'], df_sele['Mean-r-alpha'], s=10)
            
            text_x_list = df_sele['Mean-Mode-lambda'].tolist()
            text_y_list = df_sele['Mean-r-alpha'].tolist()
            text_list = df_sele['Mode-ID'].tolist()
            for text, x, y in zip(text_list, text_x_list, text_y_list):
                ax.text(x+offset_x, y+offset_y, text)

    def set_xylabel(self, ax):
        ax.set_xlabel(r'$\lambda_{\alpha}$', fontsize=self.lbfz)
        ax.set_ylabel(r'$\left< r_{\alpha}\right>$', fontsize=self.lbfz)

