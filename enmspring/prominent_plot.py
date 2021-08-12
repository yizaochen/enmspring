import matplotlib.pyplot as plt
import pandas as pd
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
                self.add_texts(ax, data_group)

        return fig, axes

    def add_texts(self, ax, data_group):
        df_sele_group = data_group[0]
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