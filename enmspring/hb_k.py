from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from enmspring.hb_util import HBAgentBigTraj

class HBResidPlot:
    sys_lst = ['AT', 'GC']
    hosts = ['a_tract_21mer', 'atat_21mer', 'g_tract_21mer', 'gcgc_21mer']
    d_systems = {
        'AT': ['a_tract_21mer', 'atat_21mer'],
        'GC': ['g_tract_21mer', 'gcgc_21mer']
    }
    d_abbr = {'a_tract_21mer': 'A-tract', 'atat_21mer': 'TATA', 'g_tract_21mer': 'G-tract', 'gcgc_21mer': 'CpG'}
    typelist = ['type1', 'type2', 'type3']
    resid_lst = list(range(4, 19))
    d_color = {'a_tract_21mer': 'tab:blue', 'atat_21mer': 'tab:red', 'g_tract_21mer': 'tab:cyan', 'gcgc_21mer': 'tab:orange'}

    n_bp = 21
    only_central = False
    split_5 = False
    one_big_window = False

    lgfz = 5
    lbfz = 6
    tickfz = 4

    def __init__(self, bigtraj_folder, interval_time, df_folder):
        self.bigtraj_folder = bigtraj_folder
        self.interval_time = interval_time
        self.df_folder = df_folder

        self.f_df_mean = path.join(df_folder, 'hb.mean.csv')
        self.f_df_std = path.join(df_folder, 'hb.std.csv')
        self.df_mean = None
        self.df_std = None

    def plot_hb_vs_resids(self, figsize):
        fig, d_axes = self.get_d_axes(figsize)
        for sys_name in self.sys_lst:
            for type_name in self.typelist:
                ax = d_axes[sys_name][type_name]
                self.plot_lines(ax, sys_name, type_name)
        self.set_yaxis_right(d_axes)
        self.set_ylims(d_axes)
        self.set_legend(d_axes)
        self.set_yticks(d_axes)
        self.set_xticks(d_axes)
        self.remove_xticks(d_axes)
        self.set_xtick_size(d_axes)
        self.set_ytick_size(d_axes)
        self.set_xlabel(d_axes)
        self.set_ylabel(d_axes)
        return fig, d_axes

    def plot_lines(self, ax, sys_name, type_name):
        for host in self.d_systems[sys_name]:
            xarray = self.get_xarray()
            yarray, y_std_array = self.get_yarray(host, type_name)
            label = self.d_abbr[host]
            ax.errorbar(xarray, yarray, yerr=y_std_array, marker='.', color=self.d_color[host], linewidth=0.5, markersize=2, label=label)

    def set_yaxis_right(self, d_axes):
        sys_name = 'GC'
        for type_name in self.typelist:
            d_axes[sys_name][type_name].yaxis.tick_right()
            d_axes[sys_name][type_name].yaxis.set_label_position("right")

    def set_ylims(self, d_axes):
        ylims = (0, 11.5)
        for sys_name in self.sys_lst:
            for type_name in self.typelist:
                d_axes[sys_name][type_name].set_ylim(ylims)

    def set_yticks(self, d_axes):
        yticks = range(2, 11, 2)
        for sys_name in self.sys_lst:
            for type_name in self.typelist:
                d_axes[sys_name][type_name].set_yticks(yticks)

    def set_xticks(self, d_axes):
        xticks = range(4, 19)
        for sys_name in self.sys_lst:
            for type_name in self.typelist:
                d_axes[sys_name][type_name].set_xticks(xticks)

    def remove_xticks(self, d_axes):
        for sys_name in self.sys_lst:
            for type_name in ['type1', 'type2']:
                d_axes[sys_name][type_name].tick_params(axis='x', bottom=False, top=False, labelbottom=False)

    def set_xtick_size(self, d_axes):
        type_name = 'type3'
        for sys_name in self.sys_lst:
            d_axes[sys_name][type_name].tick_params(axis='x', labelsize=self.tickfz, length=1.5, pad=1)

    def set_xlabel(self, d_axes):
        type_name = 'type3'
        for sys_name in self.sys_lst:
            d_axes[sys_name][type_name].set_xlabel('Base Pair ID', fontsize=self.lbfz)
    
    def set_ytick_size(self, d_axes):
        for sys_name in self.sys_lst:
            for type_name in self.typelist:
                d_axes[sys_name][type_name].tick_params(axis='y', labelsize=self.tickfz, length=1.5, pad=1)

    def set_ylabel(self, d_axes):
        for type_name in self.typelist:
            d_axes['AT'][type_name].set_ylabel('k (kcal/mol/Å$^2$)', fontsize=self.lbfz)
            #d_axes['GC'][type_name].set_ylabel('k (kcal/mol/Å$^2$)', fontsize=self.lbfz, rotation=90)

    def set_legend(self, d_axes):
        type_name = 'type1'
        for sys_name in self.sys_lst:
            d_axes[sys_name][type_name].legend(fontsize=self.lgfz, frameon=False)

    def get_xarray(self):
        return np.array(self.resid_lst)

    def get_yarray(self, host, type_name):
        key = f'{host}-{type_name}'
        yarray = np.array(self.df_mean[key].iloc[3:18])
        y_std_array = np.array(self.df_std[key].iloc[3:18])
        return yarray, y_std_array

    def get_d_axes(self, figsize):
        fig = plt.figure(figsize=figsize, facecolor='white')
        d_axes = {sys_name: dict() for sys_name in self.sys_lst}
        outer_grid = gridspec.GridSpec(1, 2, wspace=0, hspace=0)
        inner_grid_lst = [gridspec.GridSpecFromSubplotSpec(3, 1, hspace=0, subplot_spec=outer_grid[idx]) for idx in range(len(self.sys_lst))]
        for inner_idx, sys_name in enumerate(self.sys_lst):
            inner_grid = inner_grid_lst[inner_idx]
            for idx, type_name in enumerate(self.typelist):
                d_axes[sys_name][type_name] = fig.add_subplot(inner_grid[idx])
        return fig, d_axes

    def get_d_hb_agents(self):
        d_hb_agents = dict()
        for host in self.hosts:
            d_hb_agents[host] = HBAgentBigTraj(host, self.bigtraj_folder, self.n_bp, self.only_central, self.split_5, self.one_big_window, self.interval_time)
            d_hb_agents[host].initialize_basepair()
        return d_hb_agents

    def make_mean_std_df(self):
        resid_list = list(range(1, self.n_bp+1))
        d_hb_agents = self.get_d_hb_agents()
        d_mean = self.initialize_d_mean_d_std()
        d_std = self.initialize_d_mean_d_std()
        for host in self.hosts:
            k_container = d_hb_agents[host].get_k_container()
            for type_name in self.typelist:
                key = f'{host}-{type_name}'
                d_mean[key] = [k_container[resid][type_name].mean() for resid in resid_list]
                d_std[key] = [k_container[resid][type_name].std() for resid in resid_list]
        df_mean = pd.DataFrame(d_mean)
        df_std = pd.DataFrame(d_std)
        df_mean.to_csv(self.f_df_mean, index=False)
        df_std.to_csv(self.f_df_std, index=False)
        print(f'Write df_mean to {self.f_df_mean}')
        print(f'Write df_std to {self.f_df_std}')

    def read_mean_std_df(self):
        self.df_mean = pd.read_csv(self.f_df_mean)
        self.df_std = pd.read_csv(self.f_df_std)
        print(f'Read df_mean from {self.f_df_mean}')
        print(f'Read df_std from {self.f_df_std}')

    def initialize_d_mean_d_std(self):
        d_temp = dict()
        for host in self.hosts:
            for type_name in self.typelist:
                key = f'{host}-{type_name}'
                d_temp[key] = None
        return d_temp

class HBResidPlotV1(HBResidPlot):
    hosts = ['a_tract_21mer', 'g_tract_21mer', 'atat_21mer', 'gcgc_21mer']
    d_color = {'type1': 'tab:blue', 'type2': 'tab:orange', 'type3': 'tab:red'}
    d_abbr = {'a_tract_21mer': {'type1': 'rN6-yO4', 'type2': 'rN1-yN3', 'type3': 'rC2-yO2'}, 
              'atat_21mer': {'type1': 'rN6-yO4', 'type2': 'rN1-yN3', 'type3': 'rC2-yO2'}, 
              'g_tract_21mer': {'type1': 'rO6-yN4', 'type2': 'rN1-yN3', 'type3': 'rN2-yO2'}, 
              'gcgc_21mer': {'type1': 'rO6-yN4', 'type2': 'rN1-yN3', 'type3': 'rN2-yO2'}}
    group1 = ['a_tract_21mer', 'g_tract_21mer']
    group2 = ['atat_21mer', 'gcgc_21mer']
    group3 = ['a_tract_21mer', 'atat_21mer']
    group4 = ['g_tract_21mer', 'gcgc_21mer']

    def plot_hb_vs_resids(self, figsize, out_wspace):
        fig, d_axes = self.get_d_axes(figsize, out_wspace)
        for host in self.hosts:
            self.plot_lines(d_axes[host], host)
        self.set_yaxis_right(d_axes)
        #self.set_ylims(d_axes)
        self.set_ylims_all(d_axes)
        self.set_legend(d_axes)
        self.set_yticks(d_axes)
        self.set_xticks(d_axes)
        self.remove_xticks(d_axes)
        self.set_xtick_size(d_axes)
        self.set_ytick_size(d_axes)
        self.set_xlabel(d_axes)
        self.set_ylabel(d_axes)
        return fig, d_axes

    def plot_lines(self, ax, host):
        xarray = self.get_xarray()
        for type_name in self.typelist:
            yarray, y_std_array = self.get_yarray(host, type_name)
            label = self.d_abbr[host][type_name]
            ax.errorbar(xarray, yarray, yerr=y_std_array, marker='.', color=self.d_color[type_name], linewidth=0.5, markersize=2, label=label)

    def set_xlabel(self, d_axes):
        for host in self.group4:
            d_axes[host].set_xlabel('Base Pair ID', fontsize=self.lbfz)

    def set_ylabel(self, d_axes):
        for host in self.group1:
            d_axes[host].set_ylabel('k (kcal/mol/Å$^2$)', fontsize=self.lbfz)

    def set_xtick_size(self, d_axes):
        for host in self.group4:
            d_axes[host].tick_params(axis='x', labelsize=self.tickfz, length=1.5, pad=1)

    def set_ytick_size(self, d_axes):
        for host in self.hosts:
            d_axes[host].tick_params(axis='y', labelsize=self.tickfz, length=1.5, pad=1)

    def set_yticks(self, d_axes):
        yticks = range(2, 11, 2)
        for host in self.hosts:
            d_axes[host].set_yticks(yticks)

    def set_xticks(self, d_axes):
        xticks = range(4, 19)
        for host in self.hosts:
            d_axes[host].set_xticks(xticks)

    def remove_xticks(self, d_axes):
        for host in self.group3:
            d_axes[host].tick_params(axis='x', bottom=False, top=False, labelbottom=False)

    def set_yaxis_right(self, d_axes):
        for host in self.group2:
            d_axes[host].yaxis.tick_right()
            d_axes[host].yaxis.set_label_position("right")

    def set_ylims_all(self, d_axes):
        ylims = (0, 11.2)
        for host in self.hosts:
            d_axes[host].set_ylim(ylims)

    def set_ylims(self, d_axes):
        ylims = (0, 8)
        for host in self.group3:
            d_axes[host].set_ylim(ylims)
        ylims = (2, 11.2)
        for host in self.group4:
            d_axes[host].set_ylim(ylims)

    def set_legend(self, d_axes):
        host = 'atat_21mer'
        d_axes[host].legend(fontsize=self.lgfz, frameon=False)
        host = 'gcgc_21mer'
        d_axes[host].legend(fontsize=self.lgfz, frameon=False, ncol=3, columnspacing=0.3)

    def get_d_axes(self, figsize, out_wspace):
        fig = plt.figure(figsize=figsize, facecolor='white')
        d_axes = {host: None for host in self.hosts}
        outer_grid = gridspec.GridSpec(1, 2, wspace=out_wspace, hspace=0)
        host_idx = 0
        for outer_idx in range(2):
            inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, hspace=0, subplot_spec=outer_grid[outer_idx])
            for inner_idx in range(2):
                host = self.hosts[host_idx]
                d_axes[host] = fig.add_subplot(inner_grid[inner_idx])
                host_idx += 1
        return fig, d_axes