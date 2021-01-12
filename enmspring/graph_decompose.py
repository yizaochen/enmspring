from os import path
import matplotlib.pyplot as plt
import numpy as np
from enmspring.graphs import hosts
from enmspring.graphs import Stack

class SixStack:
    d_titles = {'a_tract_21mer': ('A-Tract: (AA)', 'A-Tract: (TT)'), 'g_tract_21mer': ('G-Tract: (GG)', 'G-Tract: (CC)'),
                'atat_21mer': ('AT: (AT)', 'AT: (AT)'), 'gcgc_21mer':  ('GC: (GC)', 'GC: (GC)'),
                'ctct_21mer':  ('CT: (CT)', 'CT: (GA)'), 'tgtg_21mer': ('TG: (TG)', 'TG: (AC)')}

    def __init__(self, rootfolder):
        self.rootfolder = rootfolder
        self.g_agents = self.get_g_agents_and_preprocess()

    def get_g_agents_and_preprocess(self):
        g_agents = dict()
        for host in hosts:
            g_agents[host] = Stack(host, self.rootfolder)
            g_agents[host].pre_process()
        return g_agents

    def get_lambda_id_list(self, host, start_mode, end_mode, strandid):
        d_strand = {'STRAND1': self.g_agents[host].strand1_array, 'STRAND2': self.g_agents[host].strand2_array}
        return d_strand[strandid][start_mode:end_mode+1]

    def get_Alist_Dlist(self, host, start_mode, end_mode, strandid):
        Alist = list()
        Dlist = list()
        lambda_id_list = self.get_lambda_id_list(host, start_mode, end_mode, strandid)
        for lambda_id in lambda_id_list:
            qtAq = self.g_agents[host].get_qtAq(lambda_id)
            qtDq = self.g_agents[host].get_qtDq(lambda_id)
            Alist.append(qtAq)
            Dlist.append(qtDq)
        return Alist, Dlist

    def plot_qtAq_qtDq(self, figsize, start_mode, end_mode, strandid, ylim, yvalues):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize, sharey=True)
        idx = 0
        for row_id in range(2):
            for col_id in range(3):
                host = hosts[idx]
                ax = axes[row_id, col_id]
                self.bar_qtAq_qtDq_by_ax(ax, host, start_mode, end_mode, strandid)
                if ylim is not None:
                    ax.set_ylim(ylim)
                self.draw_assist_lines(ax, yvalues)
                idx += 1
        return fig, axes

    def bar_qtAq_qtDq_by_ax(self, ax, host, start_mode, end_mode, strandid):
        w_small = 0.5
        w_big = 0.8
        x_Alist, x_Dlist, xticks = self.get_xAD_xticks(w_small, w_big)
        Alist, Dlist = self.get_Alist_Dlist(host, start_mode, end_mode, strandid)
        ax.bar(x_Alist, Alist, w_small, label=r'$\mathbf{q}_i^{T} A \mathbf{q}_i$')
        ax.bar(x_Dlist, Dlist, w_small, label=r'$\mathbf{q}_i^{T} D \mathbf{q}_i$')
        ax.set_xticks(xticks)
        ax.set_xticklabels(range(start_mode, end_mode+1))
        ax.set_ylabel(r'Decomposed $\lambda$ (kcal/mol/Å$^2$)')
        ax.set_xlabel('Mode id, $i$')
        ax.legend()
        d_strandid = {'STRAND1': 0, 'STRAND2': 1}
        title = self.d_titles[host][d_strandid[strandid]]
        ax.set_title(title)

    def get_xAD_xticks(self, w_small, w_big):
        x_Alist = list()
        x_Dlist = list()
        xticks = list()
        x = 0.
        for idx in range(20):
            x_Alist.append(x)
            x += w_small
            x_Dlist.append(x)
            x += w_big
            xticks.append((x_Alist[idx]+x_Dlist[idx]) / 2)
        return x_Alist, x_Dlist, xticks

    def draw_assist_lines(self, ax, yvalues):
        for yvalue in yvalues:
            ax.axhline(yvalue, color='grey', alpha=0.2)
        

class AtomImportance:
    d_atomlist = {'A': ['N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N6', 'N7', 'C8', 'N9'],
                  'T': ['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6', 'C7'],
                  'C': ['N1', 'C2', 'O2', 'N3', 'N4', 'C4', 'C5', 'C6'],
                  'G': ['N1', 'C2', 'N2', 'N3', 'C4', 'C5', 'C6', 'O6', 'N7', 'C8', 'N9']}
    d_color = {'N1': 'tab:blue', 'C2': 'tab:orange', 'O2': 'tab:green', 'N3': 'tab:red',
               'C4': 'tab:purple', 'C5': 'tab:brown', 'C6': 'tab:pink', 'N7': 'tab:gray',
               'C8': 'tab:olive', 'N9': 'tab:cyan',
               'N2': 'b', 'O6': 'r', 'N6': 'lime', 'O4': 'magenta', 'C7': 'gold', 'N4': 'mistyrose'}
    d_host_strand = {'a_tract_21mer': {'STRAND1': 'A', 'STRAND2': 'T'},
                     'g_tract_21mer': {'STRAND1': 'G', 'STRAND2': 'C'},
                     'atat_21mer': ('A', 'T'),
                     'gcgc_21mer': ('G', 'C'),
                     'ctct_21mer': {'STRAND1': ('C', 'T'), 'STRAND2': ('G', 'A')},
                     'tgtg_21mer': {'STRAND1': ('T', 'G'), 'STRAND2': ('A', 'C')},
                     }
    d_titles = {'a_tract_21mer': ('A-Tract: (AA)', 'A-Tract: (TT)'), 'g_tract_21mer': ('G-Tract: (GG)', 'G-Tract: (CC)'),
                'atat_21mer': ('AT: (AT)', 'AT: (AT)'), 'gcgc_21mer':  ('GC: (GC)', 'GC: (GC)'),
                'ctct_21mer':  ('CT: (CT)', 'CT: (GA)'), 'tgtg_21mer': ('TG: (TG)', 'TG: (AC)')}            
    
    def __init__(self, host, rootfolder):
        self.host = host
        self.rootfolder = rootfolder
        self.g_agent = self.get_g_agent_and_preprocess()
        self.d_strand = {'STRAND1': self.g_agent.strand1_array, 'STRAND2': self.g_agent.strand2_array}

    def get_g_agent_and_preprocess(self):
        g_agent = Stack(self.host, self.rootfolder)
        g_agent.pre_process()
        return g_agent

    def plot_lambda_qTDq_respective_atoms(self, figsize, strandid, start_mode, end_mode, bbox_to_anchor):
        """
        strandid: 'STRAND1', 'STRAND2'
        """
        resname = self.d_host_strand[self.host][strandid]
        atomlist = self.d_atomlist[resname]
        n_mode = end_mode - start_mode + 1

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        w_small = 0.5
        w_big = 0.8
        d_xarray = self.get_d_xarray(atomlist, n_mode, w_small, w_big)
        d_result = self.get_qTDq_d_result(atomlist, n_mode, start_mode, end_mode, strandid)
        xticks = self.get_xticks(atomlist, d_xarray, w_small)
        for atomname in atomlist:
            ax.bar(d_xarray[atomname], d_result[atomname], w_small, label=atomname, color=self.d_color[atomname])
        ax.set_ylabel(r'Decomposed $\lambda$ (kcal/mol/Å$^2$)')
        ax.set_xlabel('Mode id, $i$')
        ax.set_xticks(xticks)
        ax.set_xticklabels(range(start_mode, end_mode+1))
        ax.legend(ncol=1, loc='center right', bbox_to_anchor=bbox_to_anchor)
        ax.set_title(self.get_title(strandid))
        return fig, ax

    def get_title(self, strandid):
        d_strandid = {'STRAND1': 0, 'STRAND2': 1}
        return self.d_titles[self.host][d_strandid[strandid]]

    def get_qTDq_d_result(self, atomlist, n_mode, start_mode, end_mode, strandid):
        d_result = dict()
        real_mode_id_list = self.d_strand[strandid][start_mode:end_mode+1]
        for atomname in atomlist:
            d_result[atomname] = np.zeros(n_mode)
            D_mat = self.g_agent.get_D_by_atomname_strandid(atomname, strandid)
            for idx, mode_id in enumerate(real_mode_id_list):
                q = self.g_agent.get_eigenvector_by_id(mode_id)
                qTDq = np.dot(q.T, np.dot(D_mat, q))
                d_result[atomname][idx] = qTDq
        return d_result
        
    def get_d_xarray(self, atomlist, n_mode, w_small, w_big):
        d_xarray = {atomname: np.zeros(n_mode) for atomname in atomlist}
        x = 0.
        for idx in range(n_mode):
            for atomname in atomlist:
                d_xarray[atomname][idx] = x
                x += w_small
            x += w_big
        return d_xarray

    def get_xticks(self, atomlist, d_xarray, w_small):
        n_atom = len(atomlist)
        middle_atomname_idx = int(np.floor(n_atom/2))
        middle_atomname = atomlist[middle_atomname_idx]
        if n_atom % 2 == 1:    
            return d_xarray[middle_atomname]
        else:
            return d_xarray[middle_atomname] - w_small
