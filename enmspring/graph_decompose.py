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
                'atat_21mer': ('ATAT: (TA)', 'ATAT: (AT)'), 'gcgc_21mer':  ('GCGC: (CG)', 'GCGC: (GC)'),
                'ctct_21mer': ('CTCT: (TC)', 'CTCT: (AG)'), 
                'tgtg_21mer': ('TGTG: (GT)', 'TGTG: (CA)')}
    abbr_host = {'a_tract_21mer': 'A-Tract', 'g_tract_21mer': 'G-Tract', 'atat_21mer': 'ATAT',
                 'gcgc_21mer': 'GCGC', 'ctct_21mer': 'CTCT', 'tgtg_21mer': 'TGTG'}      
    
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

    def plot_lambda_qTDq_respective_atoms_by_resname(self, figsize, strandid, resname, start_mode, end_mode, bbox_to_anchor):
        """
        strandid: 'STRAND1', 'STRAND2'
        resname: 'A', 'T', 'C', 'G'
        """
        atomlist = self.d_atomlist[resname]
        n_mode = end_mode - start_mode + 1

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        w_small = 0.5
        w_big = 0.8
        d_xarray = self.get_d_xarray(atomlist, n_mode, w_small, w_big)
        d_result = self.get_qTDq_d_result_by_resname(atomlist, n_mode, start_mode, end_mode, strandid, resname)
        xticks = self.get_xticks(atomlist, d_xarray, w_small)
        for atomname in atomlist:
            ax.bar(d_xarray[atomname], d_result[atomname], w_small, label=atomname, color=self.d_color[atomname])
        ax.set_ylabel(r'Decomposed $\lambda$ (kcal/mol/Å$^2$)')
        ax.set_xlabel('Mode id, $i$')
        ax.set_xticks(xticks)
        ax.set_xticklabels(range(start_mode, end_mode+1))
        ax.legend(ncol=1, loc='center right', bbox_to_anchor=bbox_to_anchor)
        ax.set_title(self.get_title_by_resname(resname))
        return fig, ax

    def get_title(self, strandid):
        d_strandid = {'STRAND1': 0, 'STRAND2': 1}
        return self.d_titles[self.host][d_strandid[strandid]]

    def get_title_by_resname(self, resname):
        return self.d_titles[self.host][resname]

    def get_title_by_resname_i_resname_j(self, resname_i, resname_j):
        str1 = self.abbr_host[self.host]
        str2 = f'{resname_i}{resname_j}'
        return f'{str1}: {str2}'

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

    def get_qTDq_d_result_by_resname(self, atomlist, n_mode, start_mode, end_mode, strandid, resname):
        d_result = dict()
        real_mode_id_list = self.d_strand[strandid][start_mode:end_mode+1]
        for atomname in atomlist:
            d_result[atomname] = np.zeros(n_mode)
            D_mat = self.g_agent.get_D_by_atomname_strandid_resname(atomname, strandid, resname)
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


class PairImportance(AtomImportance):

    def plot_lambda_qTAq_respective_atoms_single_mode(self, figsize, strandid, mode_id_strand, bbox_to_anchor, ylim=None, assist_lines=None):
        mode_id_molecule = self.d_strand[strandid][mode_id_strand-1]
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        self.plot_lambda_qTAq_respective_atoms_one_mode(ax, strandid, mode_id_molecule, mode_id_strand, bbox_to_anchor, ylim, assist_lines)
        return fig, ax

    def plot_lambda_qTAq_respective_atoms_five_modes(self, figsize, strandid, start_mode, end_mode, bbox_to_anchor, ylim=None, assist_lines=None):
        mode_id_list_strand = list(range(start_mode, end_mode+1))
        mode_id_list_molecule = self.d_strand[strandid][start_mode:end_mode+1]
        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=figsize)
        for idx, mode_id in enumerate(mode_id_list_molecule):
            mode_id_strand = mode_id_list_strand[idx]
            self.plot_lambda_qTAq_respective_atoms_one_mode(axes[idx], strandid, mode_id, mode_id_strand, bbox_to_anchor, ylim, assist_lines)
        return fig, axes

    def plot_lambda_qTAq_respective_atoms_five_modes_by_resnames(self, figsize, strandid, resname_i, resname_j, start_mode, end_mode, bbox_to_anchor, ylim=None, assist_lines=None):
        mode_id_list_strand = list(range(start_mode, end_mode+1))
        mode_id_list_molecule = self.d_strand[strandid][start_mode:end_mode+1]
        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=figsize)
        for idx, mode_id in enumerate(mode_id_list_molecule):
            mode_id_strand = mode_id_list_strand[idx]
            self.plot_lambda_qTAq_respective_atoms_one_mode_by_resnames(axes[idx], strandid, resname_i, resname_j, mode_id, mode_id_strand, bbox_to_anchor, ylim, assist_lines)
        return fig, axes

    def plot_lambda_qTAq_respective_atoms_one_mode(self, ax, strandid, mode_id, mode_id_strand, bbox_to_anchor, ylim, assist_lines):
        """
        strandid: 'STRAND1', 'STRAND2'
        """
        resname = self.d_host_strand[self.host][strandid]
        atomlist = self.d_atomlist[resname]
        d_atomlist = self.get_d_atomlist(atomlist)

        w_small = 0.5
        w_big = 0.8
        d_xarray = self.get_d_xarray(atomlist, d_atomlist, w_small, w_big)
        d_result = self.get_d_result(atomlist, d_atomlist, strandid, mode_id)
        xticks, xticklabels = self.get_xticks_xticklabels(atomlist, d_xarray, d_atomlist)
        for atomname in atomlist:
            ax.bar(d_xarray[atomname], d_result[atomname], w_small, label=atomname, edgecolor='white', color=self.d_color[atomname])
        ax.set_ylabel(self.get_ylabel(mode_id_strand))
        ax.set_xlabel('Mode id, $i$')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.legend(ncol=1, loc='center right', bbox_to_anchor=bbox_to_anchor)
        ax.set_title(self.get_title(strandid))
        if ylim is not None:
            ax.set_ylim(ylim)
        if assist_lines is not None:
            for yvalue in assist_lines:
                ax.axhline(yvalue, color='grey', alpha=0.2)

    def plot_lambda_qTAq_respective_atoms_one_mode_by_resnames(self, ax, strandid, resname_i, resname_j, mode_id, mode_id_strand, bbox_to_anchor, ylim, assist_lines):
        """
        strandid: 'STRAND1', 'STRAND2'
        """
        atomlist_i = self.d_atomlist[resname_i]
        atomlist_j = self.d_atomlist[resname_j]
        d_atomlist = self.get_d_atomlist_by_atomlist_ij(atomlist_i, atomlist_j)

        w_small = 0.5
        w_big = 0.8
        d_xarray = self.get_d_xarray(atomlist_i, d_atomlist, w_small, w_big)
        d_result = self.get_d_result_by_resnames(atomlist_i, d_atomlist, strandid, resname_i, resname_j, mode_id)
        xticks, xticklabels = self.get_xticks_xticklabels(atomlist_i, d_xarray, d_atomlist)
        for atomname in atomlist_i:
            label = f'{resname_i}: {atomname}'
            ax.bar(d_xarray[atomname], d_result[atomname], w_small, label=label, edgecolor='white', color=self.d_color[atomname])
        ax.set_ylabel(self.get_ylabel(mode_id_strand))
        ax.set_xlabel('Mode id, $i$')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.legend(ncol=1, loc='center right', bbox_to_anchor=bbox_to_anchor)
        ax.set_title(self.get_title_by_resname_i_resname_j(resname_i, resname_j))
        if ylim is not None:
            ax.set_ylim(ylim)
        if assist_lines is not None:
            for yvalue in assist_lines:
                ax.axhline(yvalue, color='grey', alpha=0.2)

    def get_ylabel(self, mode_id):
        return r'Decomposed $\lambda_{' + f'{mode_id}' + r'}$ (kcal/mol/Å$^2$)'

    def get_d_atomlist(self, atomlist):
        d_atomlist = dict()
        atomlist_fordelete = [atomname for atomname in atomlist]
        for atomname1 in atomlist:
            d_atomlist[atomname1] = [atomname2 for atomname2 in atomlist_fordelete]
            atomlist_fordelete.remove(atomname1)
        return d_atomlist

    def get_d_atomlist_by_atomlist_ij(self, atomlist_i, atomlist_j):
        d_atomlist = dict()
        for atomname1 in atomlist_i:
            d_atomlist[atomname1] = [atomname2 for atomname2 in atomlist_j]
        return d_atomlist

    def get_d_xarray(self, atomlist, d_atomlist, w_small, w_big):
        d_xarray = dict()
        x = 0.
        for atomname1 in atomlist:
            n_atom2 = len(d_atomlist[atomname1])
            d_xarray[atomname1] = np.zeros(n_atom2)
            for idx in range(n_atom2):
                d_xarray[atomname1][idx] = x
                x += w_small
            x += w_big
        return d_xarray

    def get_xticks_xticklabels(self, atomlist, d_xarray, d_atomlist):
        xticks = list()
        xticklabels = list()
        for atomname1 in atomlist:
            xticks += list(d_xarray[atomname1])
            xticklabels += d_atomlist[atomname1]
        return xticks, xticklabels

    def get_d_result(self, atomlist, d_atomlist, strandid, mode_id):
        d_result = dict()
        for atomname1 in atomlist:
            d_result[atomname1] = np.zeros(len(d_atomlist[atomname1]))
            for idx, atomname2 in enumerate(d_atomlist[atomname1]):
                q = self.g_agent.get_eigenvector_by_id(mode_id)
                A_mat = self.g_agent.get_A_by_atomname1_atomname2(atomname1, atomname2, strandid)
                qTAq = np.dot(q.T, np.dot(A_mat, q))
                d_result[atomname1][idx] = qTAq
        return d_result

    def get_d_result_by_resnames(self, atomlist, d_atomlist, strandid, resname_i, resname_j, mode_id):
        d_result = dict()
        for atomname1 in atomlist:
            d_result[atomname1] = np.zeros(len(d_atomlist[atomname1]))
            for idx, atomname2 in enumerate(d_atomlist[atomname1]):
                q = self.g_agent.get_eigenvector_by_id(mode_id)
                A_mat = self.g_agent.get_A_by_atomname1_atomname2_by_resnames(atomname1, atomname2, resname_i, resname_j, strandid)
                qTAq = np.dot(q.T, np.dot(A_mat, q))
                d_result[atomname1][idx] = qTAq
        return d_result

