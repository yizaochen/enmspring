import matplotlib.pyplot as plt
import numpy as np
from enmspring.na_seq import sequences

class AtomSeparatePlot:
    n_bp = 21
    d_ncol = {'A': 4, 'T': 3, 'C': 2, 'G': 5}
    d_atomlist = {'A': ['N1', 'C6', 'C5', 'C4', 'N3', 'C2', 'N6', 'N7', 'C8', 'N9'],
                  'T': ['C4', 'C5', 'C6', 'N1', 'C2', 'N3', 'C7', 'O2', 'O4'],
                  'C': ['C4', 'C5', 'C6', 'N1', 'C2', 'N3', 'O2', 'N4'],
                  'G': ['N1', 'C6', 'C5', 'C4', 'N3', 'C2', 'O6', 'N2', 'N7', 'C8', 'N9']}
    lbfz = 14
    ttfz = 16

    def __init__(self, host, base_type, figsize):
        self.host = host
        self.base_type = base_type # 'A', 'T', 'C', 'G'
        self.figsize = figsize

        self.atom_list = self.d_atomlist[self.base_type]
        self.n_atom = len(self.atom_list)

        self.w, self.xlist = self.get_width_xlist()
        self.xticks = range(3,22,3)
        
    def plot_eigenvector_by_eigv_id(self, eigv_id, s_agent, b_agent, ylim=None):
        fig, axes = plt.subplots(nrows=2, ncols=6, figsize=self.figsize, facecolor='white')
        d_axes = self.get_d_axes(axes)
        for atomname in self.atom_list:
            self.barplot_single_atom(d_axes[atomname], atomname, eigv_id, s_agent, b_agent)
        if ylim is not None:
            self.set_all_ylims(d_axes, ylim)
        self.set_all_xlabel(axes)
        self.set_all_ylabel(axes, eigv_id)
        self.set_all_titles(d_axes)
        self.add_fraying_lines(d_axes)
        self.set_all_xlims(d_axes)
        self.remove_axes(axes)
        self.print_ylim(d_axes)
        return fig, axes

    def barplot_single_atom(self, ax, atomname, eigv_id, s_agent, b_agent):
        yarray = self.get_yarray(eigv_id, atomname, s_agent, b_agent)
        ax.bar(self.xlist, yarray, width=self.w, edgecolor='white')
        ax.set_xticks(self.xticks)

    def get_yarray(self, eigv_id, atomname, s_agent, b_agent):
        yarray = np.zeros(self.n_bp)
        eigenvector = s_agent.get_eigenvector_by_id(eigv_id) # eigenvector
        d_idx = b_agent.get_d_idx()
        for resid in b_agent.resid_list:
            idx = d_idx[atomname][resid]
            if idx is not None:
                yarray[resid-1] = eigenvector[idx]
        return yarray

    def get_width_xlist(self):
        w = 1
        xlist = list(range(1, self.n_bp+1))
        return w, xlist

    def get_d_axes(self, axes):
        d_axes = dict()
        atom_idx = 0
        
        row_id = 0
        for col_id in range(6):
            atomname = self.atom_list[atom_idx]
            d_axes[atomname] = axes[row_id, col_id]
            atom_idx += 1

        row_id = 1
        for col_id in range(self.d_ncol[self.base_type]):
            atomname = self.atom_list[atom_idx]
            d_axes[atomname] = axes[row_id, col_id]
            atom_idx += 1
        return d_axes

    def remove_axes(self, axes):
        d_remove_list = {'A': [(1, 4), (1, 5)], 'T': [(1, 3), (1, 4), (1, 5)], 
                         'C': [(1, 2), (1, 3), (1, 4), (1, 5)], 'G': [(1, 5)]}
        for row_id, col_id in d_remove_list[self.base_type]:
            axes[row_id, col_id].remove()

    def set_all_titles(self, d_axes):
        for atomname in self.atom_list:
            d_axes[atomname].set_title(f'{atomname}', fontsize=self.ttfz)

    def set_all_ylims(self, d_axes, ylim):
        for atomname in self.atom_list:
            d_axes[atomname].set_ylim(ylim)
            
    def set_all_xlims(self, d_axes):
        for atomname in self.atom_list:
            d_axes[atomname].set_xlim(0.5,21.5)

    def print_ylim(self, d_axes):
        ymin_list = list()
        ymax_list = list()
        for atomname in self.atom_list:
            ymin, ymax = d_axes[atomname].get_ylim()
            ymin_list.append(ymin)
            ymax_list.append(ymax)
        ymin = min(ymin_list)
        ymax = max(ymax_list)
        print(f'({ymin:.3f}, {ymax:.3f})')
        

    def add_fraying_lines(self, d_axes):
        for atomname in self.atom_list:
            d_axes[atomname].axvline(3.5, color='red', linestyle='--', alpha=0.2)
            d_axes[atomname].axvline(17.5, color='red', linestyle='--', alpha=0.2)
            
    def set_all_ylabel(self, axes, eigv_id):
        ylabel = r'$\mathbf{e}_{' + f'{eigv_id}' + r'}$'
        axes[0,0].set_ylabel(ylabel, fontsize=self.lbfz)
        axes[1,0].set_ylabel(ylabel, fontsize=self.lbfz)

    def set_all_xlabel(self, axes):
        xlabel = 'Resid'
        row_id = 1
        for col_id in range(self.d_ncol[self.base_type]):
            axes[row_id, col_id].set_xlabel(xlabel, fontsize=self.lbfz)


class BaseTypeEigenvector:

    def __init__(self, host, base_type, strand_id, s_agent):
        self.host = host
        self.base_type = base_type # 'A', 'T', 'C', 'G'
        self.strand_id = strand_id # 'STRAND1', 'STRAND2'

        self.atom_list = AtomSeparatePlot.d_atomlist[self.base_type]
        self.n_bp = AtomSeparatePlot.n_bp
        self.resid_list = list(range(1, self.n_bp+1))
        self.d_seq = {'STRAND1': sequences[host]['guide'], 'STRAND2': sequences[host]['target']}
        self.d_resid_resname_map = self.get_d_resid_resname_map()
        
        self.resid_map = s_agent.resid_map
        self.atomname_map = s_agent.atomname_map
        self.d_node_list_by_strand = s_agent.d_node_list_by_strand
        self.d_idx_list_by_strand = s_agent.d_idx_list_by_strand

        self.d_idx = self.set_d_idx()

    def get_d_resid_resname_map(self):
        d_resid_resname_map = dict()
        for resid in self.resid_list:
            d_resid_resname_map[resid] = self.d_seq[self.strand_id][resid-1]
        return d_resid_resname_map

    def set_d_idx(self):
        d_idx = {atom_id: {resid: None for resid in self.resid_list} for atom_id in self.atom_list}
        idx_list = self.d_idx_list_by_strand[self.strand_id]
        node_list= self.d_node_list_by_strand[self.strand_id]
        for idx, node_id in zip(idx_list, node_list):
            resid = self.resid_map[node_id]
            if self.d_resid_resname_map[resid] != self.base_type:
                continue
            atomname = self.atomname_map[node_id]
            d_idx[atomname][resid] = idx
        return d_idx

    def get_d_idx(self):
        return self.d_idx