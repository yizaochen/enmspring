from os import path
import numpy as np
import matplotlib.pyplot as plt
from enmspring.graphs import Stack

class StackPlot:
    d_atomlist = {'A': ['N1', 'C6', 'C5', 'C4', 'N3', 'C2', 'N6', 'N7', 'C8', 'N9'],
                  'T': ['C4', 'C5', 'C6', 'N1', 'C2', 'N3', 'C7', 'O2', 'O4'],
                  'C': ['C4', 'C5', 'C6', 'N1', 'C2', 'N3', 'O2', 'N4'],
                  'G': ['N1', 'C6', 'C5', 'C4', 'N3', 'C2', 'O6', 'N2', 'N7', 'C8', 'N9']}
    d_host_strand = {'a_tract_21mer': ('A', 'T'),
                     'g_tract_21mer': ('G', 'C')}

    def __init__(self, host, rootfolder):
        self.host = host
        self.rootfolder = rootfolder

        self.g_agent = Stack(host, rootfolder)
        self.process_g_agent()

    def process_g_agent(self):
        self.g_agent.build_node_list()
        self.g_agent.initialize_three_mat()
        self.g_agent.build_adjacency_from_df_st()
        self.g_agent.build_degree_from_adjacency()
        self.g_agent.build_laplacian_by_adjacency_degree()
        self.g_agent.eigen_decompose()
        self.g_agent.set_benchmark_array()
        self.g_agent.set_strand_array()

    def plot_a_tract_aa(self, figsize, start_mode, end_mode):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        mode_list = list(range(start_mode, end_mode+1))
        strandid = 'STRAND1'
        assit_y = [0.2, 0.4, 0.6]

        atomlist = ['N1', 'N6', 'C6']
        ax = axes[0]
        for atomname in atomlist:
            dotprod_list = self.get_dotproduct_list(atomname, strandid, start_mode, end_mode)
            ax.plot(mode_list, dotprod_list, label=atomname)
            self.draw_assit_lines(ax, assit_y)
            self.set_xylabel_legend(ax)

        atomlist = ['C2', 'N3', 'C4', 'C5', 'N7']
        ax = axes[1]
        for atomname in atomlist:
            dotprod_list = self.get_dotproduct_list(atomname, strandid, start_mode, end_mode)
            ax.plot(mode_list, dotprod_list, label=atomname)
            self.draw_assit_lines(ax, assit_y)
            self.set_xylabel_legend(ax)

        atomlist = ['C8', 'N9']
        ax = axes[2]
        for atomname in atomlist:
            dotprod_list = self.get_dotproduct_list(atomname, strandid, start_mode, end_mode)
            ax.plot(mode_list, dotprod_list, label=atomname)
            self.draw_assit_lines(ax, assit_y)
            self.set_xylabel_legend(ax)
        return fig, axes

    def plot_strand1_strand2(self, figsize, start_mode, end_mode):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        mode_list = list(range(start_mode, end_mode+1))

        ax = axes[0]
        strandid = 'STRAND1'
        resname = self.d_host_strand[self.host][0]
        for atomname in self.d_atomlist[resname]:
            dotprod_list = self.get_dotproduct_list(atomname, strandid, start_mode, end_mode)
            ax.plot(mode_list, dotprod_list, label=atomname)
            self.set_xylabel_legend(ax)

        ax = axes[1]
        strandid = 'STRAND2'
        resname = self.d_host_strand[self.host][1]
        for atomname in self.d_atomlist[resname]:
            dotprod_list = self.get_dotproduct_list(atomname, strandid, start_mode, end_mode)
            ax.plot(mode_list, dotprod_list, label=atomname)
            self.set_xylabel_legend(ax)
        return fig, axes

    def draw_assit_lines(self, ax, ylist):
        for yvalue in ylist:
            ax.axhline(yvalue, color='grey', linestyle='--', alpha=0.2)

    def set_xylabel_legend(self, ax):
        ax.set_xlabel(r'Eigenvector Index, $i$')
        ax.set_ylabel(r'$\mathbf{e}_i \cdot \vec{v}_{\mathrm{filter}}$')
        ax.legend(frameon=False)

    def get_dotproduct_list(self, atomname, strandid, start_mode, end_mode):
        d_eigve_id_list = {'STRAND1': self.g_agent.strand1_array, 'STRAND2': self.g_agent.strand2_array}
        filter_array = self.g_agent.get_filter_by_atomname_strandid(atomname, strandid)
        eigve_id_list = d_eigve_id_list[strandid][start_mode-1:end_mode]
        dotprod_list = np.zeros(len(eigve_id_list))
        for idx, mode_id in enumerate(eigve_id_list):
            eigv_sele = np.abs(self.g_agent.get_eigenvector_by_id(mode_id))
            dotprod_list[idx] = np.dot(eigv_sele, filter_array)
        return dotprod_list