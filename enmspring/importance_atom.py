from os import path
import numpy as np
import matplotlib.pyplot as plt
from enmspring.graphs import Stack

pic_out_folder = '/home/yizaochen/Documents/JPCL_ytc_2021/images/atom_importance'
class StackPlot:
    d_atomlist = {'A': ['N1', 'C6', 'C5', 'C4', 'N3', 'C2', 'N6', 'N7', 'C8', 'N9'],
                  'T': ['C4', 'C5', 'C6', 'N1', 'C2', 'N3', 'C7', 'O2', 'O4'],
                  'C': ['C4', 'C5', 'C6', 'N1', 'C2', 'N3', 'O2', 'N4'],
                  'G': ['N1', 'C6', 'C5', 'C4', 'N3', 'C2', 'O6', 'N2', 'N7', 'C8', 'N9']}

    d_host_strand = {'a_tract_21mer': ('A', 'T'),
                     'g_tract_21mer': ('G', 'C'),
                     'atat_21mer': ('A', 'T'),
                     'gcgc_21mer': ('G', 'C'),
                     'ctct_21mer': {'STRAND1': ('C', 'T'), 'STRAND2': ('G', 'A')},
                     'tgtg_21mer': {'STRAND1': ('T', 'G'), 'STRAND2': ('A', 'C')},
                     }

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

    def vmd_open_perfect_gro(self):
        aa_folder = path.join('/home/yizaochen/codes/dna_rna/all_systems', self.host, 'bdna+bdna', 'input', 'allatoms')
        perferct_gro = path.join(aa_folder, 'bdna+bdna.perfect.gro')
        print(f'vmd -gro {perferct_gro}')

    def vmd_show_atom_importance(self, resid):
        self.vmd_open_perfect_gro()
        print('The following in tk console:')
        print('mol delrep 0 0')
        self.vmd_add_resid(resid)

    def vmd_add_resid(self, resid):
        lines = ['mol color ColorID 2',
                 'mol representation Licorice 0.100000 12.000000 12.000000',
                 f'mol selection resid {resid} and not hydrogen and not (name C1\' C2\' O4\' C3\' C4\' C5\' P O1P O2P O5\' O3\')',
                 'mol material Opaque',
                 'mol addrep 0']
        return lines

    def vmd_add_atomlist_vdw(self, atomlist, resid, colorid, cpkradius):
        atomnames = ' '.join(atomlist)
        lines = [f'mol color ColorID {colorid}',
                 f'mol representation CPK {cpkradius:.3f} 0.200000 12.000000 12.000000',
                 f'mol selection resid {resid} and name {atomnames}',
                  'mol material Opaque',
                  'mol addrep 0']
        return lines

    def vmd_show_fourmer(self, resid, bigatomlist):
        colorid_list = [0, 1, 5]
        cpkradius_list = [1.2, 0.9, 0.5]
        lines = list()
        for atomlist, colorid, cpkradius in zip(bigatomlist, colorid_list, cpkradius_list):
            lines += self.vmd_add_atomlist_vdw(atomlist, resid, colorid, cpkradius)
        return lines

    def vmd_add_transparent(self):
        lines = ['mol color ColorID 6',
                 'mol representation Licorice 0.100000 12.000000 12.000000',
                 'mol selection all and not hydrogen',
                 'mol material Transparent',
                 'mol addrep 0']
        return lines

    def print_tga_out(self, out_name):
        print(path.join(pic_out_folder, 'mol_structure', out_name))

    def write_tcl_out(self, tcl_out, container):
        f = open(tcl_out, 'w')
        for line in container:
            f.write(line)
            f.write('\n')
        f.close()
        print(f'source {tcl_out}')

    def vmd_show_a_tract_aaaa(self):
        out_name = 'aaaa'
        resid_list = [3, 4, 5, 6]
        bigatomlist = [['N1', 'N6', 'C6'], ['C2', 'N3', 'C4', 'C5', 'N7'], ['C8', 'N9']]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        for resid in resid_list:
            lines += self.vmd_add_resid(resid)
            lines += self.vmd_show_fourmer(resid, bigatomlist)
        lines += self.vmd_add_transparent()
        self.write_tcl_out('../tclscripts/draw_fourmer.tcl', lines)
        self.print_tga_out(f'{self.host}_{out_name}_mol_fourmer')

    def vmd_show_a_tract_tttt(self):
        out_name = 'tttt'
        resid_list = [24, 25, 26, 27]
        bigatomlist = [['N1', 'C2', 'N3', 'C4', 'C5'], ['C6', 'C7'], ['O2', 'O4']]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        for resid in resid_list:
            lines += self.vmd_add_resid(resid)
            lines += self.vmd_show_fourmer(resid, bigatomlist)
        lines += self.vmd_add_transparent()
        self.write_tcl_out('../tclscripts/draw_fourmer.tcl', lines)
        self.print_tga_out(f'{self.host}_{out_name}_mol_fourmer')

    def a_tract_aa(self, figsize, start_mode, end_mode):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        assit_y = [0.2, 0.4, 0.6]
        bigatomlist = [['N1', 'N6', 'C6'], ['C2', 'N3', 'C4', 'C5', 'N7'], ['C8', 'N9']]
        for idx, atomlist in enumerate(bigatomlist):
            self.plot_RR_YY_by_atomlist_with_assit_y(axes[idx], atomlist, 'STRAND1', start_mode, end_mode, assit_y)
        return fig, axes

    def a_tract_tt(self, figsize, start_mode, end_mode):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharey=True)
        assit_y = [0.1, 0.2, 0.3, 0.4]
        bigatomlist = [['N1', 'C2', 'N3', 'C4', 'C5'], ['C6', 'C7'], ['O2', 'O4']]
        for idx, atomlist in enumerate(bigatomlist):
            self.plot_RR_YY_by_atomlist_with_assit_y(axes[idx], atomlist, 'STRAND2', start_mode, end_mode, assit_y)
        return fig, axes

    def vmd_show_g_tract_gggg(self):
        out_name = 'gggg'
        resid_list = [3, 4, 5, 6]
        bigatomlist = [['N1', 'C6', 'O6'], ['C2', 'N2', 'N3', 'C4', 'C5', 'N7'], ['C8', 'N9']]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        for resid in resid_list:
            lines += self.vmd_add_resid(resid)
            lines += self.vmd_show_fourmer(resid, bigatomlist)
        lines += self.vmd_add_transparent()
        self.write_tcl_out('../tclscripts/draw_fourmer.tcl', lines)
        self.print_tga_out(f'{self.host}_{out_name}_mol_fourmer')

    def vmd_show_g_tract_cccc(self):
        out_name = 'cccc'
        resid_list = [24, 25, 26, 27]
        bigatomlist =[['C2', 'C5', 'C4'], ['N3', 'N4'], ['N1', 'O2', 'C6']]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        for resid in resid_list:
            lines += self.vmd_add_resid(resid)
            lines += self.vmd_show_fourmer(resid, bigatomlist)
        lines += self.vmd_add_transparent()
        self.write_tcl_out('../tclscripts/draw_fourmer.tcl', lines)
        self.print_tga_out(f'{self.host}_{out_name}_mol_fourmer')

    def g_tract_gg(self, figsize, start_mode, end_mode):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharey=True)
        assit_y = np.arange(0.0, 0.7, 0.1)
        bigatomlist = [['N1', 'C6', 'O6'], ['C2', 'N2', 'N3', 'C4', 'C5', 'N7'], ['C8', 'N9']]
        for idx, atomlist in enumerate(bigatomlist):
            self.plot_RR_YY_by_atomlist_with_assit_y(axes[idx], atomlist, 'STRAND1', start_mode, end_mode, assit_y)
        return fig, axes

    def g_tract_cc(self, figsize, start_mode, end_mode):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharey=True)
        assit_y = [0.1, 0.2, 0.3, 0.4, 0.5]
        bigatomlist = [['C2', 'C5', 'C4'], ['N3', 'N4'], ['N1', 'O2', 'C6']]
        for idx, atomlist in enumerate(bigatomlist):
            self.plot_RR_YY_by_atomlist_with_assit_y(axes[idx], atomlist, 'STRAND2', start_mode, end_mode, assit_y)
        return fig, axes

    def vmd_show_atat(self):
        out_name = 'atat'
        resid_list = [3, 4, 5, 6]
        d_bigatomlist = {'A': [['C4', 'C5', 'C6'], ['N1', 'C2', 'N3'], ['N6', 'N7', 'C8', 'N9']],
                         'T': [['C4', 'C5'], ['C2', 'N3'], ['N1', 'O4', 'O2', 'C6', 'C7']]}
        bigatomlist_list = [d_bigatomlist['A'], d_bigatomlist['T'], d_bigatomlist['A'], d_bigatomlist['T']]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        for resid, bigatomlist in zip(resid_list, bigatomlist_list):
            lines += self.vmd_add_resid(resid)
            lines += self.vmd_show_fourmer(resid, bigatomlist)
        lines += self.vmd_add_transparent()
        self.write_tcl_out('../tclscripts/draw_fourmer.tcl', lines)
        self.print_tga_out(f'{self.host}_{out_name}_mol_fourmer')

    def atat(self, figsize, start_mode, end_mode, strandid='STRAND1'):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize, sharey=True)
        assit_y = [0.1, 0.2, 0.3]
        d_bigatomlist = {'A': [['C4', 'C5', 'C6'], ['N1', 'C2', 'N3'], ['N6', 'N7', 'C8', 'N9']],
                         'T': [['C4', 'C5'], ['C2', 'N3'], ['N1', 'O4', 'O2', 'C6', 'C7']]}
        for row_id, resname in enumerate(['A', 'T']):
            for idx, atomlist in enumerate(d_bigatomlist[resname]):
                self.plot_YR_by_resname_atomlist_with_assity(axes[row_id, idx], atomlist, strandid, resname, start_mode, end_mode, assit_y)
        return fig, axes

    def vmd_show_gcgc(self):
        out_name = 'gcgc'
        resid_list = [3, 4, 5, 6]
        d_bigatomlist = {'G': [['C4', 'C5'], ['N1', 'C2', 'N3', 'C6'], ['N2', 'O6', 'N7', 'C8', 'N9']],
                         'C': [['C2', 'N3', 'C4'], ['O2', 'N4', 'C5'], ['N1', 'C6']]}
        bigatomlist_list = [d_bigatomlist['G'], d_bigatomlist['C'], d_bigatomlist['G'], d_bigatomlist['C']]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        for resid, bigatomlist in zip(resid_list, bigatomlist_list):
            lines += self.vmd_add_resid(resid)
            lines += self.vmd_show_fourmer(resid, bigatomlist)
        lines += self.vmd_add_transparent()
        self.write_tcl_out('../tclscripts/draw_fourmer.tcl', lines)
        self.print_tga_out(f'{self.host}_{out_name}_mol_fourmer')

    def gcgc(self, figsize, start_mode, end_mode, strandid='STRAND1'):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize, sharey=True)
        assit_y = [0.1, 0.2, 0.3, 0.4]
        d_bigatomlist = {'G': [['C4', 'C5'], ['N1', 'C2', 'N3', 'C6'], ['N2', 'O6', 'N7', 'C8', 'N9']],
                         'C': [['C2', 'N3', 'C4'], ['O2', 'N4', 'C5'], ['N1', 'C6']]}
        for row_id, resname in enumerate(['G', 'C']):
            for idx, atomlist in enumerate(d_bigatomlist[resname]):
                self.plot_YR_by_resname_atomlist_with_assity(axes[row_id, idx], atomlist, strandid, resname, start_mode, end_mode, assit_y)
        return fig, axes

    def vmd_show_ctct(self):
        out_name = 'ctct'
        resid_list = [3, 4, 5, 6]
        d_bigatomlist = {'C': [['N1', 'C2', 'N3'], ['C4', 'N4'], ['C5', 'C6', 'O2']],
                         'T': [['C4', 'C5'], ['C2', 'O2', 'N3', 'C7'], ['N1', 'O4', 'C6']]}
        bigatomlist_list = [d_bigatomlist['C'], d_bigatomlist['T'], d_bigatomlist['C'], d_bigatomlist['T']]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        for resid, bigatomlist in zip(resid_list, bigatomlist_list):
            lines += self.vmd_add_resid(resid)
            lines += self.vmd_show_fourmer(resid, bigatomlist)
        lines += self.vmd_add_transparent()
        self.write_tcl_out('../tclscripts/draw_fourmer.tcl', lines)
        self.print_tga_out(f'{self.host}_{out_name}_mol_fourmer')

    def vmd_show_gaga(self):
        out_name = 'gaga'
        resid_list = [24, 25, 26, 27]
        d_bigatomlist = {'G': [['N1', 'C5', 'C6'], ['C2', 'N3', 'C4','O6'], ['N2', 'N7', 'C8', 'N9']],
                         'A': [['C6'], ['N1', 'C4', 'C5', 'N6'], ['C2', 'N3', 'N7', 'C8', 'N9']]}
        bigatomlist_list = [d_bigatomlist['G'], d_bigatomlist['A'], d_bigatomlist['G'], d_bigatomlist['A']]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        for resid, bigatomlist in zip(resid_list, bigatomlist_list):
            lines += self.vmd_add_resid(resid)
            lines += self.vmd_show_fourmer(resid, bigatomlist)
        lines += self.vmd_add_transparent()
        self.write_tcl_out('../tclscripts/draw_fourmer.tcl', lines)
        self.print_tga_out(f'{self.host}_{out_name}_mol_fourmer')

    def ctct_ct(self, figsize, start_mode, end_mode, strandid='STRAND1'):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize, sharey=True)
        assit_y = [0.1, 0.2, 0.3, 0.4]
        d_bigatomlist = {'C': [['N1', 'C2', 'N3'], ['C4', 'N4'], ['C5', 'C6', 'O2']],
                         'T': [['C4', 'C5'], ['C2', 'O2', 'N3', 'C7'], ['N1', 'O4', 'C6']]}
        for row_id, resname in enumerate(['C', 'T']):
            for idx, atomlist in enumerate(d_bigatomlist[resname]):
                self.plot_YR_by_resname_atomlist_with_assity(axes[row_id, idx], atomlist, strandid, resname, start_mode, end_mode, assit_y)
        return fig, axes

    def ctct_ga(self, figsize, start_mode, end_mode, strandid='STRAND2'):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize, sharey=True)
        assit_y = [0.1, 0.2, 0.3, 0.4, 0.5]
        d_bigatomlist = {'G': [['N1', 'C5', 'C6'], ['C2', 'N3', 'C4','O6'], ['N2', 'N7', 'C8', 'N9']],
                         'A': [['C6'], ['N1', 'C4', 'C5', 'N6'], ['C2', 'N3', 'N7', 'C8', 'N9']]}
        for row_id, resname in enumerate(['G', 'A']):
            for idx, atomlist in enumerate(d_bigatomlist[resname]):
                self.plot_YR_by_resname_atomlist_with_assity(axes[row_id, idx], atomlist, strandid, resname, start_mode, end_mode, assit_y)
        return fig, axes

    def vmd_show_tgtg(self):
        out_name = 'tgtg'
        resid_list = [3, 4, 5, 6]
        d_bigatomlist = {'T': [['N3', 'C4', 'C5'], ['C2'], ['N1', 'O2', 'O4', 'C6', 'C7']],
                         'G': [['C4'], ['N1', 'C2', 'N2', 'N3', 'C5', 'C6', 'N9'], ['N7', 'O6', 'C8']]}
        bigatomlist_list = [d_bigatomlist['T'], d_bigatomlist['G'], d_bigatomlist['T'], d_bigatomlist['G']]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        for resid, bigatomlist in zip(resid_list, bigatomlist_list):
            lines += self.vmd_add_resid(resid)
            lines += self.vmd_show_fourmer(resid, bigatomlist)
        lines += self.vmd_add_transparent()
        self.write_tcl_out('../tclscripts/draw_fourmer.tcl', lines)
        self.print_tga_out(f'{self.host}_{out_name}_mol_fourmer')

    def vmd_show_acac(self):
        out_name = 'acac'
        resid_list = [24, 25, 26, 27]
        d_bigatomlist = {'A': [['C4', 'C5'], ['N1', 'C2', 'N3', 'N6', 'C6', 'N7'], ['C8', 'N9']],
                         'C': [['C2', 'N3', 'C4'], ['N4', 'C5'], ['N1', 'O2', 'C6']]}
        bigatomlist_list = [d_bigatomlist['A'], d_bigatomlist['C'], d_bigatomlist['A'], d_bigatomlist['C']]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        for resid, bigatomlist in zip(resid_list, bigatomlist_list):
            lines += self.vmd_add_resid(resid)
            lines += self.vmd_show_fourmer(resid, bigatomlist)
        lines += self.vmd_add_transparent()
        self.write_tcl_out('../tclscripts/draw_fourmer.tcl', lines)
        self.print_tga_out(f'{self.host}_{out_name}_mol_fourmer')

    def tgtg_tg(self, figsize, start_mode, end_mode, strandid='STRAND1'):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize, sharey=True)
        assit_y = np.arange(0.05, 0.36, 0.05)
        d_bigatomlist = {'T': [['N3', 'C4', 'C5'], ['C2'], ['N1', 'O2', 'O4', 'C6', 'C7']],
                         'G': [['C4'], ['N1', 'C2', 'N2', 'N3', 'C5', 'C6', 'N9'], ['N7', 'O6', 'C8']]}
        for row_id, resname in enumerate(['T', 'G']):
            for idx, atomlist in enumerate(d_bigatomlist[resname]):
                self.plot_YR_by_resname_atomlist_with_assity(axes[row_id, idx], atomlist, strandid, resname, start_mode, end_mode, assit_y)
        return fig, axes

    def tgtg_ac(self, figsize, start_mode, end_mode, strandid='STRAND2'):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize, sharey=True)
        assit_y = [0.1, 0.2, 0.3, 0.4]
        d_bigatomlist = {'A': [['C4', 'C5'], ['N1', 'C2', 'N3', 'N6', 'C6', 'N7'], ['C8', 'N9']],
                         'C': [['C2', 'N3', 'C4'], ['N4', 'C5'], ['N1', 'O2', 'C6']]}
        for row_id, resname in enumerate(['A', 'C']):
            for idx, atomlist in enumerate(d_bigatomlist[resname]):
                self.plot_YR_by_resname_atomlist_with_assity(axes[row_id, idx], atomlist, strandid, resname, start_mode, end_mode, assit_y)
        return fig, axes

    def overview_RR_YY_homo(self, figsize, start_mode, end_mode):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        for idx, strandid in enumerate(['STRAND1', 'STRAND2']):
            resname = self.d_host_strand[self.host][idx]
            atomlist = self.d_atomlist[resname]
            self.plot_RR_YY_by_atomlist(axes[idx], atomlist, strandid, start_mode, end_mode)
        return fig, axes

    def overview_hetero(self, figsize, start_mode, end_mode):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        for row_id, strandid in enumerate(['STRAND1', 'STRAND2']):
            for col_id, resname in enumerate(self.d_host_strand[self.host][strandid]):
                atomlist = self.d_atomlist[resname]
                self.plot_YR_by_resname_atomlist(axes[row_id, col_id], atomlist, strandid, resname, start_mode, end_mode)
        return fig, axes

    def overview_YR(self, figsize, start_mode, end_mode):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        for row_id, strandid in enumerate(['STRAND1', 'STRAND2']):
            for col_id, resname in enumerate(self.d_host_strand[self.host]):
                atomlist = self.d_atomlist[resname]
                self.plot_YR_by_resname_atomlist(axes[row_id, col_id], atomlist, strandid, resname, start_mode, end_mode)
        return fig, axes

    def plot_YR_by_resname_atomlist_with_assity(self, ax, atomlist, strandid, resname, start_mode, end_mode, assit_y):
        mode_list = list(range(start_mode, end_mode+1))
        for atomname in atomlist:
            dotprod_list = self.get_dotproduct_list_YR(atomname, strandid, resname, start_mode, end_mode)
            ax.plot(mode_list, dotprod_list, label=atomname)
            self.draw_assit_lines(ax, assit_y)
            self.set_xylabel_legend(ax)

    def plot_YR_by_resname_atomlist(self, ax, atomlist, strandid, resname, start_mode, end_mode):
        mode_list = list(range(start_mode, end_mode+1))
        for atomname in atomlist:
            dotprod_list = self.get_dotproduct_list_YR(atomname, strandid, resname, start_mode, end_mode)
            ax.plot(mode_list, dotprod_list, label=atomname)
            self.set_xylabel_legend(ax)
        title = f'{strandid}-{resname}'
        ax.set_title(title)

    def plot_RR_YY_by_atomlist(self, ax, atomlist, strandid, start_mode, end_mode):
        mode_list = list(range(start_mode, end_mode+1))
        for atomname in atomlist:
            dotprod_list = self.get_dotproduct_list(atomname, strandid, start_mode, end_mode)
            ax.plot(mode_list, dotprod_list, label=atomname)
            self.set_xylabel_legend(ax)

    def plot_RR_YY_by_atomlist_with_assit_y(self, ax, atomlist, strandid, start_mode, end_mode, assit_y):
        mode_list = list(range(start_mode, end_mode+1))
        for atomname in atomlist:
            dotprod_list = self.get_dotproduct_list(atomname, strandid, start_mode, end_mode)
            ax.plot(mode_list, dotprod_list, label=atomname)
            self.draw_assit_lines(ax, assit_y)
            self.set_xylabel_legend(ax)

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

    def get_dotproduct_list_YR(self, atomname, strandid, resname, start_mode, end_mode):
        d_eigve_id_list = {'STRAND1': self.g_agent.strand1_array, 'STRAND2': self.g_agent.strand2_array}
        filter_array = self.g_agent.get_filter_by_atomname_for_YR(atomname, resname, strandid)
        eigve_id_list = d_eigve_id_list[strandid][start_mode-1:end_mode]
        dotprod_list = np.zeros(len(eigve_id_list))
        for idx, mode_id in enumerate(eigve_id_list):
            eigv_sele = np.abs(self.g_agent.get_eigenvector_by_id(mode_id))
            dotprod_list[idx] = np.dot(eigv_sele, filter_array)
        return dotprod_list