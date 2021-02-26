from os import path
import MDAnalysis
import numpy as np
from enmspring.graphs import Stack
from enmspring.miscell import check_dir_exist_and_make
enmspring_folder = '/home/yizaochen/codes/dna_rna/enmspring'
all_folder = '/home/yizaochen/codes/dna_rna/all_systems'

class BaseStackImportanceAgent:
    type_na = 'bdna+bdna'

    def __init__(self, host, rootfolder, pic_out_folder):
        self.host = host
        self.rootfolder = rootfolder
        self.tcl_folder = path.join(enmspring_folder, 'tclscripts')
        self.pic_out_folder = pic_out_folder
        self.mol_stru_folder = path.join(self.pic_out_folder, 'mol_structure')

        self.allatom_folder = path.join(all_folder, host, self.type_na, 'input', 'allatoms')
        self.perferct_gro = path.join(self.allatom_folder, f'{self.type_na}.perfect.gro')

        self.g_agent = self.get_g_agent_and_preprocess()

        self.check_folder()

    def check_folder(self):
        for folder in [self.mol_stru_folder]:
            check_dir_exist_and_make(folder)

    def get_g_agent_and_preprocess(self):
        g_agent = Stack(self.host, self.rootfolder)
        g_agent.pre_process()
        return g_agent

    def vmd_show_pair_example(self, atomname_i, atomname_j, sele_strandid):
        lines = list()
        print(f'vmd -cor {self.g_agent.npt4_crd}')
        atomidpairs = self.g_agent.get_atomidpairs_atomname1_atomname2(atomname_i, atomname_j, sele_strandid)
        lines = self.process_lines_for_edges_tcl(lines, atomidpairs)
        tcl_out = path.join(self.tcl_folder, 'illustrate_pairimportance.tcl')
        self.write_tcl_out(tcl_out, lines)
        
    def process_lines_for_edges_tcl(self, lines, atomidpairs, radius=0.25):
        u_npt4 = MDAnalysis.Universe(self.g_agent.npt4_crd, self.g_agent.npt4_crd) 
        for atomid1, atomid2 in atomidpairs:
            line = self.__get_draw_edge_line(u_npt4.atoms.positions, atomid1-1, atomid2-1, radius)
            lines.append(line)
        return lines

    def __get_draw_edge_line(self, positions, atomid1, atomid2, radius):
        str_0 = 'graphics 0 cylinder {'
        str_1 = f'{positions[atomid1,0]:.3f} {positions[atomid1,1]:.3f} {positions[atomid1,2]:.3f}'
        str_2 = '} {'
        str_3 = f'{positions[atomid2,0]:.3f} {positions[atomid2,1]:.3f} {positions[atomid2,2]:.3f}'
        str_4 = '} '
        str_5 = f'radius {radius:.2f}\n'
        return str_0 + str_1 + str_2 + str_3 + str_4 + str_5

    def vmd_show_a_tract_single_A(self):
        resid = 7
        bigatomlist = [['C6'], ['N1'], ['C4', 'C5'], ['C2', 'N3', 'N6', 'N7', 'C8', 'N9']]
        colorid_list = [0, 0, 1, 5]
        cpkradius_list = [1.2, 0.9, 1.2, 0.5]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid(resid)
        for atomlist, colorid, cpkradius in zip(bigatomlist, colorid_list, cpkradius_list):
            lines += self.vmd_add_atomlist_vdw(atomlist, resid, colorid, cpkradius)
        tcl_out = path.join(self.tcl_folder, 'show_single_nucleotide.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_A_single')

    def vmd_show_a_tract_single_T(self):
        resid = 24
        bigatomlist = [['C5'], ['N1', 'C2', 'C4'], ['N3'], ['O2', 'O4', 'C6', 'C7']]
        colorid_list = [0, 0, 0, 5]
        cpkradius_list = [1.2, 0.9, 0.7, 0.5]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid(resid)
        for atomlist, colorid, cpkradius in zip(bigatomlist, colorid_list, cpkradius_list):
            lines += self.vmd_add_atomlist_vdw(atomlist, resid, colorid, cpkradius)
        tcl_out = path.join(self.tcl_folder, 'show_single_nucleotide.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_T_single')

    def vmd_show_atat_single_A(self):
        resid = 7
        bigatomlist = [['C4', 'C5'], ['C6'], ['C2', 'N3'], ['N1', 'C6', 'N6', 'N7', 'C8', 'N9']]
        colorid_list = [0, 0, 1, 5]
        cpkradius_list = [1.2, 0.8, 0.8, 0.5]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid(resid)
        for atomlist, colorid, cpkradius in zip(bigatomlist, colorid_list, cpkradius_list):
            lines += self.vmd_add_atomlist_vdw(atomlist, resid, colorid, cpkradius)
        tcl_out = path.join(self.tcl_folder, 'show_single_nucleotide.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_A_single')

    def vmd_show_atat_single_T(self):
        resid = 8
        bigatomlist = [['C4'], ['C5'], ['N3'], ['C2'], ['N1', 'O2', 'O4', 'C6', 'C7']]
        colorid_list = [0, 0, 1, 1, 5]
        cpkradius_list = [1.2, 0.7, 1.1, 0.7, 0.5]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid(resid)
        for atomlist, colorid, cpkradius in zip(bigatomlist, colorid_list, cpkradius_list):
            lines += self.vmd_add_atomlist_vdw(atomlist, resid, colorid, cpkradius)
        tcl_out = path.join(self.tcl_folder, 'show_single_nucleotide.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_T_single')

    def vmd_show_g_tract_single_G(self):
        resid = 7
        bigatomlist = [['C6', 'C4'], ['N1', 'N3'], ['C2', 'N2', 'O6', 'C4', 'C5', 'N7', 'C8', 'N9']]
        colorid_list = [0, 0, 5]
        cpkradius_list = [1.2, 0.9, 0.5]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid(resid)
        for atomlist, colorid, cpkradius in zip(bigatomlist, colorid_list, cpkradius_list):
            lines += self.vmd_add_atomlist_vdw(atomlist, resid, colorid, cpkradius)
        tcl_out = path.join(self.tcl_folder, 'show_single_nucleotide.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_G_single')

    def vmd_show_g_tract_single_C(self):
        resid = 24
        bigatomlist = [['C4'], ['N3'], ['C2'], ['N1', 'O2', 'C6', 'C5', 'N4']]
        colorid_list = [0, 0, 0, 5]
        cpkradius_list = [1.2, 0.9, 0.7, 0.5]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid(resid)
        for atomlist, colorid, cpkradius in zip(bigatomlist, colorid_list, cpkradius_list):
            lines += self.vmd_add_atomlist_vdw(atomlist, resid, colorid, cpkradius)
        tcl_out = path.join(self.tcl_folder, 'show_single_nucleotide.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_C_single')

    def vmd_show_gcgc_single_G(self):
        resid = 7
        bigatomlist = [['C4'], ['C5'], ['N3', 'C2', 'C6', 'O6', 'N1', 'N2', 'C4', 'N7', 'C8', 'N9']]
        colorid_list = [0, 0, 5]
        cpkradius_list = [1.2, 0.9, 0.5]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid(resid)
        for atomlist, colorid, cpkradius in zip(bigatomlist, colorid_list, cpkradius_list):
            lines += self.vmd_add_atomlist_vdw(atomlist, resid, colorid, cpkradius)
        tcl_out = path.join(self.tcl_folder, 'show_single_nucleotide.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_G_single')

    def vmd_show_gcgc_single_C(self):
        resid = 8
        bigatomlist = [['N3', 'C2'], ['C4'], ['C5', 'N1', 'O2', 'C6', 'N4']]
        colorid_list = [0, 0, 5]
        cpkradius_list = [1.2, 0.7, 0.5]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid(resid)
        for atomlist, colorid, cpkradius in zip(bigatomlist, colorid_list, cpkradius_list):
            lines += self.vmd_add_atomlist_vdw(atomlist, resid, colorid, cpkradius)
        tcl_out = path.join(self.tcl_folder, 'show_single_nucleotide.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_C_single')

    def vmd_show_ctct_single_C(self):
        resid = 7
        bigatomlist = [['N1', 'N3', 'C2'], ['C5', 'C4', 'O2', 'C6', 'N4']]
        colorid_list = [0, 5]
        cpkradius_list = [1.2, 0.5]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid(resid)
        for atomlist, colorid, cpkradius in zip(bigatomlist, colorid_list, cpkradius_list):
            lines += self.vmd_add_atomlist_vdw(atomlist, resid, colorid, cpkradius)
        tcl_out = path.join(self.tcl_folder, 'show_single_nucleotide.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_C_single')

    def vmd_show_ctct_single_T(self):
        resid = 8
        bigatomlist = [['C4', 'C5'], ['N3', 'C2'], ['N1', 'O2', 'O4', 'C6', 'C7']]
        colorid_list = [0, 0, 5]
        cpkradius_list = [1.2, 0.7, 0.5]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid(resid)
        for atomlist, colorid, cpkradius in zip(bigatomlist, colorid_list, cpkradius_list):
            lines += self.vmd_add_atomlist_vdw(atomlist, resid, colorid, cpkradius)
        tcl_out = path.join(self.tcl_folder, 'show_single_nucleotide.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_T_single')

    def vmd_show_ctct_single_A(self):
        resid = 27
        bigatomlist = [['C6'], ['C5'], ['C4', 'C2', 'N3', 'N1', 'C6', 'N6', 'N7', 'C8', 'N9']]
        colorid_list = [0, 1, 5]
        cpkradius_list = [1.2, 1.0, 0.5]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid(resid)
        for atomlist, colorid, cpkradius in zip(bigatomlist, colorid_list, cpkradius_list):
            lines += self.vmd_add_atomlist_vdw(atomlist, resid, colorid, cpkradius)
        tcl_out = path.join(self.tcl_folder, 'show_single_nucleotide.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_A_single')

    def vmd_show_ctct_single_G(self):
        resid = 28
        bigatomlist = [['C6', 'N1'], ['C4'], ['N3', 'C5', 'C2', 'O6', 'N2', 'C4', 'N7', 'C8', 'N9']]
        colorid_list = [0, 1, 5]
        cpkradius_list = [1.2, 1.0, 0.5]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid(resid)
        for atomlist, colorid, cpkradius in zip(bigatomlist, colorid_list, cpkradius_list):
            lines += self.vmd_add_atomlist_vdw(atomlist, resid, colorid, cpkradius)
        tcl_out = path.join(self.tcl_folder, 'show_single_nucleotide.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_G_single')

    def vmd_show_tgtg_single_T(self):
        resid = 7
        bigatomlist = [['C4', 'C5'], ['N3'], ['C2'], ['N1', 'O2', 'O4', 'C6', 'C7']]
        colorid_list = [0, 0, 1, 5]
        cpkradius_list = [1.2, 1.2, 1.2, 0.5]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid(resid)
        for atomlist, colorid, cpkradius in zip(bigatomlist, colorid_list, cpkradius_list):
            lines += self.vmd_add_atomlist_vdw(atomlist, resid, colorid, cpkradius)
        tcl_out = path.join(self.tcl_folder, 'show_single_nucleotide.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_T_single')

    def vmd_show_tgtg_single_G(self):
        resid = 8
        bigatomlist = [['C4'], ['N3', 'C2'], ['C6', 'N1', 'C5', 'O6', 'N2', 'C4', 'N7', 'C8', 'N9']]
        colorid_list = [0, 0, 5]
        cpkradius_list = [1.2, 0.7, 0.5]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid(resid)
        for atomlist, colorid, cpkradius in zip(bigatomlist, colorid_list, cpkradius_list):
            lines += self.vmd_add_atomlist_vdw(atomlist, resid, colorid, cpkradius)
        tcl_out = path.join(self.tcl_folder, 'show_single_nucleotide.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_G_single')

    def vmd_show_tgtg_single_C(self):
        resid = 27
        bigatomlist = [['C4', 'N3', 'C2'], ['C5', 'N1', 'O2', 'C6', 'N4']]
        colorid_list = [0, 5]
        cpkradius_list = [1.2, 0.5]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid(resid)
        for atomlist, colorid, cpkradius in zip(bigatomlist, colorid_list, cpkradius_list):
            lines += self.vmd_add_atomlist_vdw(atomlist, resid, colorid, cpkradius)
        tcl_out = path.join(self.tcl_folder, 'show_single_nucleotide.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_C_single')

    def vmd_show_tgtg_single_A(self):
        resid = 26
        bigatomlist = [['C5', 'C4'], ['C2', 'C6', 'N3', 'N1', 'C6', 'N6', 'N7', 'C8', 'N9']]
        colorid_list = [0, 5]
        cpkradius_list = [1.2, 0.5]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid(resid)
        for atomlist, colorid, cpkradius in zip(bigatomlist, colorid_list, cpkradius_list):
            lines += self.vmd_add_atomlist_vdw(atomlist, resid, colorid, cpkradius)
        tcl_out = path.join(self.tcl_folder, 'show_single_nucleotide.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_A_single')

    def vmd_show_a_tract_AA_pair1(self):
        u = MDAnalysis.Universe(self.perferct_gro, self.perferct_gro)
        resid_i = 1
        resid_j = 2
        atompair_list = [('N1', 'N1'), ('N1', 'C6'), ('C6', 'C6'), ('C6', 'N6')]
        radius_list = [0.08, 0.15, 0.2, 0.08]
        color_list = [1, 1, 1, 1]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid_cpk_color_by_name(resid_i)
        lines += self.vmd_add_resid_cpk_color_by_name(resid_j)
        for atompair, radius, color in zip(atompair_list, radius_list, color_list):
            positions = self.get_pair_positions_by_resid_names(u, resid_i, resid_j, atompair[0], atompair[1])
            temp_lines = [f'graphics 0 color {color}',
                          self.__get_draw_edge_line(positions, 0, 1, radius)]
            lines += temp_lines
        tcl_out = path.join(self.tcl_folder, 'show_basestack_pair.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_AA_pair1')

    def vmd_show_a_tract_AA_pair2(self):
        u = MDAnalysis.Universe(self.perferct_gro, self.perferct_gro)
        resid_i = 1
        resid_j = 2
        atompair_list = [('C2', 'C5'), ('C2', 'C4'), ('N3', 'C4'), ('N3', 'C5'), ('C4', 'C4'), ('C4', 'C5'), ('C4', 'N7'), ('C5', 'C5')]
        radius_list = [0.08, 0.08, 0.15, 0.15, 0.08, 0.2, 0.08, 0.15]
        color_list = [7] * len(atompair_list)
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid_cpk_color_by_name(resid_i)
        lines += self.vmd_add_resid_cpk_color_by_name(resid_j)
        for atompair, radius, color in zip(atompair_list, radius_list, color_list):
            positions = self.get_pair_positions_by_resid_names(u, resid_i, resid_j, atompair[0], atompair[1])
            temp_lines = [f'graphics 0 color {color}',
                          self.__get_draw_edge_line(positions, 0, 1, radius)]
            lines += temp_lines
        tcl_out = path.join(self.tcl_folder, 'show_basestack_pair.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_AA_pair2')

    def vmd_show_a_tract_TT_pair(self):
        u = MDAnalysis.Universe(self.perferct_gro, self.perferct_gro)
        resid_i = 22
        resid_j = 23
        atompair_list = [('N1', 'C5'), ('C2', 'C5'), ('N3', 'C4'), ('N3', 'C5'), ('C2', 'C4'), ('C2', 'C6'), ('C4', 'C4')]
        radius_list = [0.2, 0.2, 0.15, 0.15, 0.08, 0.08, 0.08]
        color_list = [1] * len(atompair_list)
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid_cpk_color_by_name(resid_i)
        lines += self.vmd_add_resid_cpk_color_by_name(resid_j)
        for atompair, radius, color in zip(atompair_list, radius_list, color_list):
            positions = self.get_pair_positions_by_resid_names(u, resid_i, resid_j, atompair[0], atompair[1])
            temp_lines = [f'graphics 0 color {color}',
                          self.__get_draw_edge_line(positions, 0, 1, radius)]
            lines += temp_lines
        tcl_out = path.join(self.tcl_folder, 'show_basestack_pair.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_TT_pair')

    def vmd_show_ATAT_AT_pair1(self):
        u = MDAnalysis.Universe(self.perferct_gro, self.perferct_gro)
        resid_i = 1
        resid_j = 2
        atompair_list = [('C4', 'C4'), ('C4', 'C5'), ('C5', 'C4'), ('C5', 'C5'), ('C6', 'C4')]
        radius_list = [0.1, 0.1, 0.1, 0.1, 0.1]
        color_list = [1, 1, 1, 1, 1]
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid_cpk_color_by_name(resid_i)
        lines += self.vmd_add_resid_cpk_color_by_name(resid_j)
        for atompair, radius, color in zip(atompair_list, radius_list, color_list):
            positions = self.get_pair_positions_by_resid_names(u, resid_i, resid_j, atompair[0], atompair[1])
            temp_lines = [f'graphics 0 color {color}',
                          self.__get_draw_edge_line(positions, 0, 1, radius)]
            lines += temp_lines
        tcl_out = path.join(self.tcl_folder, 'show_basestack_pair.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_AT_pair1')

    def vmd_show_ATAT_AT_pair2(self):
        u = MDAnalysis.Universe(self.perferct_gro, self.perferct_gro)
        resid_i = 1
        resid_j = 2
        atompair_list = [('N1', 'N3'), ('C2', 'C2'), ('C2', 'N3'), ('N3', 'C2')]
        radius_list = [0.1, 0.1, 0.1, 0.1]
        color_list = [7] * len(atompair_list)
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid_cpk_color_by_name(resid_i)
        lines += self.vmd_add_resid_cpk_color_by_name(resid_j)
        for atompair, radius, color in zip(atompair_list, radius_list, color_list):
            positions = self.get_pair_positions_by_resid_names(u, resid_i, resid_j, atompair[0], atompair[1])
            temp_lines = [f'graphics 0 color {color}',
                          self.__get_draw_edge_line(positions, 0, 1, radius)]
            lines += temp_lines
        tcl_out = path.join(self.tcl_folder, 'show_basestack_pair.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_AT_pair2')

    def vmd_show_g_tract_GG_pair(self):
        u = MDAnalysis.Universe(self.perferct_gro, self.perferct_gro)
        resid_i = 1
        resid_j = 2
        atompair_list = [('N1', 'C6'), ('C6', 'C6'), ('N3', 'C4'), ('N1', 'N1'), ('C2', 'C4'), ('C4', 'C5'), ('C4', 'N7')]
        radius_list = [0.2, 0.2, 0.2, 0.08, 0.08, 0.08, 0.08]
        color_list = [1] * len(atompair_list)
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid_cpk_color_by_name(resid_i)
        lines += self.vmd_add_resid_cpk_color_by_name(resid_j)
        for atompair, radius, color in zip(atompair_list, radius_list, color_list):
            positions = self.get_pair_positions_by_resid_names(u, resid_i, resid_j, atompair[0], atompair[1])
            temp_lines = [f'graphics 0 color {color}',
                          self.__get_draw_edge_line(positions, 0, 1, radius)]
            lines += temp_lines
        tcl_out = path.join(self.tcl_folder, 'show_basestack_pair.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_GG_pair')

    def vmd_show_g_tract_CC_pair(self):
        u = MDAnalysis.Universe(self.perferct_gro, self.perferct_gro)
        resid_i = 22
        resid_j = 23
        atompair_list = [('N3', 'C4'), ('C2', 'C4'), ('N3', 'N4'), ('N3', 'N3')]
        radius_list = [0.2, 0.15, 0.08, 0.05]
        color_list = [1] * len(atompair_list)
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid_cpk_color_by_name(resid_i)
        lines += self.vmd_add_resid_cpk_color_by_name(resid_j)
        for atompair, radius, color in zip(atompair_list, radius_list, color_list):
            positions = self.get_pair_positions_by_resid_names(u, resid_i, resid_j, atompair[0], atompair[1])
            temp_lines = [f'graphics 0 color {color}',
                          self.__get_draw_edge_line(positions, 0, 1, radius)]
            lines += temp_lines
        tcl_out = path.join(self.tcl_folder, 'show_basestack_pair.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_CC_pair')

    def vmd_show_GCGC_GC_pair(self):
        u = MDAnalysis.Universe(self.perferct_gro, self.perferct_gro)
        resid_i = 1
        resid_j = 2
        atompair_list = [('C4', 'N3'), ('C4', 'C4'), ('N1', 'N3'), ('C2', 'N3'), ('C2', 'C2'), ('C5', 'C4'), ('N3', 'C2')]
        radius_list = [0.12, 0.12, 0.08, 0.08, 0.08, 0.08, 0.08]
        color_list = [1] * len(atompair_list)
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid_cpk_color_by_name(resid_i)
        lines += self.vmd_add_resid_cpk_color_by_name(resid_j)
        for atompair, radius, color in zip(atompair_list, radius_list, color_list):
            positions = self.get_pair_positions_by_resid_names(u, resid_i, resid_j, atompair[0], atompair[1])
            temp_lines = [f'graphics 0 color {color}',
                          self.__get_draw_edge_line(positions, 0, 1, radius)]
            lines += temp_lines
        tcl_out = path.join(self.tcl_folder, 'show_basestack_pair.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_GC_pair')

    def vmd_show_CTCT_CT_pair(self):
        u = MDAnalysis.Universe(self.perferct_gro, self.perferct_gro)
        resid_i = 1
        resid_j = 2
        atompair_list = [('N1', 'C5'), ('C2', 'C4'), ('C2', 'C5'), ('N3', 'C4')]
        radius_list = [0.12, 0.12, 0.12, 0.12]
        color_list = [1] * len(atompair_list)
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid_cpk_color_by_name(resid_i)
        lines += self.vmd_add_resid_cpk_color_by_name(resid_j)
        for atompair, radius, color in zip(atompair_list, radius_list, color_list):
            positions = self.get_pair_positions_by_resid_names(u, resid_i, resid_j, atompair[0], atompair[1])
            temp_lines = [f'graphics 0 color {color}',
                          self.__get_draw_edge_line(positions, 0, 1, radius)]
            lines += temp_lines
        tcl_out = path.join(self.tcl_folder, 'show_basestack_pair.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_CT_pair')

    def vmd_show_CTCT_GA_pair1(self):
        u = MDAnalysis.Universe(self.perferct_gro, self.perferct_gro)
        resid_i = 22
        resid_j = 23
        atompair_list = [('C4', 'C5')]
        radius_list = [0.15]
        color_list = [1] * len(atompair_list)
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid_cpk_color_by_name(resid_i)
        lines += self.vmd_add_resid_cpk_color_by_name(resid_j)
        for atompair, radius, color in zip(atompair_list, radius_list, color_list):
            positions = self.get_pair_positions_by_resid_names(u, resid_i, resid_j, atompair[0], atompair[1])
            temp_lines = [f'graphics 0 color {color}',
                          self.__get_draw_edge_line(positions, 0, 1, radius)]
            lines += temp_lines
        tcl_out = path.join(self.tcl_folder, 'show_basestack_pair.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_GA_pair1')

    def vmd_show_CTCT_GA_pair2(self):
        u = MDAnalysis.Universe(self.perferct_gro, self.perferct_gro)
        resid_i = 22
        resid_j = 23
        atompair_list = [('C6', 'C6'), ('N1', 'C6')]
        radius_list = [0.16, 0.08]
        color_list = [0] * len(atompair_list)
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid_cpk_color_by_name(resid_i)
        lines += self.vmd_add_resid_cpk_color_by_name(resid_j)
        for atompair, radius, color in zip(atompair_list, radius_list, color_list):
            positions = self.get_pair_positions_by_resid_names(u, resid_i, resid_j, atompair[0], atompair[1])
            temp_lines = [f'graphics 0 color {color}',
                          self.__get_draw_edge_line(positions, 0, 1, radius)]
            lines += temp_lines
        tcl_out = path.join(self.tcl_folder, 'show_basestack_pair.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_GA_pair2')

    def vmd_show_TGTG_GT_pair(self):
        u = MDAnalysis.Universe(self.perferct_gro, self.perferct_gro)
        resid_i = 4
        resid_j = 5
        atompair_list = [('C4', 'C5'), ('C4', 'C4'), ('C2', 'C2'), ('C2', 'N3')]
        radius_list = [0.15, 0.1, 0.1, 0.1]
        color_list = [1] * len(atompair_list)
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid_cpk_color_by_name(resid_i)
        lines += self.vmd_add_resid_cpk_color_by_name(resid_j)
        for atompair, radius, color in zip(atompair_list, radius_list, color_list):
            positions = self.get_pair_positions_by_resid_names(u, resid_i, resid_j, atompair[0], atompair[1])
            temp_lines = [f'graphics 0 color {color}',
                          self.__get_draw_edge_line(positions, 0, 1, radius)]
            lines += temp_lines
        tcl_out = path.join(self.tcl_folder, 'show_basestack_pair.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_GT_pair')

    def vmd_show_TGTG_AC_pair(self):
        u = MDAnalysis.Universe(self.perferct_gro, self.perferct_gro)
        resid_i = 22
        resid_j = 23
        atompair_list = [('C5', 'C4'), ('C4', 'C4'), ('N1', 'N3')]
        radius_list = [0.1, 0.1, 0.1]
        color_list = [1] * len(atompair_list)
        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid_cpk_color_by_name(resid_i)
        lines += self.vmd_add_resid_cpk_color_by_name(resid_j)
        for atompair, radius, color in zip(atompair_list, radius_list, color_list):
            positions = self.get_pair_positions_by_resid_names(u, resid_i, resid_j, atompair[0], atompair[1])
            temp_lines = [f'graphics 0 color {color}',
                          self.__get_draw_edge_line(positions, 0, 1, radius)]
            lines += temp_lines
        tcl_out = path.join(self.tcl_folder, 'show_basestack_pair.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_AC_pair')

    def get_pair_positions_by_resid_names(self, u, resid_i, resid_j, atomname_i, atomname_j):
        positions = np.zeros((2,3))
        positions[0,:] = u.select_atoms(f'resid {resid_i} and name {atomname_i}').positions[0,:]
        positions[1,:] = u.select_atoms(f'resid {resid_j} and name {atomname_j}').positions[0,:]
        return positions

    def vmd_add_resid(self, resid):
        lines = ['mol color ColorID 2',
                 'mol representation Licorice 0.100000 12.000000 12.000000',
                 f'mol selection resid {resid} and not hydrogen and not (name C1\' C2\' O4\' C3\' C4\' C5\' P O1P O2P O5\' O3\')',
                 'mol material Opaque',
                 'mol addrep 0']
        return lines

    def vmd_add_resid_cpk_color_by_name(self, resid):
        lines = ['mol color Name',
                 'mol representation CPK 1.00000 0.300000 12.000000 12.000000',
                 f'mol selection resid {resid} and not hydrogen and not (name C1\' C2\' O4\' C3\' C4\' C5\' P O1P O2P O5\' O3\')',
                 'mol material Transparent',
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

    def vmd_open_perfect_gro(self):
        print(f'vmd -gro {self.perferct_gro}')

    def write_tcl_out(self, tcl_out, container):
        f = open(tcl_out, 'w')
        for line in container:
            f.write(line)
            f.write('\n')
        f.close()
        print(f'source {tcl_out}')

    def print_tga_out(self, out_name):
        print(path.join(self.mol_stru_folder, out_name))


class StackWholeMolecule(BaseStackImportanceAgent):
    def __init__(self, host, rootfolder, pic_out_folder):
        super().__init__(host, rootfolder, pic_out_folder)

    def vmd_show_whole_stack(self, df_in, radius):
        u = MDAnalysis.Universe(self.perferct_gro, self.perferct_gro)
        zipobj = zip(df_in['Strand_i'].tolist(), df_in['Resid_i'].tolist(), df_in['Atomname_i'].tolist(), df_in['Strand_j'].tolist(), df_in['Resid_j'].tolist(), df_in['Atomname_j'].tolist())
        self.vmd_open_perfect_gro()
        lines = self.get_initial_lines()
        lines += [f'graphics 0 color 1'] # red color
        for strand_i, resid_i, atomname_i, strand_j, resid_j, atomname_j in zipobj:
            if (strand_i == 'STRAND2') and (strand_j == 'STRAND2'):
                gro_resid_i = resid_i + 21
                gro_resid_j = resid_j + 21
            else:
                gro_resid_i = resid_i
                gro_resid_j = resid_j
            positions = self.get_pair_positions_by_resid_names(u, gro_resid_i, gro_resid_j, atomname_i, atomname_j)
            lines += [self.__get_draw_edge_line(positions, 0, 1, radius)]
        tcl_out = path.join(self.tcl_folder, 'show_basestack_pair.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_stack_whole_mol')

    def get_initial_lines(self):
        return ['mol delrep 0 0',
                'mol color ColorID 2',
                'mol representation Licorice 0.200000 12.000000 12.000000',
                'mol selection all',
                'mol material Transparent',
                'mol addrep 0']

    def __get_draw_edge_line(self, positions, atomid1, atomid2, radius):
        str_0 = 'graphics 0 cylinder {'
        str_1 = f'{positions[atomid1,0]:.3f} {positions[atomid1,1]:.3f} {positions[atomid1,2]:.3f}'
        str_2 = '} {'
        str_3 = f'{positions[atomid2,0]:.3f} {positions[atomid2,1]:.3f} {positions[atomid2,2]:.3f}'
        str_4 = '} '
        str_5 = f'radius {radius:.2f}\n'
        return str_0 + str_1 + str_2 + str_3 + str_4 + str_5
