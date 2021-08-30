from os import path
import numpy as np
import MDAnalysis
from enmspring.na_seq import sequences

class TriBaseStepsVMD:
    type_na = 'bdna+bdna'

    def __init__(self, all_folder, tcl_folder, host, resid_i):
        self.all_folder = all_folder
        self.tcl_folder = tcl_folder
        self.host = host
        self.resid_i = resid_i
        self.na_folder = path.join(self.all_folder, self.host, self.type_na)
        self.animation_folder = path.join(self.na_folder, 'animations')

        self.pdb = path.join(self.animation_folder, f'tri_bs_resid{self.resid_i}.pdb')

        self.tri_agent = ThreeBaseSteps(self.host, resid_i, self.pdb)

    def vmd_show(self):
        print(f'vmd -pdb {self.pdb}')

    def highlight_tribasesteps(self, i_or_j):
        #i_or_j: 'i', 'j'
        tcl_lst = ['mol delrep 0 0', 'mol color Name', 'mol representation Licorice 0.100000 12.000000 12.000000',
                   'mol selection all', 'mol material Transparent', 'mol addrep 0']
        tcl_lst += self.tri_agent.get_tri_baseatoms_selection(i_or_j)
        f_tcl_out = path.join(self.tcl_folder, f'highlight_tribasesteps_{i_or_j}.tcl')
        self.write_tcl_out(f_tcl_out, tcl_lst)

    def highlight_dibasesteps_baseatoms(self, resid_symbol_1, resid_symbol_2):
        tcl_lst = ['mol delrep 0 0'] * 7
        tcl_lst += self.tri_agent.get_baseatoms_selection_licorice_vdw(resid_symbol_1)
        tcl_lst += self.tri_agent.get_baseatoms_selection_licorice_vdw(resid_symbol_2)
        f_tcl_out = path.join(self.tcl_folder, f'highlight_dibasesteps_{resid_symbol_1}_{resid_symbol_2}.tcl')
        self.write_tcl_out(f_tcl_out, tcl_lst)

    def highlight_springs(self, i_or_j, d_springs, radius, colorname):
        #i_or_j: 'i', 'j'
        tcl_lst = self.tri_agent.get_highlight_springs_tcl_txt(i_or_j, d_springs, radius, colorname)
        f_tcl_out = path.join(self.tcl_folder, f'highlight_springs_{i_or_j}.tcl')
        self.write_tcl_out(f_tcl_out, tcl_lst)

    def highlight_dibasesteps_springs(self, res_pair, d_springs, radius, colorname):
        tcl_lst = self.tri_agent.get_highlight_springs_tcl_txt_by_resid_symbols(d_springs, res_pair, radius, colorname)
        symbol_1, symbol_2 = res_pair
        f_tcl_out = path.join(self.tcl_folder, f'highlight_springs_{symbol_1}_{symbol_2}.tcl')
        self.write_tcl_out(f_tcl_out, tcl_lst)

    def write_tcl_out(self, tcl_out, container):
        f = open(tcl_out, 'w')
        for line in container:
            f.write(line)
            f.write('\n')
        f.close()
        print(f'source {tcl_out}')

    def tachyon_take_photo_cmd(self, drawzone_folder, tga_name):
        output = path.join(drawzone_folder, tga_name)
        str_1 = f'render Tachyon {output} '
        str_2 = '"/usr/local/lib/vmd/tachyon_LINUXAMD64" '
        str_3 = '-aasamples 12 %s -format TARGA -o %s.tga'
        print(str_1+str_2+str_3)

class ThreeBaseSteps:
    total_n_bp = 42
    strand_lst = ['STRAND1', 'STRAND2']
    strand1_resid_lst = ['i-1', 'i', 'i+1']
    strand2_resid_lst = ['j-1', 'j', 'j+1']
    d_resid_lst = {'i': strand1_resid_lst, 'j': strand2_resid_lst}
    offset_lst = [-1, 0, 1]
    d_atomlist = {'A': ['N9', 'C8', 'N7', 'C5', 'C4', 'N3', 'C2', 'N1', 'C6'],
                  'T': ['N1', 'C6', 'C5', 'C4', 'N3', 'C2'],
                  'C': ['N1', 'C6', 'C5', 'C4', 'N3', 'C2'],
                  'G': ['N9', 'C8', 'N7', 'C5', 'C4', 'N3', 'C2', 'N1', 'C6']}
    d_atomlist_full = {'A': ['N9', 'C8', 'N7', 'C5', 'C4', 'N3', 'C2', 'N1', 'C6', 'N6', 'H61', 'H62', 'H2', 'H8'],
                       'T': ['N1', 'C6', 'C5', 'C4', 'N3', 'C2', 'O2', 'O4', 'C7', 'H6', 'H71', 'H72', 'H73', 'H3'],
                       'C': ['N1', 'C6', 'C5', 'C4', 'N3', 'C2', 'O2', 'N4', 'H1', 'H21', 'H22', 'H8'],
                       'G': ['N9', 'C8', 'N7', 'C5', 'C4', 'N3', 'C2', 'N1', 'C6', 'O6', 'N2', 'H41', 'H42', 'H5', 'H6']}

    def __init__(self, host, resid_i, pdb):
        self.host = host
        self.resid_i = resid_i
        self.resid_j = self.get_resid_j()
        self.pdb = pdb

        self.d_resid_map = self.get_d_resid_map()
        self.d_resname_map = self.get_d_resname_map()

        self.u = MDAnalysis.Universe(self.pdb, self.pdb)

    def get_resid_j(self):
        return self.total_n_bp + 1 - self.resid_i

    def get_d_resid_map(self):
        d_resid_map = dict()
        for idx, offset in enumerate(self.offset_lst):
            d_resid_map[self.strand1_resid_lst[idx]] = self.resid_i + offset
            d_resid_map[self.strand2_resid_lst[idx]] = self.resid_j + offset
        return d_resid_map

    def get_d_resname_map(self):
        strand1_seq = [nt for nt in sequences[self.host]['guide']] # 5' -> 3'
        strand2_seq = [nt for nt in sequences[self.host]['target']] # 5' -> 3'
        seq_lst = strand1_seq + strand2_seq
        d_resname_map = dict()
        for resid_i_symbol, resid_j_symbol in zip(self.strand1_resid_lst, self.strand2_resid_lst):
            resid_i = self.d_resid_map[resid_i_symbol]
            resid_j = self.d_resid_map[resid_j_symbol]
            d_resname_map[resid_i_symbol] = seq_lst[resid_i - 1]
            d_resname_map[resid_j_symbol] = seq_lst[resid_j - 1]
        return d_resname_map

    def get_tri_baseatoms_selection(self, i_or_j):
        txt_list = list()
        for resid_symbol in self.d_resid_lst[i_or_j]:
            txt_list += self.get_baseatoms_selection(resid_symbol)
        return txt_list

    def get_baseatoms_selection(self, resid_symbol):
        resname = self.d_resname_map[resid_symbol]
        atom_lst = self.d_atomlist_full[resname]
        atom_text = ' '.join(atom_lst)
        resid = self.d_resid_map[resid_symbol]
        selection = f'mol selection (resid {resid}) and (name {atom_text})'
        txt_list = ['mol color Name', 'mol representation VDW 0.200000 12.000000',
                     selection, 'mol material AOChalky', 'mol addrep 0']
        return txt_list

    def get_baseatoms_selection_licorice_vdw(self, resid_symbol):
        resname = self.d_resname_map[resid_symbol]
        atom_lst = self.d_atomlist_full[resname]
        atom_text = ' '.join(atom_lst)
        resid = self.d_resid_map[resid_symbol]
        selection = f'mol selection (resid {resid}) and (name {atom_text})'
        txt_list = ['mol color Name', 'mol representation VDW 0.200000 12.000000',
                     selection, 'mol material AOChalky', 'mol addrep 0']
        txt_list += ['mol color Name', 'mol representation Licorice 0.100000 12.000000 12.000000',
                     selection, 'mol material Transparent', 'mol addrep 0']
        return txt_list

    def get_phosphate_selection(self, resid_symbol):
        resname = self.d_resname_map[resid_symbol]
        atom_lst = self.d_atomlist_full[resname]
        atom_text = ' '.join(atom_lst)
        resid = self.d_resid_map[resid_symbol]
        selection = f'mol selection (resid {resid}) and not (name {atom_text})'
        txt_list = ['mol color Name', 
                    'mol representation CPK 1.000000 0.300000 12.000000 12.000000',
                     selection,
                    'mol material Opaque',
                    'mol addrep 0']
        return txt_list

    def get_highlight_springs_tcl_txt(self, i_or_j, d_springs, radius, colorname):
        if i_or_j == 'i':
            res_pair_lst = [('i', 'i-1'), ('i', 'i+1')]
        else:
            res_pair_lst = [('j', 'j-1'), ('j', 'j+1')]
        txt_lst = []
        for res_pair in res_pair_lst:
            txt_lst += self.get_highlight_springs_tcl_txt_by_resid_symbols(d_springs, res_pair, radius, colorname)
        return txt_lst

    def get_highlight_springs_tcl_txt_by_resid_symbols(self, d_springs, res_pair, radius, colorname):
        txt_lst = []
        pair_lst = d_springs[self.host][res_pair]
        resid1 = self.d_resid_map[res_pair[0]]
        resid2 = self.d_resid_map[res_pair[1]]
        for atomname1, atomname2 in pair_lst:
            positions = self.get_pair_positions_by_resid_names(self.u, resid1, resid2, atomname1, atomname2)
            txt_lst += self.get_draw_edge_line(positions, radius, colorname)
        return txt_lst

    def get_pair_positions_by_resid_names(self, u, resid_i, resid_j, atomname_i, atomname_j):
        positions = np.zeros((2,3))
        positions[0,:] = u.select_atoms(f'resid {resid_i} and name {atomname_i}').positions[0,:]
        positions[1,:] = u.select_atoms(f'resid {resid_j} and name {atomname_j}').positions[0,:]
        return positions

    def get_draw_edge_line(self, positions, radius, colorname):
        color_line = f'graphics 0 color {colorname}'
        str_0 = 'graphics 0 cylinder {'
        str_1 = f'{positions[0,0]:.3f} {positions[0,1]:.3f} {positions[0,2]:.3f}'
        str_2 = '} {'
        str_3 = f'{positions[1,0]:.3f} {positions[1,1]:.3f} {positions[1,2]:.3f}'
        str_4 = '} '
        str_5 = f'radius {radius:.2f}\n'
        return [color_line, str_0 + str_1 + str_2 + str_3 + str_4 + str_5]