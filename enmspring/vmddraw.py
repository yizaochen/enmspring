from os import path
from enmspring.miscell import check_dir_exist_and_make
enmspring_folder = '/home/yizaochen/codes/dna_rna/enmspring'

class BaseStackImportanceAgent:

    def __init__(self, host, pic_out_folder):
        self.host = host
        self.tcl_folder = path.join(enmspring_folder, 'tclscripts')
        self.pic_out_folder = pic_out_folder
        self.mol_stru_folder = path.join(self.pic_out_folder, 'mol_structure')

        self.check_folder()

    def check_folder(self):
        for folder in [self.mol_stru_folder]:
            check_dir_exist_and_make(folder)

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

    def vmd_show_g_tract_single_G(self):
        resid = 7
        bigatomlist = [['C6'], ['C4'], ['N1'], ['O6'], ['C2', 'N2', 'N3', 'C4', 'C5', 'N7', 'C8', 'N9']]
        colorid_list = [0, 1, 0, 1, 5]
        cpkradius_list = [1.2, 1.2, 0.9, 0.9, 0.5]
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

    def vmd_open_perfect_gro(self):
        aa_folder = path.join('/home/yizaochen/codes/dna_rna/all_systems', self.host, 'bdna+bdna', 'input', 'allatoms')
        perferct_gro = path.join(aa_folder, 'bdna+bdna.perfect.gro')
        print(f'vmd -gro {perferct_gro}')

    def write_tcl_out(self, tcl_out, container):
        f = open(tcl_out, 'w')
        for line in container:
            f.write(line)
            f.write('\n')
        f.close()
        print(f'source {tcl_out}')

    def print_tga_out(self, out_name):
        print(path.join(self.mol_stru_folder, out_name))