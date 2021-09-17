from os import path
from enmspring.vmddraw_tribasesteps import TriBaseStepsVMD

class TriBaseStepsBackboneVMD(TriBaseStepsVMD):
    def highlight_single_nucleotdie(self, i_or_j):
        #i_or_j: 'i', 'j'
        tcl_lst = ['mol delrep 0 0', 'mol color Name', 'mol representation Licorice 0.100000 12.000000 12.000000',
                   'mol selection all', 'mol material Transparent', 'mol addrep 0']
        tcl_lst += self.tri_agent.get_single_nucleotide_selection(i_or_j)
        f_tcl_out = path.join(self.tcl_folder, f'highlight_single_nucleotide_{i_or_j}.tcl')
        self.write_tcl_out(f_tcl_out, tcl_lst)

    def highlight_springs(self, i_or_j, d_springs, radius, colorname):
        #i_or_j: 'i', 'j'
        tcl_lst = self.tri_agent.get_highlight_springs_tcl_txt_backbone(i_or_j, d_springs, radius, colorname)
        f_tcl_out = path.join(self.tcl_folder, f'highlight_springs_{i_or_j}.tcl')
        self.write_tcl_out(f_tcl_out, tcl_lst)