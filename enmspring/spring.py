from os import path
import pandas as pd
import MDAnalysis
from enmspring import ic_table
from enmspring import pairtype
from enmspring.miscell import check_dir_exist_and_make

class Spring:
    col_names = ['PairID', 'PairType', 'Big_Category', 'Strand_i', 'Resid_i',  
                 'Atomname_i', 'Atomid_i', 'Strand_j', 'Resid_j', 'Atomname_j', 'Atomid_j', 'k', 'b0']

    def __init__(self, rootfolder, host, type_na, n_bp):
        self.rootfolder = rootfolder
        self.host = host
        self.type_na = type_na
        self.n_bp = n_bp
        self.host_folder = path.join(rootfolder, host)
        self.na_folder = path.join(self.host_folder, type_na)
        self.prm_folder = path.join(self.na_folder, 'cutoffdata')
        self.input_folder = path.join(self.na_folder, 'input')
        self.crd = path.join(self.input_folder,
                             '{0}.nohydrogen.avg.crd'.format(self.type_na))
        self.pd_dfs_folder = path.join(self.na_folder, 'pd_dfs')
        self.initialize_folders()

        self.u = None
        self.map = None
        self.inverse_map = None
        self.residues_map = None
        self.atomid_map = None
        self.atomid_map_inverse = None
        self.atomname_map = None
        self.strandid_map = None
        self.resid_map = None
        self.mass_map = None

    def initialize_folders(self):
        for folder in [self.pd_dfs_folder]:
            check_dir_exist_and_make(folder)

    def set_mda_universe(self):
        self.u = MDAnalysis.Universe(self.crd, self.crd)

    def set_required_map(self):
        self.map, self.inverse_map, self.residues_map, self.atomid_map,\
        self.atomid_map_inverse, self.atomname_map, self.strandid_map,\
        self.resid_map, self.mass_map = self.__build_map()

    def make_k_b0_pairtype_df_given_cutoff(self, cutoff):
        d_result = self.__initialize_d_result()
        kbpair = self.__get_kbpair(cutoff)

        pair_id = 1
        for name1, name2, k, b0 in zip(kbpair.d['name1'], kbpair.d['name2'], kbpair.d['k'], kbpair.d['b']):
            site1_id = self.atomid_map[name1]
            site2_id = self.atomid_map[name2]
            strandid1, resid1, atomname1, strandid2, resid2, atomname2 =\
                self.__get_strandid_resid_atomname(site1_id, site2_id)
            temp = pairtype.Pair(strandid1, resid1, atomname1, strandid2, resid2, atomname2, n_bp=self.n_bp)
            d_result['PairID'].append(pair_id)
            d_result['PairType'].append(temp.pair_type)
            d_result['Big_Category'].append(temp.big_category)
            d_result['Strand_i'].append(strandid1)
            d_result['Resid_i'].append(resid1)
            d_result['Atomname_i'].append(atomname1)
            d_result['Atomid_i'].append(site1_id)
            d_result['Strand_j'].append(strandid2)
            d_result['Resid_j'].append(resid2)
            d_result['Atomname_j'].append(atomname2)
            d_result['Atomid_j'].append(site2_id)
            d_result['k'].append(k)
            d_result['b0'].append(b0)
            pair_id += 1
        df = pd.DataFrame(d_result)
        df = df[self.col_names]
        f_out = path.join(self.pd_dfs_folder, f'pairtypes_k_b0_cutoff_{cutoff:.2f}.csv')
        df.to_csv(f_out, index=False)
        return df

    def read_k_b0_pairtype_df_given_cutoff(self, cutoff):
        f_in = path.join(self.pd_dfs_folder, f'pairtypes_k_b0_cutoff_{cutoff:.2f}.csv')
        df = pd.read_csv(f_in)
        return df

    def __initialize_d_result(self):
        d_result = dict()
        for col_name in self.col_names:
            d_result[col_name] = list()
        return d_result

    def __get_kbpair(self, cutoff):
        f_prm = path.join(self.prm_folder, f'na_enm_{cutoff:.2f}.prm')
        return ic_table.KBPair(read_from_prm=True, filename=f_prm)
        
    def __get_selection(self, atom):
        return f'segid {atom.segid} and resid {atom.resid} and name {atom.name}'
        
    def __get_strandid_resid_atomname(self, site1_id, site2_id):
        name1 = self.atomid_map_inverse[site1_id]
        name2 = self.atomid_map_inverse[site2_id]
        strandid1 = self.strandid_map[name1]
        strandid2 = self.strandid_map[name2]
        resid1 = self.resid_map[name1]
        resid2 = self.resid_map[name2]
        atomname1 = self.atomname_map[name1]
        atomname2 = self.atomname_map[name2]
        return strandid1, resid1, atomname1, strandid2, resid2, atomname2

    def __build_map(self):
        d1 = dict()  # key: selction, value: cgname
        d2 = dict()  # key: cgname,   value: selection
        d3 = dict()
        d4 = dict()  # key: cgname, value: atomid
        d5 = dict()  # key: atomid, value: cgname
        d6 = dict()  # key: cgname, value: atomname
        d7 = dict()  # key: cgname, value: strand_id
        d8 = dict()  # key: cgname, value: resid
        d9 = dict()  # key: cgname, value: mass
        atomid = 1
        segid1 = self.u.select_atoms("segid STRAND1")
        d3['STRAND1'] = dict()
        for i, atom in enumerate(segid1):
            cgname = 'A{0}'.format(i+1)
            selection = self.__get_selection(atom)
            d1[selection] = cgname
            d2[cgname] = selection
            if atom.resid not in d3['STRAND1']:
                d3['STRAND1'][atom.resid] = list()
            d3['STRAND1'][atom.resid].append(cgname)
            d4[cgname] = atomid
            d5[atomid] = cgname
            d6[cgname] = atom.name
            d7[cgname] = 'STRAND1'
            d8[cgname] = atom.resid
            d9[cgname] = atom.mass
            atomid += 1
        segid2 = self.u.select_atoms("segid STRAND2")
        d3['STRAND2'] = dict()
        for i, atom in enumerate(segid2):
            cgname = 'B{0}'.format(i+1)
            selection = self.__get_selection(atom)
            d1[selection] = cgname
            d2[cgname] = selection
            if atom.resid not in d3['STRAND2']:
                d3['STRAND2'][atom.resid] = list()
            d3['STRAND2'][atom.resid].append(cgname)
            d4[cgname] = atomid
            d5[atomid] = cgname
            d6[cgname] = atom.name
            d7[cgname] = 'STRAND2'
            d8[cgname] = atom.resid
            d9[cgname] = atom.mass
            atomid += 1
        return d1, d2, d3, d4, d5, d6, d7, d8, d9