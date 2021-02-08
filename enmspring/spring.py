from os import path
import pandas as pd
import MDAnalysis
from enmspring import ic_table
from enmspring import pairtype
from enmspring.miscell import check_dir_exist_and_make
from enmspring.k_b0_util import get_df_by_filter_st, get_df_by_filter_bp, get_central_bps_df

class BigTrajAgent:
    start_time = 0
    end_time = 5000 # 5000 ns
    interval_time = 1000 # unit: ns
    n_bp = 21
    cutoff = 4.7
    clean_criteria = 1e-3
    interactions = ['other', 'backbone', 'stack', 'sugar', 'HB']

    def __init__(self, host, type_na, bigtraj_folder, only_central):
        self.host = host
        self.type_na = type_na
        self.bigtraj_folder = bigtraj_folder
        self.only_central = only_central
        self.time_list, self.mdnum_list = self.get_time_list()
        self.d_smallagents = self.get_all_small_agents()

        self.d_df_st = dict()
        self.d_df_hb = dict()
        
    def get_time_list(self):
        middle_interval = int(self.interval_time/2)
        time_list = list()
        mdnum_list = list()
        mdnum1 = 1
        for time1 in range(self.start_time, self.end_time, middle_interval):
            time2 = time1 + self.interval_time
            if time2 <= self.end_time:
                time_list.append((time1, time2))
                mdnum_list.append((mdnum1, mdnum1+9))
            mdnum1 += 5
        return time_list, mdnum_list

    def get_all_small_agents(self):
        d_smallagents = dict()
        for time1, time2 in self.time_list:
            time_label = f'{time1}_{time2}'
            d_smallagents[(time1,time2)] = SmallTrajSpring(self.bigtraj_folder, self.host, self.type_na, time_label, self.n_bp)
        return d_smallagents

    def set_required_dictionaries(self):
        for time1, time2 in self.time_list:
            self.d_smallagents[(time1,time2)].set_mda_universe()
            self.d_smallagents[(time1,time2)].set_required_map()

    def make_all_k_b0_pairtype_df(self):
        for time1, time2 in self.time_list:
            spring_obj = self.d_smallagents[(time1,time2)]
            df = spring_obj.make_k_b0_pairtype_df_given_cutoff(self.cutoff)
            mask = df['k'] > self.clean_criteria
            df_1 = df[mask]
            print(f'For {time1}_{time2}')
            for interaction in self.interactions:
                mask = df_1['Big_Category'] == interaction
                df_2 = df_1[mask]
                n_bonds = df_2.shape[0]
                print(f'There are {n_bonds} {interaction}-bonds')
            print('\n')

    def read_all_k_b0_pairtype_df(self):
        for time1, time2 in self.time_list:
            spring_obj = self.d_smallagents[(time1,time2)]
            spring_obj.read_k_b0_pairtype_df_given_cutoff(self.cutoff, self.only_central)

    def put_all_df_st_into_dict(self):
        for time1, time2 in self.time_list:
            spring_obj = self.d_smallagents[(time1,time2)]
            self.d_df_st[(time1,time2)] = spring_obj.get_df_st()

    def put_all_df_hb_into_dict(self):
        for time1, time2 in self.time_list:
            spring_obj = self.d_smallagents[(time1,time2)]
            self.d_df_hb[(time1,time2)] = spring_obj.get_df_hb()

    def get_k_st_list(self):
        k_st_list = list()
        for time1, time2 in self.time_list:
            k_st_list += self.d_df_st[(time1,time2)]['k'].tolist()
        return k_st_list

    def get_k_hb_list(self):
        k_hb_list = list()
        for time1, time2 in self.time_list:
            k_hb_list += self.d_df_hb[(time1,time2)]['k'].tolist()
        return k_hb_list

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

class SmallTrajSpring(Spring):
    def __init__(self, rootfolder, host, type_na, time_label, n_bp):
        self.rootfolder = rootfolder
        self.host = host
        self.type_na = type_na
        self.time_label = time_label
        self.n_bp = n_bp
        self.host_folder = path.join(rootfolder, host)
        self.na_folder = path.join(self.host_folder, type_na)
        self.time_folder = path.join(self.na_folder, time_label)

        self.prm_folder = path.join(self.time_folder, 'data')
        self.input_folder = path.join(self.time_folder, 'input')
        self.crd = path.join(self.input_folder, '{0}.nohydrogen.crd'.format(self.type_na))
        self.pd_dfs_folder = path.join(self.time_folder, 'pd_dfs')
        self.df_all_k = None
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

    def read_k_b0_pairtype_df_given_cutoff(self, cutoff, only_central):
        f_in = path.join(self.pd_dfs_folder, f'pairtypes_k_b0_cutoff_{cutoff:.2f}.csv')
        if only_central:
            self.df_all_k = get_central_bps_df(pd.read_csv(f_in))
        else:
            self.df_all_k = pd.read_csv(f_in)
        print(f'Read {f_in} into df_all_k')

    def get_df_st(self):
        criteria = 1e-3
        df1 = get_df_by_filter_st(self.df_all_k, 'st')
        mask = (df1['k'] > criteria)
        print("Read Dataframe of stacking: df_st")
        return df1[mask]

    def get_df_hb(self):
        df1 = get_df_by_filter_bp(self.df_all_k, 'hb')
        df2 = self.__read_df_at_type3()
        if len(df2) == 0:
            df_result = df1
        else:
            df3 = pd.concat([df1,df2])
            df3 = df3.sort_values(by=['Resid_i'])
            df3 = df3.reset_index()
            df_result = df3
        print("Read Dataframe of HB: df_hb")
        return df_result

    def __read_df_at_type3(self):
        df1 = get_df_by_filter_bp(self.df_all_k, 'bp1')
        df2_1 = self.__filter_C2_O2(df1)
        df2_2 = self.__filter_O2_C2(df1)
        return pd.concat([df2_1, df2_2])

    def __filter_C2_O2(self, df):
        mask0 = (df['Atomname_i'] == 'C2')
        df0 = df[mask0]
        mask1 = (df0['Atomname_j'] == 'O2')
        return df0[mask1]

    def __filter_O2_C2(self, df):
        mask0 = (df['Atomname_i'] == 'O2')
        df0 = df[mask0]
        mask1 = (df0['Atomname_j'] == 'C2')
        return df0[mask1]


