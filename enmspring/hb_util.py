import pandas as pd
from enmspring.k_b0_util import get_df_by_filter_bp
from enmspring.na_seq import sequences
from enmspring.spring import Spring
import MDAnalysis as mda

atomname_map = {'A': {'type1': 'N6', 'type2': 'N1', 'type3': 'C2'}, 
                'T': {'type1': 'O4', 'type2': 'N3', 'type3': 'O2'},
                'C': {'type1': 'N4', 'type2': 'N3', 'type3': 'O2'},
                'G': {'type1': 'O6', 'type2': 'N1', 'type3': 'N2'}}

class InputException(Exception):
    pass

class HBAgent:
    cutoff = 4.7
    type_na = 'bdna+bdna'

    def __init__(self, host, rootfolder, n_bp):
        self.rootfolder = rootfolder
        self.host = host
        self.n_bp = n_bp
        self.seq_guide = sequences[host]['guide']

        self.df_hb = self.__read_df_hb()

        self.basepairs = None

    def initialize_basepair(self):
        basepairs = dict()
        for idx, resname_i in enumerate(self.seq_guide):
            resid_i = idx + 1
            bp_obj = BasePair(resname_i, resid_i, self.df_hb)
            basepairs[resid_i] = bp_obj
        self.basepairs = basepairs

    def __read_df_hb(self):
        spring_obj = Spring(self.rootfolder, self.host, self.type_na, self.n_bp)
        df = spring_obj.read_k_b0_pairtype_df_given_cutoff(self.cutoff)
        df1 = get_df_by_filter_bp(df, 'hb')
        df2 = self.__read_df_at_type3()
        if len(df2) == 0:
            return df1
        else:
            df3 = pd.concat([df1,df2])
            df3 = df3.sort_values(by=['Resid_i'])
            df3 = df3.reset_index()
            return df3

    def __read_df_at_type3(self):
        spring_obj = Spring(self.rootfolder, self.host, self.type_na, self.n_bp)
        df0 = spring_obj.read_k_b0_pairtype_df_given_cutoff(self.cutoff)
        df1 = get_df_by_filter_bp(df0, 'bp1')
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

    def get_resid_klist_by_type(self, bptype, typename):
        resid_list = list()
        k_list = list()
        for resid in range(1, self.n_bp+1):
            basepair = self.basepairs[resid]
            if basepair.bp_type == bptype:
                resid_list.append(resid)
                k_list.append(basepair.k_dict[typename])
        return resid_list, k_list

class BasePair:
    def __init__(self, resname_i, resid_i, df):
        self.resname_i = resname_i
        self.resid_i = resid_i
        self.bp_type = self.determine_bptype(resname_i) # AT or GC
        self.k_dict = self.get_k_dict(df)

    def determine_bptype(self, resname):
        if resname in ['A', 'T']:
            return 'AT'
        elif resname in ['G', 'C']:
            return 'GC'
        else:
            raise InputException('Something wrong with the DNA sequence.')

    def get_k_dict(self, df):
        if self.bp_type == 'AT':
            typelist = ['type1', 'type2', 'type3']
            return self.get_k_by_df(df, typelist)
        else:
            typelist = ['type1', 'type2', 'type3']
            return self.get_k_by_df(df, typelist)

    def get_k_by_df(self, df0, typelist):
        d_result = dict()
        mask1 = (df0['Resid_i'] == self.resid_i)
        df1 = df0[mask1]
        for typename in typelist:
            atomname = atomname_map[self.resname_i][typename]
            mask = (df1['Atomname_i'] == atomname)
            df2 = df1[mask]
            if len(df2) == 0:
                d_result[typename] = 0
            else:
                d_result[typename] = df2['k'].iloc[0]
        return d_result


class HBPainter:

    complementary = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    d_typelist = {'A': ['type1', 'type2'],
                  'T': ['type1', 'type2'],
                  'G': ['type1', 'type2', 'type3'],
                  'C': ['type1', 'type2', 'type3']}
    d_color = {'A': {'type1': 27, 'type2': 10},
               'T': {'type1': 27, 'type2': 10},
               'G': {'type1': 0, 'type2': 3, 'type3': 7},
               'C': {'type1': 0, 'type2': 3, 'type3': 7}}
    n_bp = 21
    width = 5
    
    def __init__(self, host, gro, f_out, mol_id):
        self.host = host
        self.gro = gro
        self.seq_guide = sequences[host]['guide']
        self.f_out = f_out
        self.mol_id = mol_id

        self.u = mda.Universe(gro, gro)

    def main(self):
        f = open(self.f_out, 'w')
        vmdlines = self.get_vmdlines()
        for line in vmdlines:
            f.write(line)
        f.close()
        print(f'Write draw codes into {self.f_out}')

    def get_vmdlines(self):
        vmdlines = list()
        for idx, resname_i in enumerate(self.seq_guide):
            resid_i = idx + 1
            resid_j = self.get_resid_j(resid_i)
            resname_j = self.complementary[resname_i]
            typelist = self.d_typelist[resname_i]
            for hb_type in typelist:
                name_i = atomname_map[resname_i][hb_type]
                name_j = atomname_map[resname_j][hb_type]
                vmdlines.append(self.get_color_line(resname_i, hb_type))
                vmdlines.append(self.get_line(resid_i, resid_j, name_i, name_j))
        return vmdlines

    def get_resid_j(self, resid_i):
        return (2 * self.n_bp + 1) - resid_i

    def get_color_line(self, resname, hb_type):
        colorid = self.d_color[resname][hb_type]
        return f'graphics {self.mol_id} color {colorid}\n'

    def get_line(self, resid_i, resid_j, name_i, name_j):
        pos_str_i = self.get_position_str(resid_i, name_i)
        pos_str_j = self.get_position_str(resid_j, name_j)
        return f'graphics {self.mol_id} line {pos_str_i} {pos_str_j} width {self.width}\n'

    def get_position_str(self, resid, name):
        atom_select = self.u.select_atoms(f'resid {resid} and name {name}')
        if atom_select.n_atoms != 1:
            raise InputException('Something wrong with the input gro.')
        position = atom_select.positions[0]
        return '{' + f'{position[0]:.3f} {position[1]:.3f} {position[2]:.3f}' + '}'

