from k_b0_util import get_df_by_filter_bp
from na_seq import sequences
from spring import Spring

atomname_map = {'A': {'type1': 'N6', 'type2': 'N1'}, 
                'T': {'type1': 'O4', 'type2': 'N3'},
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
        return get_df_by_filter_bp(df, 'hb')

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
            typelist = ['type1', 'type2']
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

