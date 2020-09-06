import re
d_atomcgtype = {'O1P': 'P', 'P': 'P', 'O2P': 'P', 'O5\'': 'P',
                'C5\'': 'P', 'O3\'': 'P', 'C4\'': 'S', 'O4\'': 'S',
                'C1\'': 'S', 'C2\'': 'S', 'C3\'': 'S', 'O2\'': 'S',
                'N1': 'B', 'C2': 'B', 'N2': 'B', 'O2': 'B', 'N3': 'B',
                'C4': 'B', 'O4': 'B', 'N4': 'B', 'C5': 'B', 'C5M': 'B',
                'C6': 'B', 'N6': 'B', 'O6': 'B', 'N7': 'B', 'C8': 'B',
                'N9': 'B', 'C7': 'B'}
d_pairtype = {'P-S': 'P-S', 'S-P': 'P-S', 'P-B': 'P-B', 'B-P': 'P-B',
              'S-B': 'S-B', 'B-S': 'S-B', 'S-S': 'S-S', 'P-P': 'P-P',
              'B-B': 'B-B'}
hb_atomtypes = ['N6', 'N1', 'O4', 'N3', 'O6', 'N1', 'N2', 'N4', 'O2']

interactions = {'backbone': ['same-P-P-0', 'same-P-P-1', 'same-P-P-2', 'same-P-S-0', 'same-P-S-1', 'same-P-S-2',
                             'same-P-B-0', 'same-P-B-1', 'same-P-B-2'],
                'sugar': ['same-S-S-0', 'same-S-S-1', 'same-S-S-2', 'same-S-B-0', 'same-S-B-1', 'same-S-B-2'],
                'stack': ['STACK-1', 'STACK-2', 'STACK-3'],
                'HB': ['HB-0', 'Oppo-Ring-0', 'HB-1', 'Oppo-Ring-1', 'HB-2', 'Oppo-Ring-2', 'HB-3', 'Oppo-Ring-3']}
chemicalbond_group = ['same-S-S-0', 'same-P-S-0', 'Within-Ring', 'same-P-P-1', 'same-P-P-0', 'same-S-B-0']
d_bond = {'same-P-P-0': ['P-O1P', 'O5\'-C5\'', 'P-O5\'', 'P-O2P'],
          'same-P-P-1': ['O3\'-P'],
          'same-S-S-0': ['C4\'-O4\'', 'C1\'-C2\'', 'C3\'-C2\'', 'O4\'-C1\'', 'C2\'-O2\'', 'C4\'-C3\''],
          'same-P-S-0': ['C5\'-C4\'', 'C3\'-O3\''],
          'same-S-B-0': ['C1\'-N9', 'C1\'-N1'],
          'Within-Ring': ['C2-O2', 'N9-C4', 'N9-C8', 'C4-N3', 'C2-N2', 'N1-C2', 
                          'N1-C6', 'C6-C5', 'N7-C5', 'C2-N3', 'C5-C4', 'C8-N7',
                          'C6-O6', 'C6-N6', 'C6-N1', 'C5-C6', 'C4-N4', 'N3-C4',
                          'N3-C2', 'C4-O4', 'C5-C5M']}

d_angle = {'same-P-P-0': ['O2P-O5\'', 'O1P-O5\'', 'O5\'-O3\'', 'O1P-O2P', 'P-C5\'', 'O2P-O3\'', 'O1P-O3\''],
           'same-P-P-1': ['O3\'-O5\'', 'O3\'-O1P', 'O3\'-O2P'],
           'same-S-S-0': ['C4\'-C1\'', 'O4\'-C2\'', 'O4\'-C3\'', 'C3\'-O2\'', 'C1\'-O2\'', 'C4\'-C2\'', 'C1\'-C3\''],
           'same-P-S-0': ['C5\'-C3\'', 'C4\'-O3\'', 'C2\'-O3\'', 'O5\'-C4\'', 'C5\'-O4\''],
           'same-P-S-1': ['C3\'-P'],
           'same-S-B-0': ['C1\'-C6', 'C1\'-C2', 'O4\'-N1', 'N1-C2\'', 'O4\'-N9', 'N9-C2\'', 'C1\'-C8', 'C1\'-C4'],
           'Within-Ring': []}

d_dihedral_angle = {'same-P-P-0': ['O1P-C5\'', 'C5\'-O3\''],
                    'same-P-P-1': ['O3\'-C5\''],
                    'same-S-S-0': ['O4\'-O2\''],
                    'same-P-S-0': ['C5\'-C1\'', 'O5\'-O4\'', 'O2\'-O3\'', 'O5\'-C3\''],
                    'same-P-S-1': ['C3\'-O2P', 'C3\'-O5\''],
                    'same-S-B-0': ['O4\'-C2', 'C8-C2\'', 'C6-C2\'', 'C1\'-N3', 'C1\'-O2', 'C2-C2\'', 
                                   'O4\'-C4', 'N1-C3\'', 'C4\'-N1', 'C4\'-N9', 'O4\'-O2', 'N9-C3\'',
                                   'O4\'-C6', 'O4\'-C8', 'C4-C2\''],
                    'Within-Ring': []}


class Pair:
    def __init__(self, strand_i, resid_i, atomname_i, strand_j, resid_j, atomname_j, bigbig_category_needed=False, n_bp=10):
        self.strand_i = strand_i
        self.resid_i = resid_i
        self.atomname_i = atomname_i
        self.strand_j = strand_j
        self.resid_j = resid_j
        self.atomname_j = atomname_j
        self.n_bp = n_bp
        self.atomtype_i, self.atomtype_j = self.decide_atomtype()
        self.same_strand = self.decide_samestrand()
        self.hb, self.stack = self.decide_hb_stack()
        self.pair_type = self.decide_pairtype()
        self.big_category = self.decide_big_category()
        if self.pair_type in chemicalbond_group:
            self.chemical_bond = self.decide_chemical_bond()
        else:
            self.chemical_bond = False
        if bigbig_category_needed:
            self.bigbig_category = self.decide_bigbig_category()

    def decide_atomtype(self):
        return d_atomcgtype[self.atomname_i], d_atomcgtype[self.atomname_j]

    def decide_samestrand(self):
        if self.strand_i == self.strand_j:
            return True
        else:
            return False

    def decide_hb_stack(self):
        if self.atomtype_i != 'B' or self.atomtype_j != 'B':
            return False, False
        if self.same_strand:
            return False, True
        else:
            return True, False

    def decide_pairtype(self):
        if self.hb:
            first = self.decide_complement_resid(n_bp=self.n_bp)
            if (self.atomname_i in hb_atomtypes) and (self.atomname_j in hb_atomtypes):
                return 'HB-{0}'.format(first)
            else:
                return 'Oppo-Ring-{0}'.format(first)    # opposite ring
        elif self.stack:
            first = abs(self.resid_i - self.resid_j)
            if first == 0:
                return 'Within-Ring'
            else:
                return 'STACK-{0}'.format(first)
        else:
            if self.same_strand:
                first = d_pairtype['{0}-{1}'.format(self.atomtype_i, self.atomtype_j)]
                second = abs(self.resid_i - self.resid_j)
                pairtype = 'same-{0}-{1}'.format(first, second)
                """
                if pairtype == 'same-P-S-1':
                    if self.resid_j < self.resid_i:
                        return 'same-P-S-1-5prime'
                    else:
                        return 'same-P-S-1-3prime'
                elif pairtype == 'same-S-P-1':
                    if self.resid_j > self.resid_i:
                        return 'same-P-S-1-5prime'
                    else:
                        return 'same-P-S-1-3prime'
                else:
                    return pairtype
                """
                return pairtype
            else:
                first = d_pairtype['{0}-{1}'.format(self.atomtype_i, self.atomtype_j)]
                second = self.decide_complement_resid()
                return 'diff-{0}-{1}'.format(first, second)

    def decide_complement_resid(self, n_bp=10):
        complement_id_j = n_bp + 1 - self.resid_i
        if self.resid_j == complement_id_j:
            return 0
        else:
            return abs(self.resid_j-complement_id_j)

    def decide_big_category(self):
        if self.pair_type in interactions['backbone']:
            return 'backbone'
        elif self.pair_type in interactions['sugar']:
            return 'sugar'
        elif self.pair_type in interactions['stack']:
            return 'stack'
        elif self.pair_type in interactions['HB']:
            return 'HB'
        else:
            return 'other'

    def decide_bigbig_category(self):
        if re.match('same-P-P', self.pair_type) or re.match('same-P-S', self.pair_type) or\
                re.match('same-P-B', self.pair_type):
            return 'backbone'
        elif re.match('same-S-S', self.pair_type) or re.match('same-S-B', self.pair_type):
            return 'sugar'
        elif re.match('STACK', self.pair_type):
            return 'stack'
        elif re.match('HB', self.pair_type) or re.match('Oppo-Ring', self.pair_type):
            return 'HB'
        else:
            return 'other'
        
    def decide_chemical_bond(self):
        pair_tuple_1 = f'{self.atomname_i}-{self.atomname_j}'
        pair_tuple_2 = f'{self.atomname_j}-{self.atomname_i}'
        atompairs = d_bond[self.pair_type]
        if (pair_tuple_1 in atompairs) or (pair_tuple_2 in atompairs):
            return True
        else:
            return False
        
d_bond = {'same-P-P-0': ['P-O1P', 'O5\'-C5\'', 'P-O5\'', 'P-O2P'],
          'same-P-P-1': ['O3\'-P'],
          'same-S-S-0': ['C4\'-O4\'', 'C1\'-C2\'', 'C3\'-C2\'', 'O4\'-C1\'', 'C2\'-O2\'', 'C4\'-C3\''],
          'same-P-S-0': ['C5\'-C4\'', 'C3\'-O3\''],
          'same-P-S-1': [],
          'same-S-B-0': ['C1\'-N9', 'C1\'-N1'],
          'Within-Ring': ['C2-O2', 'N9-C4', 'N9-C8', 'C4-N3', 'C2-N2', 'N1-C2', 
                          'N1-C6', 'C6-C5', 'N7-C5', 'C2-N3', 'C5-C4', 'C8-N7',
                          'C6-O6', 'C6-N6', 'C6-N1', 'C5-C6', 'C4-N4', 'N3-C4',
                          'N3-C2', 'C4-O4', 'C5-C5M']}

d_angle = {'same-P-P-0': ['O2P-O5\'', 'O1P-O5\'', 'O5\'-O3\'', 'O1P-O2P', 'P-C5\'', 'O2P-O3\'', 'O1P-O3\''],
           'same-P-P-1': ['O3\'-O5\'', 'O3\'-O1P', 'O3\'-O2P'],
           'same-S-S-0': ['C4\'-C1\'', 'O4\'-C2\'', 'O4\'-C3\'', 'C3\'-O2\'', 'C1\'-O2\'', 'C4\'-C2\'', 'C1\'-C3\''],
           'same-P-S-0': ['C5\'-C3\'', 'C4\'-O3\'', 'C2\'-O3\'', 'O5\'-C4\'', 'C5\'-O4\''],
           'same-P-S-1': ['C3\'-P'],
           'same-S-B-0': ['C1\'-C6', 'C1\'-C2', 'O4\'-N1', 'N1-C2\'', 'O4\'-N9', 'N9-C2\'', 'C1\'-C8', 'C1\'-C4'],
           'Within-Ring': []}

d_dihedral_angle = {'same-P-P-0': ['O1P-C5\'', 'C5\'-O3\''],
                    'same-P-P-1': ['O3\'-C5\''],
                    'same-S-S-0': ['O4\'-O2\''],
                    'same-P-S-0': ['C5\'-C1\'', 'O5\'-O4\'', 'O2\'-O3\'', 'O5\'-C3\''],
                    'same-P-S-1': ['C3\'-O2P', 'C3\'-O5\''],
                    'same-S-B-0': ['O4\'-C2', 'C8-C2\'', 'C6-C2\'', 'C1\'-N3', 'C1\'-O2', 'C2-C2\'', 
                                   'O4\'-C4', 'N1-C3\'', 'C4\'-N1', 'C4\'-N9', 'O4\'-O2', 'N9-C3\'',
                                   'O4\'-C6', 'O4\'-C8', 'C4-C2\''],
                    'Within-Ring': []}
        


