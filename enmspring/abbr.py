class Abbreviation:
    d_map = {'a_tract_21mer': 'A-tract', 'tat_21mer': 'A-junction', 
             'g_tract_21mer': 'G-tract', 'gcgc_21mer': 'CpG', 'atat_21mer': 'TATA'}
    
    @staticmethod
    def get_abbreviation(host):
        return Abbreviation.d_map[host]