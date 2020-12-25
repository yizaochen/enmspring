import numpy as np
import networkx as nx

class Atom:
    def __init__(self, name, resid, cgid, positions):
        self.name = name
        self.resid = resid
        self.cgid = cgid
        self.positions = positions


class THY_Base:
    edge_list = [('C4', 'C5'), ('C5', 'C6'), ('C6', 'N1'), 
                 ('N1', 'C2'), ('C2', 'N3'), ('N3', 'C4'), 
                 ('C5', 'C7'), ('C2', 'O2'), ('C4', 'O4')]

    def __init__(self, radius):
        self.radius = radius

        self.name_list_1 = ['C4', 'C5', 'C6', 'N1', 'C2', 'N3']
        self.name_list_2 = ['C7', 'O2', 'O4']
        self.name_list = self.name_list_1 + self.name_list_2

        self.d_nodes = self.get_d_nodes()

    def get_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.name_list)
        return G

    def translate_xy(self, x, y):
        for name in self.name_list:
            self.d_nodes[name][0] -= x
            self.d_nodes[name][1] -= y

    def translate_xy_sele_nodes(self, name, x, y, d_nodes):
        d_nodes[name][0] -= x
        d_nodes[name][1] -= y
        return d_nodes

    def get_d_nodes(self):
        d_nodes = self.get_kernel()
        d_nodes = self.get_branch(d_nodes)
        return d_nodes

    def get_xy(self, angle_degree):
        angle_radian = np.deg2rad(angle_degree)
        x = self.radius * np.cos(angle_radian)
        y = self.radius * np.sin(angle_radian)
        return [x, y]

    def get_xy_by_radius(self, angle_degree, radius):
        angle_radian = np.deg2rad(angle_degree)
        x = radius * np.cos(angle_radian)
        y = radius * np.sin(angle_radian)
        return [x, y]

    def get_kernel(self):
        d_nodes = dict()
        angle = 30
        incr_angle = 60
        for name in self.name_list_1:
            d_nodes[name] = self.get_xy(angle)
            angle += incr_angle
        return d_nodes

    def get_branch(self, d_nodes):
        name = 'C7'
        d_nodes[name] = self.get_xy(90)
        d_nodes[name][1] += self.radius

        name = 'O2'
        d_nodes[name] = self.get_xy(270)
        d_nodes[name][1] -= self.radius

        name = 'O4'
        d_nodes[name] = self.get_xy_by_radius(30, 2 * self.radius)
        return d_nodes


class CYT_Base(THY_Base):
    edge_list = [('C4', 'C5'), ('C5', 'C6'), ('C6', 'N1'), 
                 ('N1', 'C2'), ('C2', 'N3'), ('N3', 'C4'), 
                 ('C2', 'O2'), ('C4', 'N4')]

    def __init__(self, radius):
        self.radius = radius

        self.name_list_1 = ['C4', 'C5', 'C6', 'N1', 'C2', 'N3']
        self.name_list_2 = ['O2', 'N4']
        self.name_list = self.name_list_1 + self.name_list_2

        self.d_nodes = self.get_d_nodes()

    def get_branch(self, d_nodes):
        name = 'O2'
        d_nodes[name] = self.get_xy(270)
        d_nodes[name][1] -= self.radius

        name = 'N4'
        d_nodes[name] = self.get_xy_by_radius(30, 2 * self.radius)
        return d_nodes



class ADE_Base(THY_Base):
    edge_list = [('N1', 'C6'), ('C6', 'C5'), ('C5', 'C4'), 
                 ('C4', 'N3'), ('N3', 'C2'), ('C2', 'N1'), 
                 ('C6', 'N6'), ('C5', 'N7'), ('N7', 'C8'),
                 ('C8', 'N9'), ('N9', 'C4')]

    def __init__(self, radius):
        self.radius = radius

        self.name_list_1 = ['N1', 'C6', 'C5', 'C4', 'N3', 'C2']
        self.name_list_2 = ['N6']
        self.name_list_3 = ['N7', 'C8', 'N9']
        self.name_list = self.name_list_1 + self.name_list_2 + self.name_list_3

        self.d_nodes = self.get_d_nodes()

    def get_branch(self, d_nodes):
        name = 'N6'
        d_nodes[name] = self.get_xy(90)
        d_nodes[name][1] += self.radius

        angle = 90
        incr_angle = 72
        for name in self.name_list_3:
            d_nodes[name] = self.get_xy(angle)
            angle += incr_angle

        temp = self.get_xy(18)
        x_move = temp[0] - d_nodes['C5'][0]
        y_move = temp[1] - d_nodes['C5'][1]
        for name in self.name_list_3:
            d_nodes = self.translate_xy_sele_nodes(name, x_move, y_move, d_nodes)
        return d_nodes


class GUA_Base(THY_Base):
    edge_list = [('N1', 'C6'), ('C6', 'C5'), ('C5', 'C4'), 
                 ('C4', 'N3'), ('N3', 'C2'), ('C2', 'N1'), 
                 ('C6', 'O6'), ('C2', 'N2'), ('C5', 'N7'),
                 ('N7', 'C8'), ('C8', 'N9'), ('N9', 'C4')]

    def __init__(self, radius):
        self.radius = radius

        self.name_list_1 = ['N1', 'C6', 'C5', 'C4', 'N3', 'C2']
        self.name_list_2 = ['O6', 'N2']
        self.name_list_3 = ['N7', 'C8', 'N9']
        self.name_list = self.name_list_1 + self.name_list_2 + self.name_list_3

        self.d_nodes = self.get_d_nodes()

    def get_branch(self, d_nodes):
        name = 'O6'
        d_nodes[name] = self.get_xy(90)
        d_nodes[name][1] += self.radius

        name = 'N2'
        d_nodes[name] = self.get_xy_by_radius(330, 2 * self.radius)

        angle = 90
        incr_angle = 72
        for name in self.name_list_3:
            d_nodes[name] = self.get_xy(angle)
            angle += incr_angle

        temp = self.get_xy(18)
        x_move = temp[0] - d_nodes['C5'][0]
        y_move = temp[1] - d_nodes['C5'][1]
        for name in self.name_list_3:
            d_nodes = self.translate_xy_sele_nodes(name, x_move, y_move, d_nodes)
        return d_nodes

class dsDNA:
    sequence = 'AAAAAAAAAAAAAAAAAAAAA'
    d_atcg = {'A': ADE_Base, 'T': THY_Base, 'C': CYT_Base, 'G': GUA_Base}

    def __init__(self, host, n_bp, radius):
        self.host = host
        self.n_bp = n_bp
        self.radius = radius

        self.base_container = self.get_base_container()

    def get_base_container(self):
        base_container = dict()
        y_move = 0
        for bp_id in range(1, self.n_bp+1):
            nt = self.sequence[bp_id-1]
            base_container[bp_id] = self.d_atcg[nt](self.radius)
            base_container[bp_id].translate_xy(0, y_move)
            y_move += -10
        return base_container
