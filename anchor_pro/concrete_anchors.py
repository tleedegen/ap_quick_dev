import numpy as np
import pandas as pd
from anchor_pro.anchor_pattern_mixin import AnchorPatternMixin

class ConcreteCMU(AnchorPatternMixin):
    # Adhesive Anchor Suffix Maps
    crack_conditions = {
        'Cracked': 'cr',
        'Uncracked': 'uncr'
    }

    moisture_conditions = {
        'Dry': 'dry',
        'Saturated': 'ws',
        'Water-Filled': 'wf',
        'Submerged': 'sub'
    }

    inspection_conditions = {
        'Periodic': 'per',
        'Continuous': 'cont'
    }

    temp_range = {
        'A': 'A',
        'B': 'B'
    }

    drill_types = {
        'Rotary Hammer Drill with Carbide Bit': 'cb',
        'Rotary Drill with Diamond Core Bit, Un-roughened': 'dbu',
        'Rotary Drill with Diamond Core Bit, Roughened': 'dbr'
    }

    def __init__(self, equipment_data=None, xy_anchors=None, seismic=True, anchor_position='top', base_or_wall='base'):

        self.seismic = seismic
        self.anchor_id = None

        # Equipment Parameters
        self.Bx = None
        self.By = None

        # Concrete Parameters
        self.fc = 4000
        self.lw_factor = None
        self.lw_adjustment = 0.8  # See Table 17.2.4.1
        self.lw_bond_adjustement = 0.6  # See Table 17.2.4.1
        self.lw_factor_a = None
        self.lw_factor_bond_failure = None
        self.cracked_concrete = True
        # self.Ec = None  # not used in calculations
        self.poisson = None
        self.t_slab = np.inf
        self.cx_neg = None  # Edge of concrete relative to bounding box (Bx and By)
        self.cx_pos = None
        self.cy_neg = None
        self.cy_pos = None
        self.edge_ordinates = None
        self.profile = None
        self.anchor_position = anchor_position

        # Anchor Geometry
        self.c_min = None  # Minimum edge distance of all base_anchors used for equipment (not local anchor group only)
        self.s_min = None
        self.cax_neg = None
        self.cax_pos = None
        self.cay_neg = None
        self.cay_pos = None
        self.xy_anchors = xy_anchors  # Coordinates of all base_anchors.
        self.spacing_matrix = None  # s_ij = distance between anchor i and anchor j
        self.groups_matrix = None  # b_ij = True if anchor i is in a group with anchor j

        # Governing Anchor Group Geometry
        self.group_idx = None
        self.xy_group = None  # Coordinates of base_anchors in governing anchor group.
        self.n_anchor = None  # Number of base_anchors in governing group.
        self.centroid = None  # Centroid of governing group
        self.sx = None  # Dictionary of anchor spacing
        self.sy = None  # Dictionary of anchor spacing

        # Mechanical Anchor Properties
        self.fya = None
        self.fua = None
        self.kc = None  # ACI318-19 17.6.2.2.1: 24 for cast-in, 17 for post installed, or per mfr
        self.kc_uncr = None
        self.kc_cr = None
        self.installation_method = None
        self.anchor_type = None  # to include ['Headed Stud', 'Headed Bolt'] as used in ACI 17.6.2.2.3
        self.abrg = None  # Net bearing area of headed stud or anchor
        self.le = None  # 17.7.2.2
        self.da = None
        self.cac = None
        self.esr = None
        self.hef_default = None
        self.K = None
        self.K_cr = None
        self.K_uncr = None
        self.Kv = None  # Shear Stiffness
        self.hmin1 = None  # Minimum member thickness required
        self.hmin2 = None
        self.c1 = None  # Minimum edge distances and spacing
        self.s1 = None
        self.c2 = None
        self.s2 = None

        # Adhesive Anchor Properties
        self.manufacturer = None
        self.product = None
        self.installation_method = None
        self.esr_adhesive = None
        self.bar_id = None
        self.type = None
        self.standard = None
        self.spec = None
        self.grade = None
        self.esr_bar = None
        self.da = None
        self.da_inside = None
        self.Ase = None
        self.Ase_inside = None
        # self.Nsa = None
        self.Nsa_inside = None  # todo: implement use
        # self.Vsa = None
        self.alpha_Vseis = None  # todo: implement use.
        self.alpha_Nseis = None
        # self.kc = None
        self.hef_min = None
        self.hef_max = None
        # self.hef_default = None
        # self.hmin = None
        self.s_min = None
        self.c_min = None
        self.cc_min = None  # Required clear cover
        self.max_torque = None
        self.tau = None
        self.category = None
        self.exp_bond = None  # exponent for strength amplification formula (fc / fc_min) ^ exp_bond
        self.maxtemp_A_short = None
        self.maxtemp_B_short = None
        self.maxtemp_long = None
        self.fc_min = None
        self.fc_max = None

        self.t_work_23 = None
        self.t_init_23 = None
        self.t_full_23 = None
        self.t_work_32 = None
        self.t_init_32 = None
        self.t_full_32 = None
        self.t_work_40 = None
        self.t_init_40 = None
        self.t_full_40 = None
        self.t_work_41 = None
        self.t_init_41 = None
        self.t_full_41 = None
        self.t_work_50 = None
        self.t_init_50 = None
        self.t_full_50 = None
        self.t_work_60 = None
        self.t_init_60 = None
        self.t_full_60 = None
        self.t_work_68 = None
        self.t_init_68 = None
        self.t_full_68 = None
        self.t_work_70 = None
        self.t_init_70 = None
        self.t_full_70 = None
        self.t_work_72 = None
        self.t_init_72 = None
        self.t_full_72 = None
        self.t_work_85 = None
        self.t_init_85 = None
        self.t_full_85 = None
        self.t_work_86 = None
        self.t_init_86 = None
        self.t_full_86 = None
        self.t_work_90 = None
        self.t_init_90 = None
        self.t_full_90 = None
        self.t_work_95 = None
        self.t_init_95 = None
        self.t_full_95 = None
        self.t_work_100 = None
        self.t_init_100 = None
        self.t_full_100 = None
        self.t_work_104 = None
        self.t_init_104 = None
        self.t_full_104 = None
        self.t_work_105 = None
        self.t_init_105 = None
        self.t_full_105 = None
        self.t_work_110 = None
        self.t_init_110 = None
        self.t_full_110 = None

        # Reduction Factors Parameters
        self.phi_saN = None
        self.phi_saV = None
        self.phi_cN = None
        self.phi_cV = None
        self.phi_aN = None  # Factor for adhesive bond strength
        self.phi_seismic = 0.75

        # CALCULATED PARAMETERS
        # Load Demands
        self.Tu = None
        self.Vu = None
        self.Tu_max = None  # Max tension on single anchor in governing group
        self.Vu_max = None  # Max shear on single anchor in governing group
        self.Vu_xp = None  # Max group shear in positive x-direction.
        self.Vu_xp_edge = None  # Max group shear for edge-most anchors
        self.Vu_xn = None  # Etc.
        self.Vu_xn_edge = None
        self.Vu_yp = None
        self.Vu_yp_edge = None
        self.Vu_yn = None
        self.Vu_yn_edge = None

        # Results
        self.Nsa = None
        self.Ncb = None
        self.Np = None
        self.Np_uncr = None
        self.Np_cr = None
        self.Nsb = None
        self.Nsbg = None
        self.Na = None
        self.Vsa = None
        self.Vsa_default = None
        self.Vsa_eq = None
        self.Vcb_xp_edge = None
        self.Vcb_xp_full = None
        self.Vcb_xn_edge = None
        self.Vcb_xn_full = None
        self.Vcb_yp_edge = None
        self.Vcb_yp_full = None
        self.Vcb_yn_edge = None
        self.Vcb_yn_full = None
        self.Vcp = None

        self.DCR_N = None
        self.DCR_V = None
        self.governing_tension_limit = None
        self.governing_shear_limit = None
        self.DCR = None

        self.anchor_forces = None  # Array of anchor tension, vx, vy values at all angles of loading
        self.max_group_forces = None  # Dictionary of arrays of tension, vx, vy values for anchors in governing group
        self.anchor_force_results = None  # Tension and shear in all base_anchors at maximum tension loading condition

        # (b) Tension Breakout Parameters
        self.ca_min = None
        self.bxN = None  # X-dimension of breakout cone
        self.byN = None  # Y-dimension of breakout cone
        self.hef_limit = None
        self.hef = None
        self.Anc = None
        self.Anco = None
        self.Nb = None
        self.ex = None
        self.ey = None
        self.psi_ecNx = None
        self.psi_ecNy = None
        self.psi_ecN = None
        self.psi_edN = None
        self.psi_cN = None
        self.psi_cpN = None

        # (c) Tension Pullout Parameters

        # (d) Side Face Blowout parameters

        # (e) Bond Strength
        self.cna = None
        self.bxNa = None
        self.byNa = None
        self.Anao = None
        self.Ana = None
        self.Nba = None
        self.Na = None
        self.psi_ecNa = None
        self.psi_edNa = None
        self.psi_cpNa = None

        # (f) Anchor Shear

        # (g) Shear Breakout
        vcb_pars_list = ['ca1', 'Avco', 'b', 'ha', 'Avc', 'Vb', 'eV', 'psi_ecV', 'psi_edV', 'psi_hV']
        self.shear_breakout_long_name_to_short_name_map = {'Shear Edge Breakout (X+)': 'xp_edge',
                                    'Shear Breakout (X+)': 'xp_full',
                                    'Shear Edge Breakout (X-)': 'xn_edge',
                                    'Shear Breakout (X-)': 'xn_full',
                                    'Shear Edge Breakout (Y+)': 'yp_edge',
                                    'Shear Breakout (Y+)': 'yp_full',
                                    'Shear Edge Breakout (Y-)': 'yn_edge',
                                    'Shear Breakout (Y-)': 'yn_full'}

        self.vcb_pars = {key: {par: None for par in vcb_pars_list} for label, key in self.shear_breakout_long_name_to_short_name_map.items()}
        self.psi_cV = None  # This applies to all shear breakout cases
        self.governing_shear_breakout_case = None

        # (h) Shear Pryout
        self.kcp = None

        # Limit State Summary Dictionary
        self.limit_states = {
            'Steel Tensile Strength': {'mode': 'Tension',
                                       'applicable': True,
                                       'function': self.check_a_anchor_tension,
                                       'demand': 'Tu_max',
                                       'capacity': 'Nsa',
                                       'phi': 'phi_saN'},
            'Concrete Tension Breakout': {'mode': 'Tension',
                                          'applicable': True,
                                          'function': self.check_b_tension_breakout,
                                          'demand': 'Tu',
                                          'capacity': 'Ncb',
                                          'phi': 'phi_cN'},
            'Anchor Pullout': {'mode': 'Tension',
                               'applicable': True,
                               'function': self.check_c_tension_pullout,
                               'demand': 'Tu',
                               'capacity': 'Np',
                               'phi': 'phi_cN'},
            'Side Face Blowout': {'mode': 'Tension',
                                  'applicable': True,
                                  'function': self.check_d_side_face_blowout,
                                  'demand': 'Tu',
                                  'capacity': 'Nsb',
                                  'phi': 'phi_cN'},
            'Bond Strength': {'mode': 'Tension',
                              'applicable': True,
                              'function': self.check_e_bond_strength,
                              'demand': 'Tu',
                              'capacity': 'Na',
                              'phi': 'phi_aN'},
            'Steel Shear Strength': {'mode': 'Shear',
                                     'applicable': True,
                                     'function': self.check_f_anchor_shear,
                                     'demand': 'Vu_max',
                                     'capacity': 'Vsa',
                                     'phi': 'phi_saV'},
            'Shear Breakout (X+)': {'mode': 'Shear',
                                    'applicable': True,
                                    'function': None,
                                    'demand': 'Vu_xp_edge',
                                    'capacity': 'Vcb_xp_full',
                                    'phi': 'phi_cV'},
            'Shear Edge Breakout (X+)': {'mode': 'Shear',
                                         'applicable': True,
                                         'function': None,
                                         'demand': 'Vu_xp',
                                         'capacity': 'Vcb_xp_edge',
                                         'phi': 'phi_cV'},
            'Shear Breakout (X-)': {'mode': 'Shear',
                                    'applicable': True,
                                    'function': None,
                                    'demand': 'Vu_xn_edge',
                                    'capacity': 'Vcb_xn_edge',
                                    'phi': 'phi_cV'},
            'Shear Edge Breakout (X-)': {'mode': 'Shear',
                                         'applicable': True,
                                         'function': None,
                                         'demand': 'Vu_xn',
                                         'capacity': 'Vcb_xn_full',
                                         'phi': 'phi_cV'},
            'Shear Breakout (Y+)': {'mode': 'Shear',
                                    'applicable': True,
                                    'function': None,
                                    'demand': 'Vu_yp_edge',
                                    'capacity': 'Vcb_yp_edge',
                                    'phi': 'phi_cV'},
            'Shear Edge Breakout (Y+)': {'mode': 'Shear',
                                         'applicable': True,
                                         'function': None,
                                         'demand': 'Vu_yp',
                                         'capacity': 'Vcb_yp_full',
                                         'phi': 'phi_cV'},
            'Shear Breakout (Y-)': {'mode': 'Shear',
                                    'applicable': True,
                                    'function': None,
                                    'demand': 'Vu_yn_edge',
                                    'capacity': 'Vcb_yn_edge',
                                    'phi': 'phi_cV'},
            'Shear Edge Breakout (Y-)': {'mode': 'Shear',
                                         'applicable': True,
                                         'function': None,
                                         'demand': 'Vu_yn',
                                         'capacity': 'Vcb_yn_full',
                                         'phi': 'phi_cV'},
            'Shear Pryout': {'mode': 'Shear',
                             'applicable': True,
                             'function': self.check_h_shear_pryout,
                             'demand': 'Vu',
                             'capacity': 'Vcp',
                             'phi': 'phi_cV'}}

        # Results Dataframe
        self.results = pd.DataFrame(columns=['Limit State',
                                             'Mode',
                                             'Demand',
                                             'Nominal Capacity',
                                             'Reduction Factor',
                                             'Seismic Factor',
                                             'Factored Capacity',
                                             'Utilization']).set_index('Limit State')

        self.spacing_requirements = {'slab_thickness_ok': True,
                                     'edge_and_spacing_ok': True,
                                     'anchor_position_ok': True}

        if equipment_data is not None:
            self.set_data(equipment_data, base_or_wall=base_or_wall)

    def set_data(self, equipment_data, xy_anchors=None, base_or_wall='base'):
        """equipment_data is expected to be series (row) extracted from the relevant dataframe"""

        # Concrete Parameters
        for key in vars(self).keys():
            if key + '_' + base_or_wall in equipment_data.keys():
                setattr(self, key, equipment_data.at[key + '_' + base_or_wall])
            elif key in equipment_data.keys():
                setattr(self, key, equipment_data.at[key])

        if equipment_data['weight_classification' + '_' + base_or_wall] == 'LWC':
            self.lw_factor_a = self.lw_factor * self.lw_adjustment
            self.lw_factor_bond_failure = self.lw_factor * self.lw_bond_adjustement
        else:
            self.lw_factor_a = self.lw_factor
            self.lw_factor_bond_failure = self.lw_factor

        # Replace NaN edge distances with inf
        for attr_name in ['cx_neg', 'cx_pos', 'cy_neg', 'cy_pos']:
            ca_attr = getattr(self, attr_name)
            if ca_attr is None or np.isnan(ca_attr):
                setattr(self, attr_name, np.inf)

        # Anchor Geometry
        if xy_anchors is not None:
            self.xy_anchors = np.array(xy_anchors)

        # Get Inter-Anchor Spacing
        self.spacing_matrix = self.get_anchor_spacing_matrix(self.xy_anchors)
        

        # Global anchor edge distance and min spacing
        self.edge_ordinates = [min(self.xy_anchors[:, 0].min(), -0.5 * equipment_data['Bx']) - self.cx_neg,
                               max(self.xy_anchors[:, 0].max(), 0.5 * equipment_data['Bx']) + self.cx_pos,
                               min(self.xy_anchors[:, 1].min(), -0.5 * equipment_data['By']) - self.cy_neg,
                               max(self.xy_anchors[:, 1].max(), 0.5 * equipment_data['By']) + self.cy_pos]

        edge_distances, c_min = self.get_edge_distances(*self.edge_ordinates,
                                                        self.xy_anchors)
        if len(self.spacing_matrix)>1:
            s_min = np.ma.masked_equal(self.spacing_matrix, 0).min()
        else:
            s_min = np.inf

        self.c_min = c_min
        self.s_min = s_min

    def reset_results(self):
        self.results = pd.DataFrame(columns=['Limit State',
                                             'Mode',
                                             'Demand',
                                             'Nominal Capacity',
                                             'Reduction Factor',
                                             'Seismic Factor',
                                             'Factored Capacity',
                                             'Utilization']).set_index('Limit State')

        self.spacing_requirements = {'slab_thickness_ok': True,
                                     'edge_and_spacing_ok': True,
                                     'anchor_position_ok': True}

        self.DCR = None

    def get_governing_anchor_group(self):
        # Identify Governing Anchor Tension
        idx_anchor_t, idx_theta_t = np.unravel_index(np.argmax(self.anchor_forces[:, :, 0]),
                                                     self.anchor_forces[:, :, 0].shape)
        idx_group = self.groups_matrix[idx_anchor_t]
        _, idx_theta_vxp = np.unravel_index(np.argmax(self.anchor_forces[idx_group, :, 1]),
                                            self.anchor_forces[idx_group, :, 1].shape)
        _, idx_theta_vxn = np.unravel_index(np.argmin(self.anchor_forces[idx_group, :, 1]),
                                            self.anchor_forces[idx_group, :, 1].shape)
        _, idx_theta_vyp = np.unravel_index(np.argmax(self.anchor_forces[idx_group, :, 2]),
                                            self.anchor_forces[idx_group, :, 2].shape)
        _, idx_theta_vyn = np.unravel_index(np.argmin(self.anchor_forces[idx_group, :, 2]),
                                            self.anchor_forces[idx_group, :, 2].shape)

        self.anchor_force_results = self.anchor_forces[:, idx_theta_t, :]

        # Dictionary to store the extracted load cases
        self.max_group_forces = {
            'tension': self.anchor_forces[idx_group, idx_theta_t, :],
            'vxp': self.anchor_forces[idx_group, idx_theta_vxp, :],
            'vxn': self.anchor_forces[idx_group, idx_theta_vxn, :],
            'vyp': self.anchor_forces[idx_group, idx_theta_vyp, :],
            'vyn': self.anchor_forces[idx_group, idx_theta_vyn, :]}

        # Set Anchor-group Coordinates
        self.group_idx = np.where(idx_group)[0]
        self.xy_group = self.xy_anchors[self.groups_matrix[idx_anchor_t]]
        self.n_anchor = self.xy_group.shape[0]
        self.centroid = np.mean(self.xy_group, axis=0)

        # Set edge distances
        [self.cax_neg,
         self.cax_pos,
         self.cay_neg,
         self.cay_pos], self.ca_min = self.get_edge_distances(*self.edge_ordinates, self.xy_group)

        # Set anchor spacing
        """Sets the xy array of anchor locations and calculates other geometric parameters related to the anchor group.
                Returns two dictionaries, sx and sy,
                which provide a list of anchor spacing lengths s between base_anchors at a given x, or y ordinate
                implicit in the formulation is that base_anchors are not in staggered patterns."""
        self.sy = {}
        self.sx = {}
        indices = np.lexsort((self.xy_group[:, 0], self.xy_group[:, 1]))
        sorted_by_x = self.xy_group[indices]

        indices = np.lexsort((self.xy_group[:, 1], self.xy_group[:, 0]))
        sorted_by_y = self.xy_group[indices]

        # Calculate spacing in y-direction
        for x in np.unique(sorted_by_x[:, 0]):
            pts = sorted_by_x[sorted_by_x[:, 0] == x]
            if len(pts) > 1:
                self.sy[x] = np.diff(pts[:, 1])
            else:
                self.sy[x] = np.array([0])

        # Calculate spacing in x-direction
        for y in np.unique(sorted_by_y[:, 1]):
            pts = sorted_by_y[sorted_by_y[:, 1] == y]
            if len(pts) > 1:
                self.sx[y] = np.diff(pts[:, 0])
            else:
                self.sx[y] = np.array([0])

        '''Define Anchor Loading'''
        self.Tu = self.max_group_forces['tension'][:, 0].sum()
        self.Tu = max(self.Tu, 0)
        self.Vu = np.max(
            np.sum((self.anchor_forces[idx_group, :, 1] ** 2 + self.anchor_forces[idx_group, :, 2] ** 2) ** 0.5,
                   axis=0))
        self.Tu_max = self.max_group_forces['tension'][:, 0].max()
        self.Tu_max = max(self.Tu_max, 0)
        self.Vu_max = np.max(
            (self.anchor_forces[idx_group, :, 1] ** 2 + self.anchor_forces[idx_group, :, 2] ** 2) ** 0.5)

        self.Vu_xp = self.max_group_forces['vxp'][:, 1].sum()
        self.Vu_xp_edge = self.max_group_forces['vxp'][self.xy_group[:, 0] == max(self.xy_group[:, 0]), 1].sum()
        self.Vu_xn = -self.max_group_forces['vxn'][:, 1].sum()
        self.Vu_xn_edge = -self.max_group_forces['vxn'][self.xy_group[:, 0] == min(self.xy_group[:, 0]), 1].sum()
        self.Vu_yp = self.max_group_forces['vyp'][:, 2].sum()
        self.Vu_yp_edge = self.max_group_forces['vyp'][self.xy_group[:, 1] == max(self.xy_group[:, 1]), 2].sum()
        self.Vu_yn = -self.max_group_forces['vyn'][:, 2].sum()
        self.Vu_yn_edge = -self.max_group_forces['vyn'][self.xy_group[:, 1] == min(self.xy_group[:, 1]), 2].sum()

        return

    def set_mechanical_anchor_properties(self, anchor_data):
        for key in vars(self).keys():
            if key in anchor_data.keys():
                setattr(self, key, anchor_data.at[key])

        # Check anchor applicability
        if self.profile in ['Slab', 'Wall']:
            self.spacing_requirements['anchor_position_ok'] = anchor_data['slab_ok']
        elif self.profile == 'Filled Deck' and self.anchor_position == 'top':
            self.spacing_requirements['anchor_position_ok'] = anchor_data['deck_top_ok']
        elif self.profile == 'Filled Deck' and self.anchor_position == 'soffit':
            self.spacing_requirements['anchor_position_ok'] = anchor_data['deck_soffit_ok']

        # Interpolate Spacing Limits
        if self.profile in ['Filled Deck']:
            self.hmin1 = anchor_data['hmin1_deck']
            self.hmin2 = anchor_data['hmin2_deck']
            h_vals = [self.hmin1, self.hmin2]
            c1_vals = [anchor_data['c11_deck'], anchor_data['c21_deck']]
            c2_vals = [anchor_data['c12_deck'], anchor_data['c22_deck']]
            s1_vals = [anchor_data['s11_deck'], anchor_data['s21_deck']]
            s2_vals = [anchor_data['s12_deck'], anchor_data['s22_deck']]
            cac_vals = [anchor_data['cac1_deck'], anchor_data['cac2_deck']]
        else:  # self.profile in ['Slab', 'Wall']:
            self.hmin1 = anchor_data['hmin1_slab']
            self.hmin2 = anchor_data['hmin2_slab']
            h_vals = [self.hmin1, self.hmin2]
            c1_vals = [anchor_data['c11_slab'], anchor_data['c21_slab']]
            c2_vals = [anchor_data['c12_slab'], anchor_data['c22_slab']]
            s1_vals = [anchor_data['s11_slab'], anchor_data['s21_slab']]
            s2_vals = [anchor_data['s12_slab'], anchor_data['s22_slab']]
            cac_vals = [anchor_data['cac1_slab'], anchor_data['cac2_slab']]

        self.c1 = np.interp(self.t_slab, h_vals, c1_vals)
        self.s1 = np.interp(self.t_slab, h_vals, s1_vals)
        self.c2 = np.interp(self.t_slab, h_vals, c2_vals)
        self.s2 = np.interp(self.t_slab, h_vals, s2_vals)
        self.cac = np.interp(self.t_slab, h_vals, cac_vals)

        # Used Cracked/Uncracked Properties
        if self.cracked_concrete:
            self.kc = self.kc_cr
            self.Np = self.Np_cr
            self.K = self.K_cr
        else:
            self.kc = self.kc_uncr
            self.Np = self.Np_uncr
            self.K = self.K_uncr

        # Use Seismic/Non-Seismic Properties
        if self.seismic:
            self.Vsa = self.Vsa_eq
        else:
            self.Vsa = self.Vsa_default

        # Find Anchor Groups
        radius = 1.5 * self.hef_default
        self.groups_matrix = self.get_anchor_groups(radius, self.spacing_matrix)

    def set_adhesive_anchor_properties(self, anchor_data,
                                       cracked_condition='Cracked',
                                       moisture_condition='Dry',
                                       inspection_condition='Periodic',
                                       drill='Rotary Hammer Drill with Carbide Bit',
                                       short_temp=60,
                                       long_temp=60):
        # Loop to assign all named parameters
        for key in vars(self).keys():
            if key in anchor_data.keys():
                setattr(self, key, anchor_data.at[key])

        ''' Additional Parameter Parsing Based on Installation Conditions'''
        # Get Suffix Terms
        cracked = ConcreteAnchors.crack_conditions[cracked_condition]
        moisture = ConcreteAnchors.moisture_conditions[moisture_condition]
        inspection = ConcreteAnchors.inspection_conditions[inspection_condition]
        temp = self.get_adhevise_temp_range()
        drill = ConcreteAnchors.drill_types[drill]

        # Effectiveness Factor, kc
        kc_var = '_'.join(['kc', cracked, drill])
        self.kc = anchor_data[kc_var]

        # Embedment Limits
        hef_min_var = '_'.join(['hef_min', drill])
        hef_max_var = '_'.join(['hef_min', drill])
        hef_var = '_'.join(['hef_min', drill])
        self.hef_min = anchor_data[hef_min_var]
        self.hef_max = anchor_data[hef_max_var]
        self.hef_default = anchor_data[hef_var]  # todo: verify design scrip has logic to auto-choose if Hef is nan

        # Edge Distance and Spacing
        cc_var = '_'.join(['cc', drill])
        c_var = '_'.join(['c', drill])
        s_var = '_'.join(['s', drill])
        h_var = '_'.join(['hmin', drill])
        cac_var = '_'.join(['cac', drill])
        self.cc_min = anchor_data[cc_var]
        self.c_min = anchor_data[c_var]
        self.c1 = self.c2 = max([self.c_min, self.cc_min + self.da / 2])
        self.s1 = self.s2 = anchor_data[s_var]
        self.hmin1 = self.hmin2 = anchor_data[h_var]
        self.cac = anchor_data[cac_var]

        # Phi Values
        phi_cN_var = '_'.join(['phi_cN', drill])
        phi_cV_var = '_'.join(['phi_cV', drill])
        self.phi_cN = anchor_data[phi_cN_var]
        self.phi_cV = anchor_data[phi_cV_var]

        # Bond Strength, Tau
        tau_var = '_'.join(['tau', cracked, drill, moisture, inspection, temp])
        self.tau = anchor_data[tau_var]

        # Anchor Category (ACI 355)
        cat_var = '_'.join(['category', drill, moisture, inspection])
        self.category = anchor_data[cat_var]

        # Bond Strength Reduction Factors
        phi_var = '_'.join(['phi', drill, moisture, inspection])
        alpha_var = '_'.join(['alpha_Nseis', drill])
        self.phi_aN = anchor_data[phi_var]
        self.alpha_Nseis = anchor_data[alpha_var]

        # Bond Strength Amplification Factor
        exp_var = '_'.join(['exp_bond', cracked, drill])
        self.exp_bond = anchor_data[exp_var]


    def check_anchor_spacing(self):

        # Check slab depth
        if self.t_slab < self.hmin1:
            self.spacing_requirements['slab_thickness_ok'] = False

        # Check spacing and edge distance
        if self.c_min < self.c1:
            self.spacing_requirements['edge_and_spacing_ok'] = False
        if self.s_min < self.s2:
            self.spacing_requirements['edge_and_spacing_ok'] = False

        if (self.c2, self.s2) != (self.c1, self.s1):
            if self.c_min == self.c1 and self.s_min < self.s1:
                self.spacing_requirements['edge_and_spacing_ok'] = False
            elif self.c_min != self.c1:
                m_min = (self.s2 - self.s1) / (self.c2 - self.c1)
                m_existing = (self.s_min - self.s1) / (self.c_min - self.c1)
                if m_existing < m_min:
                    '''If the slope from the (c1,s1) point to the (ca,smin) point is less than
                    the slop to the (c2,s2) point, the spacing requirements are not met.'''
                    self.spacing_requirements['edge_and_spacing_ok'] = False

    def check_anchor_capacities(self):
        # Verify Applicable Limit States
        if np.isinf(self.cx_pos):
            self.limit_states['Shear Breakout (X+)']['applicable'] = False
            self.limit_states['Shear Edge Breakout (X+)']['applicable'] = False
        if np.isinf(self.cx_neg):
            self.limit_states['Shear Breakout (X-)']['applicable'] = False
            self.limit_states['Shear Edge Breakout (X-)']['applicable'] = False
        if np.isinf(self.cy_pos):
            self.limit_states['Shear Breakout (Y+)']['applicable'] = False
            self.limit_states['Shear Edge Breakout (Y+)']['applicable'] = False
        if np.isinf(self.cy_neg):
            self.limit_states['Shear Breakout (Y-)']['applicable'] = False
            self.limit_states['Shear Edge Breakout (Y-)']['applicable'] = False

        if self.Np is None or np.isnan(self.Np):
            self.limit_states['Anchor Pullout']['applicable'] = False

        if 'Headed' not in self.anchor_type:
            self.limit_states['Side Face Blowout']['applicable'] = False

        if 'Adhesive' not in self.anchor_type:
            self.limit_states['Bond Strength']['applicable'] = False

        self.check_g_shear_breakout()

        # Perform Limit State Checks
        for limit, pars in self.limit_states.items():
            # print(f'Checking {limit}')
            mode = pars['mode']
            if mode == 'Tension' and pars['applicable']:
                pars['function']()
                demand = getattr(self, pars['demand'])
                phi = getattr(self, pars['phi'])
                phi_seismic = 1.0 if limit == 'Steel Tensile Strength' else self.phi_seismic  # 17.10.5.4
                capacity = getattr(self, pars['capacity'])
                dcr = demand / (phi * phi_seismic * capacity)
                self.results.loc[limit] = [mode,
                                           demand,
                                           capacity,
                                           phi,
                                           phi_seismic,
                                           phi * phi_seismic * capacity,
                                           dcr]

            if pars['mode'] == 'Shear' and pars['applicable']:
                if pars['function']:
                    pars['function']()  # this avoids redundantly running shear breakout check.
                demand = getattr(self, pars['demand'])
                phi = getattr(self, pars['phi'])
                phi_seismic = self.phi_seismic
                capacity = getattr(self, pars['capacity'])
                dcr = demand / (phi * capacity)
                self.results.loc[limit] = [mode,
                                           demand,
                                           capacity,
                                           phi,
                                           1.0,
                                           phi * capacity,
                                           dcr]
        self.get_governing_shear_breakout()
        self.get_tension_shear_interaction()

    def get_governing_shear_breakout(self):

        # Find governing breakout dcr
        cases = list(set(self.shear_breakout_long_name_to_short_name_map.keys()) & set(self.results.index))
        if cases:
            governing_name = self.results.loc[cases, 'Utilization'].idxmax()
            self.governing_shear_breakout_case = self.shear_breakout_long_name_to_short_name_map[governing_name]

    def get_tension_shear_interaction(self):
        # Governing Tension Limit
        idt = self.results.loc[self.results['Mode'] == 'Tension', 'Utilization'].idxmax()
        self.DCR_N = self.results.loc[idt, 'Utilization']
        self.governing_tension_limit = idt

        # Governing Shear Limit
        idv = self.results.loc[self.results['Mode'] == 'Shear', 'Utilization'].idxmax()
        self.DCR_V = self.results.loc[idv, 'Utilization']
        self.governing_shear_limit = idv

        # Combined Shear and Tension
        self.DCR = self.DCR_N ** (5 / 3) + self.DCR_V ** (5 / 3)

    # def print_capacities(self):
    #     for limit, pars in self.limit_states.items():
    #         if pars['applicable']:
    #             demand_name = pars['demand']
    #             demand = getattr(self, pars['demand'])
    #             phi_name = pars['phi']
    #             phi = getattr(self, pars['phi'])
    #             capacity_name = pars['capacity']
    #             capacity = getattr(self, pars['capacity'])
    #             dcr = demand / (phi * self.phi_seismic * capacity)
    #             print(f'{limit}: {demand_name}={demand}, {phi_name}={phi}, {capacity_name}={capacity}, DCR={dcr}')

    def check_a_anchor_tension(self):
        if not self.Nsa:
            raise Exception('No tabulated value for Nsa. Need to implement calculations')

    def check_b_tension_breakout(self):  # 17.6.2
        """Calculate properties of anchor or anchor group for tensile limit state calculations"""
        ca_array = np.array([self.cax_neg, self.cax_pos, self.cay_neg, self.cay_pos])

        if sum(ca_array < 1.5 * self.hef_default) >= 3:  # Edge distance < 1.5hef on 3 or more sides
            ca_max = ca_array[ca_array < 1.5 * self.hef_default].max()
            s_max = 0
            for x in [self.xy_group[:, 0].min(), self.xy_group[:, 0].max()]:
                s_max = max([s_max, self.sy[x].max()])
            for y in [self.xy_group[:, 1].min(), self.xy_group[:, 1].max()]:
                s_max = max([s_max, self.sx[y].max()])

            self.hef_limit = max(ca_max / 1.5, s_max / 3)
            self.hef = self.hef_limit
        else:
            self.hef = self.hef_default

        # Breakout Cone Dimensions 17.6.2.1
        self.bxN = min([self.cax_neg, 1.5 * self.hef]) + \
                   (self.xy_group[:, 0].max() - self.xy_group[:, 0].min()) + \
                   min([self.cax_pos, 1.5 * self.hef])

        self.byN = min([self.cay_neg, 1.5 * self.hef]) + \
                   (self.xy_group[:, 1].max() - self.xy_group[:, 1].min()) + \
                   min([self.cay_pos, 1.5 * self.hef])

        self.Anc = self.bxN * self.byN
        self.Anco = 9 * self.hef ** 2

        # Basic Breakout Strength 17.6.2.2
        if all([self.n_anchor == 1,
                self.anchor_type in ['Headed Stud', 'Headed Bolt'],
                11 <= self.hef <= 25]):  # ACI 17.6.2.2.3
            self.Nb = 16 * self.lw_factor_a * (self.fc ** 0.5) * (self.hef ** (5 / 3))
        else:
            self.Nb = self.kc * self.lw_factor_a * self.fc ** 0.5 * self.hef ** 1.5

        # Breakout Eccentricity Factor 17.6.2.3
        t_anchor = self.max_group_forces['tension'][:, 0]
        weights = t_anchor if sum(t_anchor != 0) else np.ones(len(self.xy_group))
        self.ex, self.ey = np.average(self.xy_group, axis=0, weights=weights) - self.centroid

        self.psi_ecNx = min([1 / (1 + self.ex / (1.5 * self.hef)), 1.0])
        self.psi_ecNy = min([1 / (1 + self.ey / (1.5 * self.hef)), 1.0])
        self.psi_ecN = self.psi_ecNx * self.psi_ecNy

        # Breakout Edge Factor 17.6.2.4
        self.psi_edN = min([1.0, 0.7 + 0.3 * self.ca_min / (1.5 * self.hef)])

        # Breakout cracking factor 17.6.2.5
        self.psi_cN = 1.0  # In general this factor will be 1.0, when used with mfr-provided kc values

        # Breakout splitting factor 17.6.2.6
        self.psi_cpN = min([1.0, max([self.ca_min / self.cac, 1.5 * self.hef / self.cac])])

        # Breakout Strength
        self.Ncb = (self.Anc / self.Anco) * \
                   self.psi_ecN * self.psi_edN * self.psi_cN * self.psi_cpN * self.Nb

    def check_c_tension_pullout(self):  # 17.6.3
        # Basic Pullout Strength
        # todo: need to impliment pullout amplification factor (fc/fc_min)**b (Compare ESR and Bond)
        if not self.Np:
            raise Exception('No tabulated value for Np. Need to implement calculations')

    def check_d_side_face_blowout(self):  # 17.6.4
        if min([self.cax_neg, self.cax_pos]) < min([self.cay_neg, self.cay_pos]):
            ca1 = min([self.cax_neg, self.cax_pos])
            ca2 = min([self.cay_neg, self.cay_pos])
            s1 = self.sx
        else:
            ca2 = min([self.cax_neg, self.cax_pos])
            ca1 = min([self.cay_neg, self.cay_pos])
            s1 = self.sy

        factor = max([1.0, min([3.0, ca2 / ca1])])  # 17.6.4.1.1
        self.Nsb = factor * 160 * ca1 * self.abrg ** 0.5 * self.lw_factor_a * self.fc ** 0.5
        self.Nsbg = (1 + s1 / (6 * ca1)) * self.Nsb

    def check_e_bond_strength(self):
        # todo: [Adhesive] add adhesive anchor check
        # Note, will need to adjust lw_factor_a in this calc by using local
        # lw_factor_a = 0.6*self.lw_factor, not self.lw_factor_a
        # See table 17.2.4.1. Lw  factor is specific to bond failure mode.

        # Determine Influence Areas (See ACI Fig R17.6.5.1)

        self.cna = 10 * self.da * (self.tau_anchor / 1100)  # ACI 17.6.5.1.2b
        self.Anao = (2 * self.cna) ** 2  # ACI 17.6.5.1.2a

        self.bxNa = min([self.cax_neg, self.cna]) + \
                    (self.xy_group[:, 0].max() - self.xy_group[:, 0].min()) + \
                    min([self.cax_pos, self.cna])

        self.byNa = min([self.cay_neg, self.cna]) + \
                    (self.xy_group[:, 1].max() - self.xy_group[:, 1].min()) + \
                    min([self.cay_pos, self.cna])

        self.Ana = self.bxNa * self.byNa

        self.Nba = self.lw_factor_bond_failure * self.tau_anchor * np.pi * self.da * self.hef

        self.Na = (self.Ana / self.Anao) * self.psi_ecNa * self.psi_edNa * self.psi_cpNa * self.Nba

        pass

    def check_f_anchor_shear(self):
        if not self.Vsa:
            raise Exception('No tabulated value for Vsa. Need to implement calculations')
        # todo: [Adhesive Anchors] Inclusion of 17.7.1.2.1 anchor shear (for non-tabulated Vsa)

    def check_g_shear_breakout(self):

        # Helper function for eccentricity Calculation
        def get_psi_ecv(coordinates, forces, ca1):
            centroid = np.mean(coordinates)
            zero_x = np.sum(forces, axis=0) == 0
            weights = forces if not zero_x else np.ones(len(coordinates))
            ecc = np.average(coordinates, axis=0, weights=weights) - centroid
            psi_ecv = 1 / (1 + ecc / (1.5 * ca1))
            return ecc, psi_ecv

        ha = self.t_slab  # Slab thickness

        # Assign ca1 values for each case
        # X-direction
        xn_edge = min(self.xy_group[:, 0])
        xn_full = max(self.xy_group[:, 0])
        self.vcb_pars['xn_edge']['ca1'] = self.cax_neg
        self.vcb_pars['xn_edge']['ca2+'] = self.cay_pos
        self.vcb_pars['xn_edge']['ca2-'] = self.cay_neg

        self.vcb_pars['xn_full']['ca1'] = self.cax_neg + (self.xy_group[:, 0].max() - self.xy_group[:, 0].min())
        self.vcb_pars['xn_full']['ca2+'] = self.cay_pos
        self.vcb_pars['xn_full']['ca2-'] = self.cay_neg

        xp_edge = max(self.xy_group[:, 0])
        xp_full = min(self.xy_group[:, 0])
        self.vcb_pars['xp_edge']['ca1'] = self.cax_pos
        self.vcb_pars['xp_edge']['ca2+'] = self.cay_pos
        self.vcb_pars['xp_edge']['ca2-'] = self.cay_neg

        self.vcb_pars['xp_full']['ca1'] = self.cax_pos + (self.xy_group[:, 0].max() - self.xy_group[:, 0].min())
        self.vcb_pars['xp_full']['ca2+'] = self.cay_pos
        self.vcb_pars['xp_full']['ca2-'] = self.cay_neg

        # Y-direction
        yn_edge = min(self.xy_group[:, 1])
        yn_full = max(self.xy_group[:, 1])
        self.vcb_pars['yn_edge']['ca1'] = self.cay_neg
        self.vcb_pars['yn_edge']['ca2+'] = self.cax_pos
        self.vcb_pars['yn_edge']['ca2-'] = self.cax_neg

        self.vcb_pars['yn_full']['ca1'] = self.cay_neg + (self.xy_group[:, 1].max() - self.xy_group[:, 1].min())
        self.vcb_pars['yn_full']['ca2+'] = self.cax_pos
        self.vcb_pars['yn_full']['ca2-'] = self.cax_neg

        yp_edge = max(self.xy_group[:, 1])
        yp_full = min(self.xy_group[:, 1])
        self.vcb_pars['yp_edge']['ca1'] = self.cay_pos
        self.vcb_pars['yp_edge']['ca2+'] = self.cax_pos
        self.vcb_pars['yp_edge']['ca2-'] = self.cax_neg

        self.vcb_pars['yp_full']['ca1'] = self.cay_pos + (self.xy_group[:, 1].max() - self.xy_group[:, 1].min())
        self.vcb_pars['yp_full']['ca2+'] = self.cax_pos
        self.vcb_pars['yp_full']['ca2-'] = self.cax_neg

        breakout_cases = ['xp_edge', 'xp_full', 'xn_edge', 'xn_full', 'yp_edge', 'yp_full', 'yn_edge', 'yn_full']
        ordinates = [xp_edge, xp_full, xn_edge, xn_full, yp_edge, yp_full, yn_edge, yn_full]
        load_cases = ['vxp', 'vxp', 'vxn', 'vxn', 'vyp', 'vyp', 'vyn', 'vyn']
        for case, ordinate, load_case in zip(breakout_cases, ordinates, load_cases):
            direction = case[0]
            # Total spacing of bolts parallel to breakout edge
            perpendicular_index = 0 if direction == 'x' else 1
            parallel_index = 1 if direction == 'x' else 0
            self.vcb_pars[case]['s_total'] = self.xy_group[self.xy_group[:,
                                                           perpendicular_index] == ordinate, parallel_index].max() - \
                                             self.xy_group[self.xy_group[:,
                                                           perpendicular_index] == ordinate, parallel_index].min()

            # Calculate Avco, b, ha, Avc, Vb
            # Maximum anchor spacing perpendicular to breakout direction
            s_list = self.sy[ordinate] if direction == 'x' else self.sx[ordinate]
            s_max = max(s_list)

            # Adjust ca1 per 17.7.2.1
            # todo: [QA] Review report that shear area is calculated correctly
            if all(np.array([ha, self.vcb_pars[case]['ca2+'],
                             self.vcb_pars[case]['ca2-']]) < 1.5 * self.vcb_pars[case]['ca1']):
                self.vcb_pars[case]['ca1'] = max(ha / 1.5, self.vcb_pars[case]['ca2+'] / 1.5,
                                                 self.vcb_pars[case]['ca2-'] / 1.5, s_max / 3)

            # Adjustments for b and ha
            reduction = 0
            for s in s_list:
                reduction += max(0, (s - 3 * self.vcb_pars[case]['ca1']))

            # Calculate Avco per 17.7.2.1.3
            self.vcb_pars[case]['Avco'] = 4.5 * self.vcb_pars[case]['ca1'] ** 2

            self.vcb_pars[case]['b'] = self.vcb_pars[case]['s_total'] + \
                                       min(self.vcb_pars[case]['ca2+'], 1.5 * self.vcb_pars[case]['ca1']) + \
                                       min(self.vcb_pars[case]['ca2-'], 1.5 * self.vcb_pars[case]['ca1']) - reduction
            self.vcb_pars[case]['ha'] = min(ha, 1.5 * self.vcb_pars[case]['ca1'])
            self.vcb_pars[case]['Avc'] = self.vcb_pars[case]['b'] * self.vcb_pars[case]['ha']

            # Basic shear breakout strength
            self.vcb_pars[case]['Vb'] = min([(7 * (self.le / self.da) ** 0.2 * self.da ** 0.5), 9]) * \
                                        self.lw_factor_a * self.fc ** 0.5 * self.vcb_pars[case]['ca1'] ** 1.5

            # Breakout Eccentricity Factor 17.7.2.3
            coords = self.xy_group[:, 0 if direction == 'x' else 1]
            forces = self.max_group_forces[load_case][:, 0 if direction == 'x' else 1]
            if 'edge' in case:
                forces = forces[coords == ordinate]
                coords = coords[coords == ordinate]

            eccentricity, psi_ecV = get_psi_ecv(coords, forces, self.vcb_pars[case]['ca1'])
            self.vcb_pars[case]['eV'] = eccentricity
            self.vcb_pars[case]['psi_ecV'] = psi_ecV

            # Breakout Edge Effect Factor 17.7.2.4
            numerator = min(self.vcb_pars[case]['ca2+'], self.vcb_pars[case]['ca2-'])
            denominator = 1.5 * self.vcb_pars[case]['ca1']
            if np.isinf(numerator) and np.isinf(denominator):  # Handle the division, considering possible infinities
                ratio = np.nan  # Both are infinite, so the ratio is undefined (nan)
            elif np.isinf(denominator) and not np.isinf(numerator):
                ratio = 0.0  # Finite numerator over infinite denominator -> ratio is 0
            else:
                ratio = numerator / denominator

            # Compute the final expression, handling the result of the ratio
            self.vcb_pars[case]['psi_edV'] = min(1.0, 0.7 + 0.3 * ratio)

            # Breakout Cracking Factor 17.7.2.5
            self.vcb_pars[case]['psi_cV'] = 1.0
            # Todo: [Calc Refinement] This factor taken as 1.0, conservatively.
            #  possible to justify higher value with additional user inputs.

            # Breakout Thickness Factor 17.7.2.6
            self.vcb_pars[case]['psi_hV'] = max((1.5 * self.vcb_pars[case]['ca1'] / self.vcb_pars[case]['ha']) ** 0.5,
                                                1)

            # Breakout Capacity
            if np.isinf(self.vcb_pars[case]['Avc']) and np.isinf(self.vcb_pars[case]['Avco']):
                self.vcb_pars[case]['Vcb'] = np.nan
            else:
                self.vcb_pars[case]['Vcb'] = (self.vcb_pars[case]['Avc'] / self.vcb_pars[case]['Avco']) * \
                                             self.vcb_pars[case]['psi_ecV'] * \
                                             self.vcb_pars[case]['psi_edV'] * \
                                             self.vcb_pars[case]['psi_cV'] * \
                                             self.vcb_pars[case]['psi_hV'] * \
                                             self.vcb_pars[case]['Vb']

        # Extract breakout capacities for use with results dataframe
        self.Vcb_xp_full = self.vcb_pars['xp_full']['Vcb']
        self.Vcb_xn_full = self.vcb_pars['xn_full']['Vcb']
        self.Vcb_yp_full = self.vcb_pars['yp_full']['Vcb']
        self.Vcb_yn_full = self.vcb_pars['yn_full']['Vcb']
        self.Vcb_xp_edge = self.vcb_pars['xp_edge']['Vcb']
        self.Vcb_xn_edge = self.vcb_pars['xn_edge']['Vcb']
        self.Vcb_yp_edge = self.vcb_pars['yp_edge']['Vcb']
        self.Vcb_yn_edge = self.vcb_pars['yn_edge']['Vcb']

    def check_h_shear_pryout(self):
        self.kcp = 1.0 if self.hef < 2.5 else 2.0
        self.Vcp = self.kcp * self.Ncb
        # todo: [Adhesive Anchors] verify pry-out for adhesive base_anchors per 17.7.3.1.1

    @staticmethod
    def get_edge_distances(x_neg, x_pos, y_neg, y_pos, xy_points):
        """Takes coordinates of four edges, and an array of xy points
        Returns four edge distances"""
        xmin = xy_points[:, 0].min()
        xmax = xy_points[:, 0].max()
        ymin = xy_points[:, 1].min()
        ymax = xy_points[:, 1].max()

        cax_neg = xmin - x_neg
        cax_pos = x_pos - xmax
        cay_neg = ymin - y_neg
        cay_pos = y_pos - ymax
        edge_distances = np.array([cax_neg, cax_pos, cay_neg, cay_pos])
        ca_min = edge_distances.min()
        return edge_distances, ca_min

class CMUAnchors(ConcreteCMU):
    def __init__(self, equipment_data=None, xy_anchors=None, seismic=True, anchor_position='top', base_or_wall='base'):
        super().__init__(equipment_data, xy_anchors, seismic, anchor_position=anchor_position, base_or_wall=base_or_wall)

class ConcreteAnchors(ConcreteCMU):
    def __init__(self, equipment_data=None, xy_anchors=None, seismic=True, anchor_position='top', base_or_wall='base'):
        super().__init__(equipment_data, xy_anchors, seismic, anchor_position=anchor_position,
                         base_or_wall=base_or_wall)