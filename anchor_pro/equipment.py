import numpy as np
import math
import multiprocessing as mp

import os
import json
import pandas as pd
from scipy.optimize import root, broyden1, newton_krylov




# import warnings
from functools import partial

from anchor_pro.utilities import Utilities
from anchor_pro.concrete_anchors import ConcreteCMU, ConcreteAnchors, CMUAnchors
from anchor_pro.elements.wood_fasteners import WoodFastener


def root_solver_worker(queue, residual_func, u_init, p, root_kwargs):
    try:
        res = root(residual_func, u_init, args=(p,), **root_kwargs)
        queue.put(res)
    except Exception as e:
        queue.put(e)

def solve_equilibrium_with_timeout(residual_func, u_init, p, timeout=5, **root_kwargs):
    queue = mp.Queue()
    process = mp.Process(target=root_solver_worker, args=(queue, residual_func, u_init, p, root_kwargs))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return np.zeros_like(u_init), False

    if not queue.empty():
        result = queue.get()
        if isinstance(result, Exception):
            raise result
        return result.x, result.success
    else:
        print('Analysis timeout occurred')
        return np.zeros_like(u_init), False

# @dataclass(slots=True)
# class ModelElements:
#     floor_plates: Optional[List[FloorPlateElement]]
#     base_anchors: Optional[Union[WoodFastener, ConcreteAnchors]]
#     base_straps: Optional[List[BaseStrap]]
#
#     wall_brackets: Optional[List[WallBracketElement]]
#     wall_backing: Optional[List[WallBackingElement]]
#     wall_anchors: Optional[List[SupportedWallAnchors]]
#
#     # Old map giving backing elements for each wall. May not be necessary anymore
#     #backing_indices = {'X+': [], 'X-': [], 'Y+': [], 'Y-': []}
#     self.wall_anchors = {'X+': None, 'X-': None, 'Y+': None, 'Y-': None}

def get_max_demand(el_vs_theta_matrix):
    idx_anchor, idx_theta = np.unravel_index(np.argmax(el_vs_theta_matrix), el_vs_theta_matrix.shape)
    return idx_anchor, idx_theta

class EquipmentModel:
    WALL_NORMAL_VECTORS = {'X+': (-1, 0, 0),
                           'X-': (1, 0, 0),
                           'Y+': (0, -1, 0),
                           'Y-': (0, 1, 0)}

    ATTACHMENT_NORMAL_VECTORS = {'X+': (1, 0, 0),
                                 'X-': (-1, 0, 0),
                                 'Y+': (0, 1, 0),
                                 'Y-': (0, -1, 0),
                                 'Z+': (0, 0, 1),
                                 'Z-': (0, 0, -1)}


    def __init__(self):
        '''User Input Parameters'''
        # Identification Parameters
        self.equipment_id = None
        self.group = None
        self.equipment_type = None

        # Code parameters
        self.code_edition = None
        self.factor_type = 'LRFD'  # todo: [USER-INPUT] read these values from project info input table

        # Global Geometry Parameters
        self.Wp = None  # Item Weight (lbs)
        self.Bx = None  # Item width (in X)
        self.By = None  # Item depth (in Y)
        self.H = None  # Item Height (in Z)
        self.zCG = None  # Height of center of gravity
        self.ex = None  # Eccentricity of COG from assumed origin
        self.ey = None  # Eccentricity of COG from assumed origin
        self.installation_type = None

        # Code Parameters for Seismic Force Calculations
        self.code_pars = {}
        self.include_overstrength = False

        # Hardware Attachement Parameters
        self.gauge_equip = None
        self.fy_equip = None
        self.cxn_sms_id = None

        # Base Attachment Parameters
        self.profile_base = None  # Profile of concrete member at base
        self.base_material = None

        self.base_anchor_id = None
        self.base_anchor_group = None

        # Wall Attachment Parameters
        self.wall_type = None
        self.profile_wall = None  # Profile of concrete wall member
        self.bracket_group = None
        self.bracket_id = None
        self.wall_anchor_group = None  # Concrete/CMU Anchors
        self.wall_anchor_id = None
        self.wall_fastener_group = None  # Wood Fasteners
        self.wall_fastener_id = None
        self.wall_sms_id = None  # Sheet Metal Screws
        self.wall_offsets = {'X+': None, 'X-': None, 'Y+': None, 'Y-': None}

        '''CALCULATED PARAMETERS'''
        # Seismic force results
        self.Fp_code = None
        self.Fp_max = None
        self.Fp_min = None
        self.Fp_dynamic = None
        self.Fp = None

        # Factored load parameters
        self.Eh = None
        self.Emh = None
        self.Ev = None
        self.Fuh = None  # Ultimate Loads [LRFD]
        self.Fuv_max = None
        self.Fuv_min = None
        self.Fah = None  # Applied Loads [ASD]
        self.Fav_max = None
        self.Fav_min = None
        self.Fv_min = None  # The applicable Fv_min using LRFD or ASD
        self.Fv_max = None  # The applicable Fv_max using LRFD or ASD
        self.Fv = None  # Applicable (governing) vertical load
        self.Fh = None  # The applicable Fh using LRFD or ASD
        self.asd_lrfd_ratio = None  # ASD/LRFD ratio

        # Model Attachment Elements
        self.floor_plates = []
        self.base_anchors = None
        self.base_straps = []
        self.base_strap = None
        self._num_base_anchors_in_tension = 0  # Used for stiffness update logic in solver

        self.wall_brackets = []
        self.wall_bracket_forces = None
        self.wall_backing = []
        self.backing_indices = {'X+': [], 'X-': [], 'Y+': [], 'Y-': []}
        self.wall_anchors = {'X+': None, 'X-': None, 'Y+': None, 'Y-': None}

        # Analysis Parameters
        self.omit_analysis = False
        self.model_unstable = False
        self.n_dof = None
        self.disp_dofs = []
        self.floor_plate_dofs = None  # DOF connectivity array for floor plate elements
        self.K = None
        self._u_previous = None
        self.converged = []
        self.cor_coordinates = None  # Center of Rigidity Coordinates Dictionary
        self._u_full = None
        self.u_init = None
        self.residual_call_count = None

        # Results
        self.theta_z = None  # List of angles at which horizontal load is applied
        self.equilibrium_solutions = None  # Array of equilibrium solutions for all angles theta_z
        self.sol = None
        self.theta_z_max = None
        self.governing_solutions = {'base_anchor_tension': {'sol': None, 'theta_z': None},
                                    'base_anchor_shear': {'sol': None, 'theta_z': None},
                                    'wall_bracket_tension': {'sol': None, 'theta_z': None},
                                    'wall_bracket_shear': {'sol': None, 'theta_z': None},
                                    'wall_anchor_tension': {'sol': None, 'theta_z': None},
                                    'wall_anchor_shear': {'sol': None, 'theta_z': None}}

        # Report Parameters
        self.frontmatter_file = None
        self.endmatter_file = None
        self.include_pull_test = False
        self.omit_bracket_output = False

    def set_model_data(self, project_info, equipment_data, df_fasteners=None, df_sms=None):
        """Populates instance attributes from input panda series.
        The input excel_tables is read from the user-input spreadsheet and parsed using the ProjectController object."""

        # Load Project Parameters
        self.code_edition = project_info['code_edition']

        # Load Equipment Parameters
        for key in vars(self).keys():
            if key in equipment_data.index:
                setattr(self, key, equipment_data.at[key])

        # Manual Corrections
        if self.ex is None or np.isnan(self.ex):
            self.ex = 0
        if self.ey is None or np.isnan(self.ey):
            self.ey = 0

        # Append full file path to front and end matter references:
        self.frontmatter_file = os.path.join(project_info['auxiliary_folder'],
                                             self.frontmatter_file) if self.frontmatter_file else None
        if self.frontmatter_file and not os.path.exists(self.frontmatter_file):
            raise Exception(f'Frontmatter file {self.frontmatter_file} not found.')

        self.endmatter_file = os.path.join(project_info['auxiliary_folder'],
                                           self.endmatter_file) if self.endmatter_file else None
        if self.endmatter_file and not os.path.exists(self.endmatter_file):
            raise Exception(f'Endmatter file {self.endmatter_file} not found.')

        # Load Code-Specific Parameters
        if self.code_edition == 'CBC 1998, 16B':
            self.code_pars['Ip'] = project_info['Ip']
            self.code_pars['Z'] = project_info['Z']
            self.code_pars['Cp'] = equipment_data['Cp']  # CBC 1998, 16B parameter
            self.code_pars['cp_amplification'] = equipment_data['cp_amplification']
            self.code_pars['cp_category'] = equipment_data['cp_category']
            self.code_pars['below_grade'] = equipment_data['below_grade']
            self.code_pars['grade_factor'] = 1 if not self.code_pars['below_grade'] else 2 / 3
            self.code_pars['Cp_eff'] = None
            self.include_overstrength = False
        elif self.code_edition == 'ASCE 7-16':
            self.code_pars['ap'] = equipment_data['ap']
            self.code_pars['Rp'] = equipment_data['Rp']
            self.code_pars['Ip'] = equipment_data['Ip']
            self.code_pars['sds'] = project_info['sds']
            self.code_pars['z'] = equipment_data['z']
            self.code_pars['h'] = equipment_data['building_height']
            self.code_pars['omega'] = equipment_data['omega']
            self.code_pars['use_dynamic'] = False  # todo: [Future Feature] add a use dynamic toggle in workbook
            self.code_pars['ai'] = None
            self.code_pars['Ax'] = None
        elif self.code_edition == 'ASCE 7-22 OPM':
            self.code_pars['Cpm'] = equipment_data['Cpm']
            self.code_pars['Cv'] = equipment_data['Cv']
            self.code_pars['omega'] = equipment_data['omega_opm']

        # Base Anchored and Wall Bracket Installation Type
        if self.installation_type in ['Wall Brackets', 'Base Anchored']:
            # Create Floor Plate Elements
            base_geometry_json = equipment_data['Pattern Definition_base']
            if not isinstance(base_geometry_json,str):
                print(f"WARNING: {self.equipment_id} base geometry {equipment_data['base_geometry']} not found.")
            else:
                E_base = None
                poisson = None

                if self.base_material == 'Concrete':
                    E_base = equipment_data['Ec_base']
                    poisson = equipment_data['poisson_base']
                elif self.base_material == 'Wood':
                    E_base = equipment_data['E_wood_base']
                    poisson = E_base/(2*equipment_data['G_wood']) - 1
                else:
                    raise Exception(f'Specified base material {self.base_material} not supported.')

                self.floor_plate_elements_from_json(base_geometry_json, E_base, poisson, df_fasteners, df_sms)

                # Create Base Anchor Element
                xy_anchors_list = [plate.xy_anchors for plate in self.floor_plates if plate.xy_anchors.size > 0]

                if xy_anchors_list:  # Ensure list is not empty before concatenation
                    xy_anchors = np.concatenate(xy_anchors_list, axis=0)
                else:
                    xy_anchors = np.array([])

                if len(xy_anchors) > 0:
                    if self.base_material == 'Concrete':
                        self.include_overstrength = True
                        self.base_anchors = ConcreteAnchors(equipment_data, xy_anchors, base_or_wall='base')
                    elif self.base_material == 'Wood':
                        self.include_overstrength = False
                        self.base_anchors = WoodFastener(xy_anchors) # todo: [WOOD] Finish this
                        self.base_anchors.set_member_properties_from_data_table(equipment_data,base_or_wall='base')
                        t_steel = 0.0478  # todo [WOOD, base plate thickness] for now, this is hard-coded. Need to add input for thickness of base material.
                        Fse = 33
                        self.base_anchors.set_steel_props(t_steel, Fse)
        # Wall Bracket and Wall Mounted Installation Type
        if self.installation_type in ['Wall Brackets', 'Wall Mounted']:
            # Create Wall Bracket Elements
            self.wall_bracket_backing_anchors_from_json(equipment_data, df_fasteners, df_sms)

            for position, wall_anchors in self.wall_anchors.items():
                if isinstance(wall_anchors, ConcreteCMU):
                    wall_anchors.set_data(equipment_data=equipment_data,
                                          xy_anchors=wall_anchors.xy_anchors,
                                          base_or_wall='wall')

    def floor_plate_elements_from_json(self, json_string, E_base, poisson, df_fasteners, df_sms):
        """Creates floor plate elements based on a json string provided by the user-input excel form"""
        orientations = {'X+': 0.0, 'X-': np.pi, 'Y+': np.pi / 2, 'Y-': 3 * np.pi / 2}
        pattern = json.loads(json_string)

        selected_straps = {item['element']['straps']['strap'] for item in pattern if item['element']['straps']['strap'] != 'null'}
        if len(selected_straps) == 0:
            self.base_strap = None
        elif len(selected_straps) > 1:
            print(
                'Warning: user should not specify different strap types for base plates in a single geometry pattern.')
            self.base_strap = selected_straps.pop()
        else:
            self.base_strap = selected_straps.pop()

        for item in pattern:
            element = item['element']
            shape = element['shape']
            anchors = element['anchors']
            straps = element['straps']
            layout = item['layout']
            shape_points = [[float(x) + float(xr) * self.Bx, float(y) + float(yr) * self.By] for x, y, xr, yr in
                            zip(shape['X'], shape['Y'], shape['Xr'], shape['Yr'])]
            anchor_points = [[float(x) + float(xr) * self.Bx, float(y) + float(yr) * self.By] for x, y, xr, yr in
                             zip(anchors['X'], anchors['Y'], anchors['Xr'], anchors['Yr'])]
            strap_geometry = [[float(x), float(y), float(z), float(dx), float(dy), float(dz)] for x, y, z, dx, dy, dz in
                              zip(straps['X'], straps['Y'], straps['Z'], straps['DX'], straps['DY'], straps['DZ']) if
                              self.base_strap]

            # Determine if releases are present
            release_keys = ['release_mx', 'release_my', 'release_mz',
                            'release_xp', 'release_xn', 'release_yp', 'release_yn', 'release_zp', 'release_zn']

            # Check if any value is True for the specified keys
            releases_present = any(value for key in release_keys for value in layout.get(key, []))

            if releases_present or element['check_fasteners']:
                # Create a separate floor_plate_element for each insertion in the layout
                for x, y, xr, yr, rotation_angle, reflection_angle, \
                    r_mx, r_my, r_mz, r_xp, r_xn, r_yp, r_yn, r_zp, r_zn in zip(
                    layout["X"], layout["Y"], layout["Xr"], layout["Yr"],
                    layout["Rotation"], layout["Reflection"],
                    layout["release_mx"], layout["release_my"], layout["release_mz"],
                    layout["release_xp"], layout["release_xn"],
                    layout["release_yp"], layout["release_yn"],
                    layout["release_zp"], layout["release_zn"]
                ):

                    translation = np.array([x + xr * self.Bx,
                                            y + yr * self.By])

                    boundary = Utilities.transform_points(shape_points, translation, rotation_angle, reflection_angle)

                    anchors = Utilities.transform_points(anchor_points, translation, rotation_angle, reflection_angle)

                    (x0, y0) = Utilities.transform_points([[element['x0'], element['y0']]],
                                                          translation, rotation_angle, reflection_angle)[0]
                    z0 = element['z0']

                    (xc, yc) = Utilities.transform_points([[element['xc'], element['yc']]],
                                                          translation, rotation_angle, reflection_angle)[0]
                    zc = element['zc']

                    self.floor_plates.append(FloorPlateElement([boundary],
                                                               xy_anchors=None if anchors.size == 0 else anchors,
                                                               x0=x0,
                                                               y0=y0,
                                                               z0=z0,
                                                               xc=xc,
                                                               yc=yc,
                                                               zc=zc,
                                                               release_mx=r_mx,
                                                               release_my=r_my,
                                                               release_mz=r_mz,
                                                               release_xp=r_xp,
                                                               release_xn=r_xn,
                                                               release_yp=r_yp,
                                                               release_yn=r_yn,
                                                               release_zp=r_zp,
                                                               release_zn=r_zn,
                                                               E_base=E_base,
                                                               poisson=poisson))

                    plate = self.floor_plates[-1]
                    if element['check_fasteners']:
                        # get all the inputs sorted out
                        fastener_pattern = element['fastener_geometry_name']
                        data = df_fasteners.loc[fastener_pattern]
                        faying_local_angle = orientations[element['fastener_orientation']]
                        faying_local_vector = (np.cos(faying_local_angle), np.sin(faying_local_angle))
                        faying_global_vector = Utilities.transform_vectors([faying_local_vector],
                                                                           rotation_angle, reflection_angle)[0]
                        local_z = (*faying_global_vector, 0)
                        local_y = (0, 0, 1)  # Assumes connection element if vertical
                        local_x = np.cross(local_y, local_z)

                        if np.isclose(faying_global_vector[0], 0):
                            B = self.By
                        elif np.isclose(faying_global_vector[1], 0):
                            B = self.Bx
                        else:
                            B = 0
                        w = data['W'] + data['Wr'] * B
                        h = data['H'] + data['Hr'] * self.H
                        if w == 0:
                            raise Exception(
                                "For base plate attachments with fasteners, you must specify absolute dimension for fastener pattern or place floor plates orthogonal to reference box")

                        L_horiz = w - 2 * data['X Edge']
                        L_vert = h - 2 * data['Y Edge']
                        y_offset = data['Y Offset']
                        x_offset = data['X Offset']
                        place_by_horiz = data['X Placement']
                        place_by_vert = data['Y Placement']

                        xy_points = Utilities.compute_backing_xy_points(data['X Number'],
                                                                        data['Y Number'],
                                                                        L_horiz, L_vert, x_offset, y_offset,
                                                                        place_by_horiz=place_by_horiz,
                                                                        place_by_vert=place_by_vert)

                        plate.connection = SMSHardwareAttachment(w, h, xy_points, df_sms,
                                                                 centroid=(xc, yc, zc),
                                                                 local_x=local_x,
                                                                 local_y=local_y,
                                                                 local_z=local_z)

                        gauge = self.gauge_equip if self.gauge_equip is not None else 18
                        fy = self.fy_equip if self.fy_equip is not None else 33
                        plate.connection.anchors_obj.set_sms_properties(gauge, fy)

                    for strap_pts in strap_geometry:
                        (x_eq, y_eq,) = Utilities.transform_points([[strap_pts[0] + strap_pts[3],
                                                                     strap_pts[1] + strap_pts[4]]],
                                                                   translation, rotation_angle, reflection_angle)[0]
                        z_eq = strap_pts[2] + strap_pts[5]

                        (x_pl, y_pl,) = Utilities.transform_points([[strap_pts[0],
                                                                     strap_pts[1]]],
                                                                   translation, rotation_angle, reflection_angle)[0]
                        z_pl = strap_pts[2]

                        self.base_straps.append(BaseStrap((x_eq, y_eq, z_eq), (x_pl, y_pl, z_pl), plate))


            else:
                bearing_boundaries = []
                xy_anchors = []
                for x, y, xr, yr, rotation_angle, reflection_angle in zip(layout['X'],
                                                                          layout['Y'],
                                                                          layout['Xr'],
                                                                          layout['Yr'],
                                                                          layout['Rotation'],
                                                                          layout['Reflection']):
                    for par in [x, y, xr, yr]:
                        if isinstance(par, str):
                            par = float(par)

                    translation = np.array([x + xr * self.Bx,
                                            y + yr * self.By])

                    boundary = Utilities.transform_points(shape_points, translation, rotation_angle, reflection_angle)
                    anchors = Utilities.transform_points(anchor_points, translation, rotation_angle, reflection_angle)

                    bearing_boundaries.append(boundary)
                    xy_anchors += anchors.tolist()

                xy_anchors = None if xy_anchors == [] else np.array(xy_anchors)
                self.floor_plates.append(FloorPlateElement(bearing_boundaries,
                                                           xy_anchors=xy_anchors,
                                                           x0=0.0,
                                                           y0=0.0,
                                                           z0=element['z0'],
                                                           xc=0.0,
                                                           yc=0.0,
                                                           zc=element['z0'],
                                                           E_base=E_base,
                                                           poisson=poisson))

    def wall_bracket_backing_anchors_from_json(self, equipment_data, df_fasteners, df_sms):

        b_dimension = {'X+': self.By,
                       'X-': self.By,
                       'Y+': self.Bx,
                       'Y-': self.Bx}

        wall_geometry_json = equipment_data['Pattern Definition_wall']
        if not isinstance(wall_geometry_json,str):
            print(f"WARNING: {self.equipment_id} wall geometry {equipment_data['wall_geometry']} not found.")
            return

        json_data = json.loads(wall_geometry_json)

        self.omit_bracket_output = json_data['omit_bracket_output']

        # Extract Wall Offsets
        for wall_key, offset_val in json_data['wall_offsets'].items():
            self.wall_offsets[wall_key] = offset_val

        # Extract bracket_locations to DataFrame
        bracket_locations = json_data['bracket_locations']
        df_bracket_locations = pd.DataFrame(bracket_locations)
        df_backing_groups = pd.DataFrame.from_dict(json_data['backing_groups'], orient='columns')
        df_brackets = df_bracket_locations.merge(df_backing_groups, left_on='Backing Group', right_on='group_number',
                                                 how='left')
        df_brackets = df_brackets.merge(df_fasteners, left_on='backing_pattern', right_on='Pattern Name', how='left')

        # Wall Properties
        L = equipment_data['wall_height']
        E = equipment_data['E_wall']
        I = equipment_data['I_wall']

        bracket_backing_map = {}
        for index, bracket in df_brackets.iterrows():
            # Attachment Point
            x0 = bracket['X'] + bracket['Xr'] * self.Bx
            y0 = bracket['Y'] + bracket['Yr'] * self.By
            z0 = bracket['Z'] + bracket['Zr'] * self.H
            xyz_equipment = np.array((x0, y0, z0))

            # Wall Normal Vector
            supporting_wall = bracket['Supporting Wall']
            normal_vec = EquipmentModel.WALL_NORMAL_VECTORS[supporting_wall]

            # Calculate Bracket Centerline Point with Connection Offset
            # attachment_normal_vec = np.array([EquipmentModel.ATTACHMENT_NORMAL_VECTORS[supporting_wall]])
            # cxn_offset = json_data['attachment_offset']  #todo: remove
            # x_offset, y_offset, z_offset = (attachment_normal_vec * cxn_offset + np.array(xyz_0))[0]
            # xyz_offset = (x_offset, y_offset, z_offset)

            z_offset = bracket['Plate Y Offset']

            wall_gap = 0 if not self.wall_offsets[supporting_wall] else self.wall_offsets[supporting_wall]
            backing_depth = bracket['D']
            if supporting_wall == 'X+':
                xyz_wall = np.array((0.5 * self.Bx + (wall_gap - backing_depth), y0, z0))
                x_offset = 0
                y_offset = -bracket['Plate Y Offset']
            elif supporting_wall == 'X-':
                xyz_wall = np.array((-0.5 * self.Bx - (wall_gap - backing_depth), y0, z0))
                x_offset = 0
                y_offset = bracket['Plate Y Offset']
            elif supporting_wall == 'Y+':
                xyz_wall = np.array((x0, 0.5 * self.By + (wall_gap - backing_depth), z0))
                x_offset = bracket['Plate X Offset']
                y_offset = 0
            elif supporting_wall == 'Y-':
                xyz_wall = np.array((x0, -0.5 * self.By - (wall_gap - backing_depth), z0))
                x_offset = -bracket['Plate X Offset']
                y_offset = 0
            else:
                raise Exception('Supporting Wall Incorrectly Defined')

            # Stiffness Releases
            releases = (bracket['N+'], bracket['N-'], bracket['P+'], bracket['P-'], bracket['Z+'], bracket['Z-'])

            # Wall Flexibility
            a = z0
            b = L - a
            wall_flexibility = (a ** 2 + b ** 2) / (3 * E * I * L)  # Wall idealized as simple span

            self.wall_brackets.append(
                WallBracketElement(supporting_wall, xyz_equipment, xyz_wall, normal_vec, wall_flexibility,
                                   releases=releases,
                                   backing_offset_x=x_offset, backing_offset_y=y_offset, backing_offset_z=z_offset))

            # Create Bracket Connection Elements
            if json_data['check_fasteners']:
                bracket_obj = self.wall_brackets[-1]
                fastener_pattern = json_data["fastener_geometry"]
                data = df_fasteners.loc[fastener_pattern]
                try:
                    local_z = EquipmentModel.ATTACHMENT_NORMAL_VECTORS[bracket['Attachment Normal']]
                    if bracket['Attachment Normal'] in ['X+', 'Y+', 'X-', 'Y-']:
                        local_y = (0,0,1)
                        local_x = np.cross(local_y,local_z)
                    else:
                        local_x = -1*np.array(EquipmentModel.WALL_NORMAL_VECTORS[bracket_obj.supporting_wall])
                        local_y = np.cross(local_z,local_x)
                except:
                    raise Exception("Must specify wall bracket attachment normal direction.")



                B = 0  # Note, wall brackets are set to ignore any relative dimension
                w = data['W'] + data['Wr'] * B
                h = data['H'] + data['Hr'] * self.H
                if np.isclose(w, 0.0) or np.isclose(h, 0.0):
                    raise Exception(
                        "For wall bracket attachments, you must specify absolute dimensions")

                L_horiz = w - 2 * data['X Edge']
                L_vert = h - 2 * data['Y Edge']
                y_offset = data['Y Offset']
                x_offset = data['X Offset']
                place_by_horiz = data['X Placement']
                place_by_vert = data['Y Placement']

                xy_points = Utilities.compute_backing_xy_points(data['X Number'],
                                                                data['Y Number'],
                                                                L_horiz, L_vert, x_offset, y_offset,
                                                                place_by_horiz=place_by_horiz,
                                                                place_by_vert=place_by_vert)

                bracket_obj.connection = SMSHardwareAttachment(w, h, xy_points, df_sms,
                                                               centroid=(x0, y0, z0),
                                                               local_x=local_x,
                                                               local_y=local_y,
                                                               local_z=local_z)

                gauge = self.gauge_equip if self.gauge_equip is not None else 18
                fy = self.fy_equip if self.fy_equip is not None else 33
                bracket_obj.connection.anchors_obj.set_sms_properties(gauge, fy)

            # Map bracket index to backing group
            bracket_backing_map[index] = bracket['Backing Group']

        # Create Bracket-Backing mapping array
        backing_bracket_map = {}
        for i, bracket in enumerate(self.wall_brackets):
            backing_group = bracket_backing_map[i]
            if backing_group not in backing_bracket_map:
                backing_bracket_map[backing_group] = []
            backing_bracket_map[backing_group].append(i)

        # Create Backing Elements
        for backing_group, bracket_indices in backing_bracket_map.items():
            if backing_group == 0:
                for i in bracket_indices:
                    backing_data = df_brackets.iloc[i]
                    self.create_backing_element(backing_data, [i], b_dimension)
            else:
                backing_data = df_brackets[df_brackets['Backing Group'] == backing_group].iloc[0]
                self.create_backing_element(backing_data, bracket_indices, b_dimension)

        # Create Wall-Backing Mapping Array
        for index, backing in enumerate(self.wall_backing):
            self.backing_indices[backing.supporting_wall].append(index)

        # Create Wall Anchor Elements
        supporting_walls = set([el.supporting_wall for el in self.wall_backing])
        wall_type = equipment_data['wall_type']
        if wall_type in ['Concrete', 'CMU']:
            for wall_position in supporting_walls:
                # Collect anchor points from all wall anchors

                xy_anchors = np.concatenate(
                    [plate.pz_anchors + (plate.global_to_local_transformation@plate.centroid)[0:2] for plate in
                     [self.wall_backing[i] for i in self.backing_indices[wall_position]]], axis=0)

                self.include_overstrength = True  # todo: [Calc Refinement] Define when overstrength can be omitted
                self.wall_anchors[wall_position] = ConcreteAnchors(
                    xy_anchors=xy_anchors) if wall_type == 'Concrete' else CMUAnchors(xy_anchors=xy_anchors)

        elif wall_type == 'Metal Stud':
            wall_data = equipment_data.loc[['stud_gauge', 'stud_fy', 'num_gyp']]
            num_gyp = wall_data['num_gyp']
            for backing in self.wall_backing:
                if num_gyp == 2:
                    condition_x = 'Condition 3'
                    condition_y = 'Condition 3'
                elif num_gyp == 1:
                    condition_x = 'Condition 2'
                    condition_y = 'Condition 2'
                else:
                    condition_x = 'Condition 1'
                    condition_y = 'Condition 1'

                if backing.backing_type == 'Wall Backing (Strut)' and backing.w > backing.h:
                    condition_y = 'Condition 4'
                elif backing.backing_type == 'Wall Backing (Strut)' and backing.h > backing.w:
                    condition_x = 'Condition 4'

                backing.anchors_obj = SMSAnchors(wall_data=wall_data,
                                                              backing_type=backing.backing_type, df_sms=df_sms,
                                                              condition_x=condition_x, condition_y=condition_y)

        elif wall_type == 'Wood Stud':
            g = 0.5 * equipment_data['num_gyp']
            for backing in self.wall_backing:
                backing.anchors_obj = WoodFastener(backing.pz_anchors)
                backing.anchors_obj.set_member_properties_from_data_table(equipment_data, base_or_wall='wall')
                t_steel = backing.t_steel  #todo: pick up here
                Fes = backing.fy*1000
                backing.anchors_obj.set_steel_props(t_steel,Fes,g=g)
        else:
            raise Exception(f'Wall type {wall_type} for {self.equipment_id} not supported')

    def create_backing_element(self, backing_data: dict | pd.Series, bracket_indices, b_dimension_dict):
        supporting_wall = backing_data['Supporting Wall']
        backing_type = backing_data['Connection Type']
        b = b_dimension_dict[supporting_wall]  # Width of unit parallel to supporting wall
        wb = backing_data['W'] + backing_data['Wr'] * b  # Width of bracket
        hb = backing_data['H'] + backing_data['Hr'] * self.H  # Height of bracket
        db = backing_data['D']
        L_horizontal = wb - 2 * backing_data['X Edge']
        L_vert = hb - 2 * backing_data['Y Edge']
        y_offset = backing_data['Y Offset']
        x_offset = backing_data['X Offset']
        place_by_horiz = backing_data['X Placement']
        place_by_vert = backing_data['Y Placement']

        t_steel = backing_data['Steel Thickness']
        fy = backing_data['Steel Fy']

        manual_points_json = backing_data['Manual Points']
        if manual_points_json is np.nan:
            raise Exception(f'The specified wall backing for {self.equipment_id} is not found the fastener patterns list.')
        manual_points = json.loads(manual_points_json)
        manual_x = manual_points['x']
        manual_y = manual_points['y']

        xy_anchor_points = Utilities.compute_backing_xy_points(backing_data['X Number'], backing_data['Y Number'],
                                                               L_horizontal, L_vert, x_offset, y_offset,
                                                               place_by_horiz=place_by_horiz,
                                                               place_by_vert=place_by_vert,
                                                               manual_x=manual_x,
                                                               manual_y=manual_y)



        # Convert Bracket locations from global XYZ coordinates to local NPZ Coordinates
        npz_brackets = np.array(
            [bracket.G @ bracket.xyz_backing for bracket in [self.wall_brackets[i] for i in bracket_indices]])
        pz_brackets = npz_brackets[:, [1, 2]]
        pz_cent = np.mean(pz_brackets, axis=0)
        xy_brackets_local = pz_brackets - pz_cent
        centroid_in_global_coordinates = np.mean(np.array([bracket.xyz_backing
                                                           for bracket in [self.wall_brackets[i] for i in bracket_indices]]),axis=0)
        local_z = EquipmentModel.WALL_NORMAL_VECTORS[supporting_wall]
        local_y = (0,0,1)
        local_x = np.cross(local_y,local_z)

        self.wall_backing.append(
            WallBackingElement(wb, hb, db, xy_anchor_points, xy_brackets_local, bracket_indices, supporting_wall,
                               backing_type=backing_type, centroid=centroid_in_global_coordinates,
                               local_x=local_x,
                               local_y=local_y,
                               local_z=local_z,
                               fy=fy, t_steel=t_steel))

    def number_degrees_of_freedom(self):
        """Numbers the degrees of freedom, including additional DOFs for floor plates with moment releases."""
        self.floor_plate_dofs = np.full((len(self.floor_plates), 6), [0, 1, 2, 3, 4, 5], dtype=int)
        dof_count = 6
        self.disp_dofs = [1, 1, 1, 0, 0, 0]
        for i, element in enumerate(self.floor_plates):
            if element.release_zp and element.release_zn:
                self.floor_plate_dofs[i, 2] = dof_count
                self.disp_dofs.append(1)
                dof_count += 1
            if element.release_mx: # or element.prying_mx:
                self.floor_plate_dofs[i, 3] = dof_count
                self.disp_dofs.append(0)
                dof_count += 1
            if element.release_my: # or element.prying_my:
                self.floor_plate_dofs[i, 4] = dof_count
                self.disp_dofs.append(0)
                dof_count += 1
            if element.release_mz:
                self.floor_plate_dofs[i, 5] = dof_count
                self.disp_dofs.append(0)
                dof_count += 1
        self.n_dof = dof_count

    def set_element_dofs(self):
        """Updates element constraint matrices and dof mapping based on global dofs"""
        for i, element in enumerate(self.floor_plates):
            element.set_dof_constraints(self.n_dof, self.floor_plate_dofs[i, :])
        for element in self.wall_brackets:
            element.set_dof_constraints(self.n_dof)
        for strap in self.base_straps:
            strap.pre_compute_matrices()

    # def check_base_plate_prying(self):
        # for each plate, identify maximum anchor force
        # update element resultants
        # check prying
        # store prying result for report (largest force, if no prying, any if yes prying)
        # if any have prying,

            # refactor model to add dofs at plates
            # reanalyze model.

    def set_base_anchor_data(self, anchor_data):
        """Sets anchor properties for anchor object and updates stiffness properties for floor plate elements"""
        self.base_anchors.set_mechanical_anchor_properties(anchor_data)
        for element in self.floor_plates:
            element.set_anchor_properties(self.base_anchors)

    def set_wall_bracket_data(self, bracket_data):
        for element in self.wall_brackets:
            element.set_bracket_properties(bracket_data, self.asd_lrfd_ratio)

    def set_base_strap_data(self, strap_data):
        for element in self.base_straps:
            element.set_brace_properties(strap_data)

    def calculate_fp(self, use_dynamic=False):
        if self.code_edition == 'ASCE 7-16':
            sds = self.code_pars['sds']
            Ip = self.code_pars['Ip']
            ap = self.code_pars['ap']
            Rp = self.code_pars['Rp']
            z = self.code_pars['z']
            h = self.code_pars['h']
            omega = self.code_pars['omega']
            Wp = self.Wp

            self.Fp_min = 0.3 * sds * Ip * Wp  # ASCE7-16 13.3-2
            self.Fp_max = 1.6 * sds * Ip * Wp  # ASCE7-16 13.3-3
            self.Fp_code = (0.4 * ap * sds * Wp * Ip / Rp) * (1 + 2 * z / h)  # ASCE7-16 13.3-1

            if use_dynamic:
                ai = self.code_pars['ai']
                Ax = self.code_pars['Ax']
                self.Fp_dynamic = (ap * Wp * Ip / Rp) * ai * Ax  # ASCE7-16 13.3-4
                self.Fp = max(min(self.Fp_max, self.Fp_dynamic), self.Fp_min)  # ASCE7-16 13.3.1.1
            else:
                self.Fp = max(min(self.Fp_max, self.Fp_code), self.Fp_min)  # ASCE7-16 13.3.1.1

            self.Eh = self.Fp  # ASCE7-16 13.3.1.1
            self.Emh = omega * self.Eh
            self.Ev = 0.2 * sds * Wp  # ASCE7-16 13.3.1.2

        elif self.code_edition == 'ASCE 7-22':
            raise Exception('Definition of fp for ASCE7-22 not yet defined')

        elif self.code_edition == 'ASCE 7-22 OPM':
            self.Fp = self.code_pars['Cpm'] * self.Wp
            self.Eh = self.Fp
            omega = self.code_pars['omega']
            self.Emh = omega * self.Eh
            self.Eh = self.Fp
            self.Ev = self.code_pars['Cv'] * self.Wp

        elif self.code_edition == 'CBC 1998, 16B':
            Z = self.code_pars['Z']
            Ip = self.code_pars['Ip']
            Cp = self.code_pars['Cp']
            max_amplification = {1: 999,
                                 2: 2,
                                 4: 3}
            Cp_eff = min([Cp * self.code_pars['cp_amplification'],
                          max_amplification[self.code_pars['cp_amplification']]]) * self.code_pars['grade_factor']
            self.code_pars['Cp_eff'] = Cp_eff
            Wp = self.Wp

            self.Fp = Z * Ip * Cp_eff * Wp
            self.Eh = self.Fp
            self.Ev = self.Eh / 3
            self.Emh = self.Fp

        else:
            raise Exception('Specified code year not supported.')

    def calculate_factored_loads(self):
        if self.code_edition in ['ASCE 7-16', 'ASCE 7-22 OPM']:
            # LRFD
            self.Fuv_min = -0.9 * self.Wp + 1.0 * self.Ev  # Minimum downward force
            self.Fuv_max = -1.2 * self.Wp - 1.0 * self.Ev  # Maximum downward force
            if self.include_overstrength:
                self.Fuh = 1.0 * self.Emh
            else:
                self.Fuh = 1.0 * self.Eh

            # ASD
            self.Fav_min = -0.6 * self.Wp + 0.7 * self.Ev
            self.Fav_max = -1.0 * self.Wp - 0.7 * self.Ev
            if self.include_overstrength:
                self.Fah = 0.7 * self.Emh
            else:
                self.Fah = 0.7 * self.Eh




        elif self.code_edition == 'ASCE 7-22':
            raise Exception('Definition of fp for ASCE7-22 not yet defined')

        elif self.code_edition == 'CBC 1998, 16B':
            # LRFD
            self.Fuv_min = -0.9*self.Wp + 1.3 * 1.1*self.Ev # Minimum downward force
            self.Fuv_max = -0.75*(1.4*self.Wp + 1.7*1.1 * self.Ev)  # 98CBC 9B-2  # Maximum downward force
            self.Fuh = 0.75*1.7*1.1 * self.Eh
            # ASD
            # self.Fav_min = self.Fuv_min
            # self.Fav_max = self.Fuv_max
            self.Fah = self.Eh
        else:
            raise Exception('Specified code year not supported.')

        self.asd_lrfd_ratio = self.Fah / self.Fuh
        if self.factor_type == "LRFD":
            self.Fv_min = self.Fuv_min
            self.Fv_max = self.Fuv_max
            self.Fv = self.Fuv_min if self.base_anchors is not None else self.Fuv_max
            self.Fh = self.Fuh
        elif self.factor_type == 'ASD':
            self.Fv_min = self.Fav_min
            self.Fv_max = self.Fav_max
            self.Fv = self.Fav_min if self.base_anchors is not None else self.Fav_max
            self.Fh = self.Fah

    def set_product_data_and_analyze(self, base_anchor_data=None, base_strap_data=None,
                                     bracket_data=None, wall_anchor_data=None,
                                     hardware_screw_size=None,
                                     initial_solution_cache=None):

        self.omit_analysis = False

        # Set Base Anchor Data
        if self.base_anchors is not None:
            self.base_anchors.reset_results()
            if base_anchor_data is not None:
                self.set_base_anchor_data(base_anchor_data)
                self.base_anchors.check_anchor_spacing()
                if not all(self.base_anchors.spacing_requirements.values()):
                    self.omit_analysis = True

        # Set Bast Strap Data
        if base_strap_data is not None:
            self.set_base_strap_data(base_strap_data)

        # Set Wall Brackets and Wall Anchors Data
        if self.wall_brackets != [] and bracket_data is not None:
            self.set_wall_bracket_data(bracket_data)

            for position, wall_anchors in self.wall_anchors.items():
                if wall_anchors is not None and wall_anchor_data is not None:
                    wall_anchors.reset_results()
                    if isinstance(wall_anchors, ConcreteCMU):
                        wall_anchors.set_mechanical_anchor_properties(wall_anchor_data)
                        wall_anchors.check_anchor_spacing()
                        if not all(wall_anchors.spacing_requirements.values()):
                            self.omit_analysis = True

            for wall_anchors in [b.anchors_obj for b in self.wall_backing]:
                if isinstance(wall_anchors, SMSAnchors) and wall_anchor_data is not None:
                    wall_anchors.reset_results()
                    wall_anchors.set_screw_size(wall_anchor_data)
                if isinstance(wall_anchors, WoodFastener) and wall_anchor_data is not None:
                    wall_anchors.set_fastener_properties(wall_anchor_data)

        # Set Hardware Attachment Screw Size
        for plate in self.floor_plates:
            if isinstance(plate.connection, SMSHardwareAttachment) and hardware_screw_size is not None:
                plate.connection.anchors_obj.reset_results()
                plate.connection.anchors_obj.set_screw_size(hardware_screw_size)
        for bracket in self.wall_brackets:
            if isinstance(bracket.connection, SMSHardwareAttachment) and hardware_screw_size is not None:
                bracket.connection.anchors_obj.reset_results()
                bracket.connection.anchors_obj.set_screw_size(hardware_screw_size)

        # Number DOFs and Set-up Element Constraints
        self.number_degrees_of_freedom()
        self.set_element_dofs()

        # Check Model Stability
        self.check_model_stability()

        if not self.omit_analysis:
            self.analyze_model(initial_solution_cache=initial_solution_cache)

            # Base Anchor Checks
            if self.base_anchors is not None:
                if isinstance(self.base_anchors,ConcreteAnchors):
                    self.base_anchors.get_governing_anchor_group()
                    self.base_anchors.check_anchor_capacities()
                elif isinstance(self.base_anchors,WoodFastener):
                    self.base_anchors.check_fasteners()

            # Base Plate Connection Checks
            # for plate in self.floor_plates:
            #     if isinstance(plate.connection, SMSHardwareAttachment) and hardware_screw_size is not None:
            #         plate.connection.anchors_obj.check_anchors(self.asd_lrfd_ratio)

            # Base Strap Checks
            for strap in self.base_straps:
                asd_lrfd_ratio = None if strap.capacity_method == 'LRFD' else self.asd_lrfd_ratio
                strap.check_brace(self.governing_solutions['base_anchor_tension']['sol'], asd_lrfd_ratio=asd_lrfd_ratio)

            # Wall Bracket Checks
            if self.wall_brackets:
                bracket_max_tension = self.wall_bracket_forces[:, :, 0].max(axis=1)
                bracket = max([(b, t) for b, t in zip(self.wall_brackets, bracket_max_tension)], key=lambda x: x[1])[0]
                bracket.check_brackets()

            # Bracket Connection Checks
            # for bracket in self.wall_brackets:
            #     if isinstance(bracket.connection, SMSHardwareAttachment) and hardware_screw_size is not None:
            #         bracket.connection.anchors_obj.check_anchors(self.asd_lrfd_ratio)

            # Wall Anchor Checks
            all_wall_anchors = [anchors_obj for wall_position, anchors_obj in self.wall_anchors.items()
                                if anchors_obj is not None] + [b.anchors_obj for b in self.wall_backing if
                                                               b.anchors_obj is not None]
            for wall_anchors in all_wall_anchors:
                if isinstance(wall_anchors, SMSAnchors):
                    wall_anchors.check_anchors(self.asd_lrfd_ratio)
                elif isinstance(wall_anchors, CMUAnchors):
                    pass
                elif isinstance(wall_anchors, ConcreteAnchors):
                    wall_anchors.get_governing_anchor_group()
                    wall_anchors.check_anchor_capacities()
                elif isinstance(wall_anchors, WoodFastener):
                    wall_anchors.check_fasteners()
                elif wall_anchors is None:
                    pass
                else:
                    raise Exception("Anchor type not supported")

    def get_load_vector(self, theta_z):
        """ Return a load vector in the form: [Vx, Vy, N, Mx, My, T].
        Load vector corresponds only to first six primary dofs. It is assumed all other dofs are zero.
        Assumes self.calculate_factored_loads has been previously called."""
        #  Loads at basic DOFs
        vx = self.Fh * math.cos(theta_z)
        vy = self.Fh * math.sin(theta_z)
        fz = self.Fv
        mx = -vy * self.zCG + fz * self.ey
        my = vx * self.zCG - fz * self.ex
        t = -vx * self.ey + vy * self.ex

        p = np.zeros(self.n_dof)

        # Load Vector
        p[0:6] = [vx, vy, fz, mx, my, t]

        return p

    def check_model_stability(self, tol=1e-12):
        """ Can be run after setting all anchor and hardware data.
        Will impose small dof displacements in principal directions
        and verify model stabilit by checking for non-zero eigenvalues."""

        dof_labels = ['Dx', 'Dy', 'Dz', 'Rx', 'Ry', 'Rz']
        dir_labels = ['+', '-']

        u0 = 1 * np.eye(self.n_dof)
        for dof, u_dof in enumerate(u0[0:5, :]):
            for direction, u in enumerate([u_dof, -1*u_dof]):
                k = self.update_stiffness_matrix(u)
                # eigenvalues = np.linalg.eigenvalues(k)
                # zero_modes = np.sum(np.abs(eigenvalues) < tol)
                # unstable = zero_modes > 1
                p = k @ u
                unstable = np.abs(p[dof]) < tol

                if unstable:
                    if self.installation_type == 'Wall Brackets' and dof == 2 and direction == 0:
                        '''Ignore Dz+ instability for Wall Brackets units.
                        It is assumed that resultant vertical forces will always be downward.'''
                        continue
                    else:
                        self.model_unstable = self.omit_analysis = True
                        print(f'WARNING: Model instability detected at DOF {dof_labels[dof] + dir_labels[direction]}. '
                              f'Check model geometry definitions, material properties, and releases.')
                        return

        self.cor_coordinates = self.get_center_of_rigidity()


    def get_center_of_rigidity(self):



        cor_coordinates = {'x_displacement':{1:None,
                                             -1:None},
                           'y_displacement': {1:None,
                                             -1:None},
                           'z_displacement': {1:None,
                                             -1:None}}
        axes = ['x_displacement', 'y_displacement', 'z_displacement']
        u_vecs = np.eye(self.n_dof,3)

        ''' Center of Rigidity coordinates are computed generally as m/p,
            where m is a relevant component of moment (mx, my, or mz).
            for a force in x-direction: cor = (0, -mz/px, my/px),
            for a force in y-direction: cor = (mz/py, 0, -mx / py),
            for a force in z-direction: cor = (-my/pz, mx / pz, 0).
                        
            For conciseness in the code, this is computed by extracting the signs of the various terms,
            and the index of the relevant moment component into lists which can be iterated over.'''

        moment_term_signs = [np.array([0, -1, 1]),
                             np.array([1, 0, -1]),
                             np.array([-1, 1, 0])]
        moment_component_idx = [(0, 2, 1),
                                (2, 1, 0),
                                (1, 0, 2)]

        for i, axis in enumerate(axes):
            for dir_sign in [-1,1]:
                u = dir_sign*u_vecs[:,i]
                k = self.update_stiffness_matrix(u)
                p = k @ u
                force = p[i]
                moments = p[3:]
                cor_coordinates[axis][dir_sign] = [sgn*moments[idx]/force for sgn, idx in
                                                   zip (moment_term_signs[i],moment_component_idx[i])]

        return cor_coordinates

        # # Find Center of Rigidity for x-direction
        # u = np.eye(1, self.n_dof, 0).T
        # k = self.update_stiffness_matrix(u)
        # p = k@u
        # mx, my, mz = p[3:,0]
        # px = p[0,0]
        # self.cor_x = (0, -mz/px, my/px)
        #
        # # Find Center of Rigidity for y-direction
        # u = np.eye(1, self.n_dof, 1).T
        # k = self.update_stiffness_matrix(u)
        # p = k @ u
        # mx, my, mz = p[3:, 0]
        # py = p[1, 0]
        # self.cor_y = (mz/py, 0, -mx / py)
        #
        # # Find center of Rigidity for z-direction
        # u = np.eye(1, self.n_dof, 2).T
        # k = self.update_stiffness_matrix(u)
        # p = k @ u
        # mx, my, mz = p[3:, 0]
        # pz = p[2, 0]
        # self.cor_z = (-my/pz, mx / pz, 0)


    def get_initial_dof_guess(self, theta_z):
        """ Returns an initial guess for the DOF displacements U0 = [dx, dy, dz, rx, ry, rz, ...]"""

        u_init_list = [self._center_of_rigidity_based_dof_guess(theta_z, rot_magnitude=0.01/min(self.Bx,self.By)),# Initial guess at given angle
                       self._center_of_rigidity_based_dof_guess(theta_z, rot_magnitude=1e-7),
                       self._center_of_rigidity_based_dof_guess(theta_z + np.pi / 16),  # At perturbed angle
                       self._center_of_rigidity_based_dof_guess(theta_z - np.pi / 16)]

        # u_init_list = [self._p_proportional_dof_guess(theta_z),  # Proportional to p at given angle
        #                self._p_proportional_dof_guess(theta_z + np.pi / 16),  # Proportional to p at perturbed angle
        #                self._p_proportional_dof_guess(theta_z - np.pi / 16)]  # Proportional to p at perturbed angle
        # # np.array([0, 0, 1e-6, 0, 0, 0] + [0]*(self.n_dof-6)),  # Uplift Translation Only
        # # np.array([0, 0, -1e-6, 0, 0, 0] + [0]*(self.n_dof-6)),  # Gravity Translation Only
        # # np.zeros(self.n_dof)]  # Zero

        return u_init_list

    def get_interpolated_dof_guess(self, idx):
        """ Returns trial solutions for an unconverged point
        by taking weighted averages of bounding converged solutions points"""

        n = len(self.converged)
        before = None
        after = None

        # Wrapped search backward
        for offset in range(1, n):
            i = (idx - offset) % n
            if self.converged[i]:
                sol_before = self.equilibrium_solutions[:, i]
                break

        # Wrapped search forward
        for offset in range(1, n):
            i = (idx + offset) % n
            if self.converged[i]:
                sol_after = self.equilibrium_solutions[:, i]
                break

        # sol_before = self.equilibrium_solutions[:, before]
        # sol_after = self.equilibrium_solutions[:, after]

        u_tries = [0.5 * sol_before + 0.5 * sol_after,
                   0.25 * sol_before + 0.75 * sol_after,
                   0.75 * sol_before + 0.25 * sol_after,
                   sol_after]

        return u_tries

    def _initial_stiffness_dof_guess_UNUSED(self, theta):
        p = self.get_load_vector(theta)
        u = np.linalg.solve(self.initial_stiffness_matrix_UNUSED(), p)
        return u

    def _center_of_rigidity_based_dof_guess(self,theta, disp_magnitude=1e-6, rot_magnitude=1e-6):
        p = self.get_load_vector(theta)

        u = np.zeros(self.n_dof)

        # Calculate Moments about COR
        fx, fy, fz = p[0:3]

        sign_fx = 1 if fx >= 0 else -1
        sign_fy = 1 if fy >= 0 else -1
        sign_fz = 1 if fz >= 0 else -1

        cor_x = self.cor_coordinates['x_displacement'][sign_fx]  # COR for displacement in x-direction
        cor_y = self.cor_coordinates['y_displacement'][sign_fy]  # COR for displacement in y-direction
        cor_z = self.cor_coordinates['z_displacement'][sign_fz]  # COR for displacement in z-direction

        mx = (cor_y[2] - self.zCG) * fy - (cor_z[1] - self.ey) * fz
        my = -(cor_x[2] - self.zCG) * fx + (cor_z[0] - self.ex) * fz
        mz = (cor_x[1] - self.ey) * fx - (cor_y[0] - self.ex) * fy
        m_array = np.array((mx, my, mz))

        # Apply DOF Rotations proportional to moments about COR
        unit_rot = m_array / np.linalg.norm(m_array)
        rx, ry, rz = rot_magnitude * unit_rot

        # Apply DOF translations are proportional to applied forces such that rotation occurs about COR
        unit_disp = p[0:3] if p[0:3].sum() == 0 else p[0:3] / np.linalg.norm(p[0:3])
        cor_disp = disp_magnitude * unit_disp

        y, z = cor_x[1:]
        dx = cor_disp[0] -z * ry + y * rz

        x = cor_y[0]
        z = cor_y[2]
        dy =  cor_disp[1] + z * rx - x * rz

        x = cor_z[0]
        y = cor_z[1]
        dz = cor_disp[2] + x * ry - y * rx

        u[0:6] = [dx, dy, dz, rx, ry, rz]

        # Compute Floor Plate DOF initial guesses
        for dof_map, plate in zip(self.floor_plate_dofs, self.floor_plates):

            # # NEW METHOD
            # if len(plate.free_dofs)>0:
            #     u_free, success = plate.solve_released_dof_equilibrium(u, dof_map)
            #     if success:
            #         u[dof_map[plate.free_dofs]] = u_free


            ## OLD METHOD
            dzp = dz + plate.yc * rx - plate.xc * ry  # z-displacement at (xc,yc)

            # # Handle Vertical Releases
            # if (plate.release_zp and not plate.release_zn and dzp >=0) or (plate.release_zn and not plate.release_zp and dzp <0):
            #     if plate.release_mx:
            #         u[dof_map[3]] = 0
            #     if plate.release_my:
            #         u[dof_map[4]] = 0
            #     return u

            # Handle Independent Vertical DOF (Triggered when both zn and zp releases are present)
            if plate.release_zp and plate.release_zn:
                # dzp = max(0, 0.5*dzp)
                dzp = 0.5 * abs(dzp)
                # dzp = 1e-7
                u[dof_map[2]] = dzp

            dza = 0.5 * abs(dzp)  # Initial anchor point "small tension" target value

            # Handle Rotation Releases
            if plate.release_mx and plate.release_my:
                dx = plate.xy_anchors[:, 0] - plate.x0
                dy = plate.xy_anchors[:, 1] - plate.y0
                delta = dza - dzp

                # Safe division: skip entries with zero denominator
                with np.errstate(divide='ignore', invalid='ignore'):
                    ryp = np.divide(-0.5 * delta, dx, where=dx != 0)
                    rxp = np.divide(0.5 * delta, dy, where=dy != 0)

                # Filter out any NaNs that might remain due to zero division
                ryp = ryp[~np.isnan(ryp)]
                rxp = rxp[~np.isnan(rxp)]

                if rxp.size > 0:
                    u[dof_map[3]] = np.mean(rxp)
                if ryp.size > 0:
                    u[dof_map[4]] = np.mean(ryp)

            elif plate.release_mx and not plate.release_my:
                dy = plate.xy_anchors[:, 1] - plate.y0
                dx = plate.xy_anchors[:, 0] - plate.x0
                delta = dza - dzp + ry * dx

                with np.errstate(divide='ignore', invalid='ignore'):
                    rxp = np.divide(delta, dy, where=dy != 0)

                rxp = rxp[~np.isnan(rxp)]
                if rxp.size > 0:
                    u[dof_map[3]] = np.mean(rxp)

            elif plate.release_my and not plate.release_mx:
                dy = plate.xy_anchors[:, 1] - plate.y0
                dx = plate.xy_anchors[:, 0] - plate.x0
                delta = dza - dzp - rx * dy

                with np.errstate(divide='ignore', invalid='ignore'):
                    ryp = np.divide(delta, -dx, where=dx != 0)

                ryp = ryp[~np.isnan(ryp)]
                if ryp.size > 0:
                    u[dof_map[4]] = np.mean(ryp)
        return u

    def _p_proportional_dof_guess_OBSOLETE(self, theta):
        """ Determines the load vector corresponding to theta and returns an initial dof array proportional to p"""
        p = self.get_load_vector(theta)
        u = np.zeros(self.n_dof)
        p_unit_disp = p[0:3] if p[0:3].sum() == 0 else p[0:3] / np.linalg.norm(p[0:3])
        p_unit_rot = p[3:6] if p[0:6].sum() == 0 else p[3:6] / np.linalg.norm(p[3:6])
        dx, dy, dz = 1e-6 * p_unit_disp

        '''todo: Refine initial uplift guess by considering aspect ratio and ratio of overturning to restoring moment.
         Heuristically, item centroid must uplift for tipping about edge. Centerline is set to uplift'''
        delta_x = 0.5*self.Bx-self.ex if (theta<=np.pi/2 or theta >= 3*np.pi/2) else 0.5*self.Bx+self.ex
        delta_y = 0.5*self.By-self.ey if 0<=theta<=np.pi else 0.5*self.By+self.ey
        l_from_x = np.inf if np.isclose(np.cos(theta), 0) else delta_x / np.cos(theta)
        l_from_y = np.inf if np.isclose(np.sin(theta), 0) else delta_y / np.sin(theta)
        l_ot_approx = min(abs(l_from_x),abs(l_from_y))
        m_ot = (p[3]**2+p[4]**2)**0.5
        net_ot_approx = m_ot + p[2]*l_ot_approx
        if net_ot_approx/m_ot > 0.1:
            dz = abs(dz)
            # rx = ry = rz = 0



        rx, ry, rz = 1e-7 * p_unit_rot

        u[0:6] = [dx, dy, dz, rx, ry, rz]

        # Compute Floor Plate DOF initial guesses
        for dof_map, plate in zip(self.floor_plate_dofs, self.floor_plates):
            dzp = dz + plate.yc * rx - plate.xc * ry  # z-displacement at (xc,yc)

            # # Handle Vertical Releases
            # if (plate.release_zp and not plate.release_zn and dzp >=0) or (plate.release_zn and not plate.release_zp and dzp <0):
            #     if plate.release_mx:
            #         u[dof_map[3]] = 0
            #     if plate.release_my:
            #         u[dof_map[4]] = 0
            #     return u

            # Handle Independent Vertical DOF (Triggered when both zn and zp releases are present)
            if plate.release_zp and plate.release_zn:
                # dzp = max(0, 0.5*dzp)
                dzp = 0.5 * abs(dzp)
                # dzp = 1e-7
                u[dof_map[2]] = dzp

            dza = 0.5 * abs(dzp)  # Initial anchor point "small tension" target value

            # Handle Rotation Releases
            if plate.release_mx and plate.release_my:
                dx = plate.xy_anchors[:, 0] - plate.x0
                dy = plate.xy_anchors[:, 1] - plate.y0
                delta = dza - dzp

                # Safe division: skip entries with zero denominator
                with np.errstate(divide='ignore', invalid='ignore'):
                    ryp = np.divide(-0.5 * delta, dx, where=dx != 0)
                    rxp = np.divide(0.5 * delta, dy, where=dy != 0)

                # Filter out any NaNs that might remain due to zero division
                ryp = ryp[~np.isnan(ryp)]
                rxp = rxp[~np.isnan(rxp)]

                if rxp.size > 0:
                    u[dof_map[3]] = np.mean(rxp)
                if ryp.size > 0:
                    u[dof_map[4]] = np.mean(ryp)

            elif plate.release_mx and not plate.release_my:
                dy = plate.xy_anchors[:, 1] - plate.y0
                dx = plate.xy_anchors[:, 0] - plate.x0
                delta = dza - dzp + ry * dx

                with np.errstate(divide='ignore', invalid='ignore'):
                    rxp = np.divide(delta, dy, where=dy != 0)

                rxp = rxp[~np.isnan(rxp)]
                if rxp.size > 0:
                    u[dof_map[3]] = np.mean(rxp)

            elif plate.release_my and not plate.release_mx:
                dy = plate.xy_anchors[:, 1] - plate.y0
                dx = plate.xy_anchors[:, 0] - plate.x0
                delta = dza - dzp - rx * dy

                with np.errstate(divide='ignore', invalid='ignore'):
                    ryp = np.divide(delta, -dx, where=dx != 0)

                ryp = ryp[~np.isnan(ryp)]
                if ryp.size > 0:
                    u[dof_map[4]] = np.mean(ryp)
        return u

    def initial_stiffness_matrix_UNUSED(self):
        """ Calculates an initial stiffness matrix based on full compression of bearing areas and tension of anchors."""
        u = np.zeros(self.n_dof)
        k = sum(element.get_element_stiffness_matrix(u, initial=True) for element in self.floor_plates)

        if self.base_straps:
            k_straps = sum(element.get_element_stiffness_matrix(u, initial=True) for element in self.base_straps)
            k += k_straps

        # add brackets to this function and or modify update_stiffness_matrix
        # k_brackets = [element.get_element_stiffness_matrix(u)[0] for element in self.wall_brackets if
        #               element.get_element_stiffness_matrix(u)[0] is not None]
        #
        # if k_brackets:
        #     k[:6, :6] += np.sum(k_brackets, axis=0)

        return k

    def update_stiffness_matrix(self, u):
        k = np.zeros((self.n_dof, self.n_dof))

        k += sum(element.get_element_stiffness_matrix(u) for element in self.floor_plates)
        k += sum(element.get_element_stiffness_matrix(u) for element in self.base_straps)

        k_brackets = [element.get_element_stiffness_matrix(u)[0] for element in self.wall_brackets if
                      element.get_element_stiffness_matrix(u)[0] is not None]

        if k_brackets:
            k[:6, :6] += np.sum(k_brackets, axis=0)

        return k

    def equilibrium_residual_6_dof(self, u, p, stiffness_update_threshold=0.01, penalty_factor=1e3, verbose=False):
        """Returns the residual for the equilibrium equation p = ku
        stiffness_update_threshold indicates percentage of dof norm change at which stiffness matrix should be updated.
        pentalty factor is applided to zero-force dof residuals to ensure better convergence"""


        # load_vector_weights = np.ones(self.n_dof)
        # load_vector_weights += 20 * np.array(self.disp_dofs)
        load_vector_weights = np.array([20, 20, 20, 1, 1, 1])
        disp_vector_weights = np.array([1, 1, 1, 0.1, 0.1, 0.1])

        # Penalty to prevent "run-away" dofs
        disp_limit = 10
        rotation_limit = 1

        # disp_indices = np.where(np.array(self.disp_dofs) == 1)[0]
        # rot_indices = np.where(np.array(self.disp_dofs) == 0)[0]

        # Extract corresponding u values
        disp_values = u[0:3]
        rot_values = u[3:6]

        # Apply penalty logic
        disp_excess = np.maximum(0, np.abs(disp_values) - disp_limit)
        rot_excess = np.maximum(0, np.abs(rot_values) - rotation_limit)

        # Define a quadratic penalty for each DOF
        penalty_disp = 1e6 * disp_excess ** 2
        penalty_rot = 1e6 * rot_excess ** 2
        penalty_vector = np.zeros(6)
        penalty_vector[0:3] = penalty_disp
        penalty_vector[3:6] = penalty_rot

        # Penalty on zero-force DOFs
        zero_force_mask = np.isclose(p, 0, 1e-10)
        load_vector_weights[zero_force_mask] = load_vector_weights[zero_force_mask]*penalty_factor

        # Solve for independent DOFs
        self._u_full = np.zeros(self.n_dof)
        self._u_full[0:6] = u
        for dof_map, plate in zip(self.floor_plate_dofs, self.floor_plates):
            u_free, success = plate.solve_released_dof_equilibrium(self._u_full, dof_map,
                                                                   u_free_init=self.u_init[dof_map[plate.free_dofs]])
            if success:
                self._u_full[dof_map[plate.free_dofs]] = u_free
            else:
                print('Warning: base plate free dof failed to converge')

        # Update Stiffness Matrix for large change in norm of u or if any base anchors have reversed signs
        if self.K is None or self._u_previous is None:
            update_K = True  # First iteration, always update K
        else:
            norm_change = np.linalg.norm(disp_vector_weights * u - disp_vector_weights * self._u_previous[0:6]) / \
                          (np.linalg.norm(disp_vector_weights * self._u_previous[0:6]) + 1e-12)
            norm_change_trigger = norm_change > stiffness_update_threshold
            update_K = norm_change_trigger  # or anchor_change_trigger

        self.residual_call_count += 1
        if self.residual_call_count % 10 == 0:
            update_K = True

        if update_K:
            self.K = self.update_stiffness_matrix(self._u_full)
            self._u_previous = self._u_full.copy()

        residual = load_vector_weights * (np.dot(self.K[0:6,:], self._u_full) - p) + penalty_vector
        if verbose:
            print(f'Norm: {np.linalg.norm(residual):.3e}, Update K: {update_K}, Penalties: {penalty_vector}')
        return residual

    def equilibrium_residual(self, u, p, stiffness_update_threshold=0.01, penalty_factor=1e3, verbose=False):
        """Returns the residual for the equilibrium equation p = ku
        stiffness_update_threshold indicates percentage of dof norm change at which stiffness matrix should be updated.
        pentalty factor is applided to zero-force dof residuals to ensure better convergence"""

        load_vector_weights = np.ones(self.n_dof)
        load_vector_weights += 20 * np.array(self.disp_dofs)

        disp_vector_weights = np.ones(self.n_dof)
        # disp_vector_weights[0:3] = [0.1, 0.1, 0.1]

        # Update Stiffness Matrix for large change in norm of u or if any base anchors have reversed signs
        if self.K is None or self._u_previous is None:
            update_K = True  # First iteration, always update K
        else:
            norm_change = np.linalg.norm(disp_vector_weights * u - disp_vector_weights * self._u_previous) / \
                          (np.linalg.norm(disp_vector_weights * self._u_previous) + 1e-12)
            norm_change_trigger = norm_change > stiffness_update_threshold

            # num_base_anchors_in_tension = 0 if self.base_anchors is None else sum(
            #     Utilities.vertical_point_displacements(self.base_anchors.xy_anchors, u) > 0)
            # anchor_change_trigger = num_base_anchors_in_tension != self._num_base_anchors_in_tension

            update_K = norm_change_trigger  # or anchor_change_trigger
            # self._num_base_anchors_in_tension = num_base_anchors_in_tension.copy()

        self.residual_call_count+=1
        if self.residual_call_count % 10 == 0:
            update_K = True

        if update_K:
            self.K = self.update_stiffness_matrix(u)
            self._u_previous = u.copy()

        # Penalty to prevent "run-away" dofs
        disp_limit = 10
        rotation_limit = 1

        disp_indices = np.where(np.array(self.disp_dofs) == 1)[0]
        rot_indices = np.where(np.array(self.disp_dofs) == 0)[0]

        # Extract corresponding u values
        disp_values = u[disp_indices]
        rot_values = u[rot_indices]

        # Apply penalty logic
        disp_excess = np.maximum(0, np.abs(disp_values) - disp_limit)
        rot_excess = np.maximum(0, np.abs(rot_values) - rotation_limit)

        # Define a quadratic penalty for each DOF
        penalty_disp = 1e6 * disp_excess ** 2
        penalty_rot = 1e6 * rot_excess ** 2
        penalty_vector = np.zeros(self.n_dof)
        penalty_vector[disp_indices] = penalty_disp
        penalty_vector[rot_indices] = penalty_rot

        # Pentalty on zero-force dofs
        zero_force_mask = np.isclose(p, 0, 1e-10)
        load_vector_weights[zero_force_mask] *= penalty_factor
        residual = load_vector_weights * (np.dot(self.K, u) - p) + penalty_vector
        if verbose:
            print(f'Norm: {np.linalg.norm(residual):.3e}, Update K: {update_K}, Penalties: {penalty_vector}')
        return residual

    def solve_equilibrium_Newton_Krylov(self, u_init, p):
        """ Solve the nonlinear equilibrium equation with Newton-Krylov solver. """
        sol = newton_krylov(lambda u: self.equilibrium_residual(u, p), u_init, method='lgmres')
        return sol, np.linalg.norm(self.equilibrium_residual(sol, p)) < 1e-6

    def solve_equilibrium_6_dof(self, p):
        p_primary = p[0:6]
        u_primary_init = self.u_init[0:6]
        res = None
        methods = ['hybr']
        for method in methods:
            self._u_previous = None
            self.residual_call_count = 0
            res = root(self.equilibrium_residual, u_primary_init, args=p_primary, method=method, #xtol=1e-8,
                       options={'maxfev': 200,'xtol':1e-6})
            if res.success:
                # print(f'Success with {res.nfev} function calls')
                break
        self._u_full[0:6] = res.x
        sol = self._u_full
        return sol, res.success

    def solve_equilibrium(self, u_init, p):
        methods = ['hybr']
        for method in methods:
            self._u_previous = None
            self.residual_call_count = 0
            res = root(self.equilibrium_residual, u_init, args=p, method=method, #xtol=1e-8,
                       options={'maxfev': 200,'xtol':1e-6})
            if res.success:
                # print(f'Success with {res.nfev} function calls')
                break
        sol = res.x
        return sol, res.success

    def solve_equilibrium_timed(self, u_init, p, timeout=10):
        # Wrap the instance method to make it picklable for multiprocessing
        residual_func = partial(self.equilibrium_residual, verbose=False)

        # Call timeout-enabled solver
        sol, success = solve_equilibrium_with_timeout(
            residual_func,
            u_init,
            p,
            timeout=timeout,
            method='hybr',
            options={'maxfev': 200, 'xtol': 1e-6}
        )
        return sol, success

    def solve_equilibrium_broyden_OLD(self, u_init, p):
        sol = broyden1(lambda u_i: self.equilibrium_residual(u_i, p), u_init, f_tol=1e-1, verbose=False)
        success = np.linalg.norm(self.equilibrium_residual(sol, p)) < 1e-1
        return sol, success

    def analyze_model(self, initial_solution_cache, verbose=False):
        """Applies Horizontal Loads at all angles, solves for equilibrium and stores the solution displacements"""
        # Initialize Analysis Range
        num_theta_z = 4 * 8 + 1
        self.theta_z = np.linspace(0, 2 * math.pi, num_theta_z)
        self.converged = []
        self.equilibrium_solutions = np.zeros((self.n_dof, len(self.theta_z)))

        try_previous_converged = False
        u_prev = np.zeros(self.n_dof)
        # Analysis Attempt with Initial DOF Guesses
        for i, t in enumerate(self.theta_z):
            p = self.get_load_vector(t)

            u_tries = self.get_initial_dof_guess(t)
            if  try_previous_converged:
                u_tries = [u_prev] + u_tries
            if initial_solution_cache is not None:
                u_tries = [initial_solution_cache[:,i]] + u_tries

            success = False
            for j, u_init in enumerate(u_tries):
                # if verbose:
                #     print(f'Theta {np.degrees(t):.2f} trying with u guess {j}')
                self.u_init = u_init
                sol, success = self.solve_equilibrium(u_init, p)
                self.equilibrium_solutions[:, i] = sol  # todo: this line is for testing. Delete and uncomment below
                if success:
                    if verbose:
                        print(f'Theta {np.degrees(t):.0f} success with initial u guess {j}')
                    # self.equilibrium_solutions[:, i] = sol
                    u_prev = sol
                    try_previous_converged = True
                    break

            if not success and verbose:
                pass
                # print(f'Theta {np.degrees(t):.0f} UNCONVERGED with initial u guess {j}')

            self.converged.append(success)

        # Secondary Analysis Attempt to "Fill-in" Failed Convergence Points
        while any(self.converged) and not all(self.converged):
            if verbose:
                print('Attempting to reanalyze unconverged points...')
            unconverged = [i for i, suc in enumerate(self.converged) if not suc]
            new_converge_found = False
            for idx in unconverged:
                t = self.theta_z[idx]
                p = self.get_load_vector(t)
                u_tries = self.get_interpolated_dof_guess(idx)
                success = False
                for j, u_init in enumerate(u_tries):
                    self.u_init = u_init
                    sol, success = self.solve_equilibrium(u_init, p)
                    if success:
                        if verbose:
                            print(f'Theta {np.degrees(t):.0f} success with interpolated u guess {j}')
                        self.equilibrium_solutions[:, idx] = sol
                        new_converge_found = True
                        break
                self.converged[idx] = success
                if not success and verbose:
                    print(f'Theta {np.degrees(t):.0f} UNCONVERGED with interpolated u guess {j}')
            if not new_converge_found:
                break
        # Compile arrays of element forces
        if not any(self.converged):
            raise Exception(f'Item {self.equipment_id} failed to converge to any solutions. Check geoemtry definition.')
        self.get_element_results()

    def get_element_results(self):
        # Initialize Base Anchor Results Array
        if self.base_anchors is not None:
            self.base_anchors.anchor_forces = np.zeros((len(self.base_anchors.xy_anchors), len(self.theta_z), 3))

        # Initialize Wall Bracket Results Array
        if self.wall_brackets:
            self.wall_bracket_forces = np.zeros(
                (len(self.wall_brackets), len(self.theta_z), 3))  # n_bracket, n_theta, (N,P,Z)-forces

        # Initialize Wall Anchors Results Array
        ''' When concrete/cmu anchors are used, the anchor's object is assigned as an attribute of the model.
                    When sms/wood anchors are used, the anchor's objects are assigned as an attribute of the backing elements.
                    This is because concrete anchors must consider spacing requirements (and thus include all wall anchors
                    in a single object). However, sms anchors must consider pyring conditions caused by unistrut backing,
                    and so must be considered separately backing-by-backing.'''
        for position, wall_anchors in self.wall_anchors.items():
            if wall_anchors is not None:
                wall_anchors.anchor_forces = np.zeros((wall_anchors.xy_anchors.shape[0], len(self.theta_z), 3))

        for backing in self.wall_backing:
            if backing.anchors_obj is not None:
                backing.anchors_obj.anchor_forces = np.zeros((backing.pz_anchors.shape[0],len(self.theta_z), 3))

        for i, sol in enumerate(self.equilibrium_solutions.T):

            # Update Element Resultants
            self.update_element_resultants(sol)

            # Extract Base Anchor Results
            if self.base_anchors is not None:
                self.base_anchors.anchor_forces[:, i, 0] = np.concatenate(
                    [el.anchor_result['tension'] for el in self.floor_plates if el.n_anchor > 0], axis=0)
                self.base_anchors.anchor_forces[:, i, 1] = np.concatenate(
                    [el.anchor_result['vx'] for el in self.floor_plates if el.n_anchor > 0], axis=0)
                self.base_anchors.anchor_forces[:, i, 2] = np.concatenate(
                    [el.anchor_result['vy'] for el in self.floor_plates if el.n_anchor > 0], axis=0)

            # Extract Wall Bracket Results
            for b, element in enumerate(self.wall_brackets):
                self.wall_bracket_forces[b, i, 0] = element.bracket_forces['fn']
                self.wall_bracket_forces[b, i, 1] = element.bracket_forces['fp']
                self.wall_bracket_forces[b, i, 2] = element.bracket_forces['fz']

            # Extract Wall Anchor Results
            for wall_loc, wall_anchors in self.wall_anchors.items():
                if wall_anchors is not None:
                    forces = np.concatenate(
                        [el.anchor_forces for el in [self.wall_backing[idx] for idx in self.backing_indices[wall_loc]]],
                        axis=0)
                    wall_anchors.anchor_forces[:, i, :] = forces

            for backing in self.wall_backing:
                if backing.anchors_obj is not None:
                    backing.anchors_obj.anchor_forces[:, i, :] = backing.anchor_forces

        self.get_governing_solutions()

    def get_governing_solutions(self):
        """Post Processes the element forces to identify the maximum demand and angle of loading"""



        # Base Anchors
        if self.base_anchors is not None:
            # Tension
            matrix = self.base_anchors.anchor_forces[:, self.converged, 0]
            idx_anchor, idx_theta = get_max_demand(matrix)
            self.governing_solutions['base_anchor_tension']['theta_z'] = self.theta_z[self.converged][idx_theta]
            self.governing_solutions['base_anchor_tension']['sol'] = self.equilibrium_solutions[:,self.converged][:, idx_theta]

            # Shear  # todo: this may need to be modified for the four shear cases xp, xn, yp, yn
            matrix = (self.base_anchors.anchor_forces[:, self.converged, 1] ** 2 + self.base_anchors.anchor_forces[:, self.converged,
                                                                      2] ** 2) ** 0.5
            idx_anchor, idx_theta = get_max_demand(matrix)
            self.governing_solutions['base_anchor_shear']['theta_z'] = self.theta_z[self.converged][idx_theta]
            self.governing_solutions['base_anchor_shear']['sol'] = self.equilibrium_solutions[:,self.converged][:, idx_theta]

        # Wall Brackets
        if self.wall_brackets:
            # Tension
            matrix = self.wall_bracket_forces[:, self.converged, 0]
            idx_anchor, idx_theta = get_max_demand(matrix)
            self.governing_solutions['wall_bracket_tension']['theta_z'] = self.theta_z[self.converged][idx_theta]
            self.governing_solutions['wall_bracket_tension']['sol'] = self.equilibrium_solutions[:,self.converged][:, idx_theta]

            # Shear
            matrix = (self.wall_bracket_forces[:, self.converged, 1] ** 2 + self.wall_bracket_forces[:, self.converged, 2] ** 2) ** 0.5
            idx_anchor, idx_theta = get_max_demand(matrix)
            self.governing_solutions['wall_bracket_shear']['theta_z'] = self.theta_z[self.converged][idx_theta]
            self.governing_solutions['wall_bracket_shear']['sol'] = self.equilibrium_solutions[:,self.converged][:, idx_theta]

            # Wall Anchors
            t_max = -np.inf
            v_max = -np.inf
            all_wall_anchors = [anchors_obj for wall_position, anchors_obj in self.wall_anchors.items()
                                if anchors_obj is not None] + [b.anchors_obj for b in self.wall_backing if b.anchors_obj is not None]
            for wall_anchors in all_wall_anchors:
                # Tension
                matrix = wall_anchors.anchor_forces[:, self.converged, 0]
                idx_anchor, idx_theta = get_max_demand(matrix)
                t_new = matrix[idx_anchor, idx_theta]
                if t_new > t_max:
                    t_max = t_new.copy()
                    self.governing_solutions['wall_anchor_tension']['theta_z'] = self.theta_z[self.converged][idx_theta]
                    self.governing_solutions['wall_anchor_tension']['sol'] = self.equilibrium_solutions[:,self.converged][:, idx_theta]

                # Shear
                matrix = (wall_anchors.anchor_forces[:, self.converged, 1] ** 2 + wall_anchors.anchor_forces[:, self.converged,
                                                                     2] ** 2) ** 0.5
                idx_anchor, idx_theta = get_max_demand(matrix)
                v_new = matrix[idx_anchor, idx_theta]
                if v_new > v_max:
                    v_max = v_new.copy()
                self.governing_solutions['wall_anchor_shear']['theta_z'] = self.theta_z[self.converged][idx_theta]
                self.governing_solutions['wall_anchor_shear']['sol'] = self.equilibrium_solutions[:,self.converged][:, idx_theta]

    def update_element_resultants(self, sol):
        for element in self.floor_plates:
            # todo: [Refinement] revise the three methods below, so that matrices are not re-computed for each method. Consider adding an update_state(u) method to be called before the get_resultant methods
            element.get_connection_forces(sol)
            element.get_anchor_resultants(sol)
            element.get_compression_resultants(sol)

            if element.connection:
                element.connection.get_anchor_forces(*[-1 * val for val in element.connection_forces],
                                                     convert_to_local=True)
                element.connection.anchors_obj.check_anchors(self.asd_lrfd_ratio)

        for element in self.base_straps:
            element.check_brace(sol, self.asd_lrfd_ratio)

        for element in self.wall_brackets:
            element.get_element_forces(sol)
            element.check_brackets()

            if element.connection:
                element.connection.get_anchor_forces(*element.reactions_equipment,convert_to_local=True)
                element.connection.anchors_obj.check_anchors(self.asd_lrfd_ratio)

        for wall_loc, indices in self.backing_indices.items():
            for el in [self.wall_backing[idx] for idx in indices]:
                el.get_anchor_forces([self.wall_brackets[i] for i in el.bracket_indices])


class FloorPlateElement:
    def __init__(self, bearing_boundaries, xy_anchors=None,
                 x0=0.0, y0=0.0, z0=0.0,
                 xc=0.0, yc=0.0, zc=0.0,
                 release_mx=False, release_my=False, release_mz=False,
                 release_xp=False, release_xn=False,
                 release_yp=False, release_yn=False, release_zp=False, release_zn=False,
                 E_base=None, poisson=None):

        # Geometry
        self.bearing_boundaries = bearing_boundaries

        if xy_anchors is None:
            self.xy_anchors = np.array([])
            self.n_anchor = 0
        else:
            self.xy_anchors = xy_anchors
            self.n_anchor = np.shape(self.xy_anchors)[0]

        # Material Properties
        self.anchor_shear_stiffness = None
        self.anchor_tension_stiffness = None
        self.E_base = E_base  # Elastic modulus of base material
        self.poisson = poisson  # Poisson ratio of base material

        # Attachment Fasteners
        self.connection = None

        # DOF and Constraint Parameters
        self.x0 = x0  # Inflection point of element
        self.y0 = y0  # Inflection point of element
        self.z0 = z0  # Inflection point of element

        self.xc = xc  # Connection Point of element
        self.yc = yc  # Connection Point of element
        self.zc = zc  # Connection Point of element

        self.release_mx = release_mx  # Moment release at attachment point about x-axis
        self.release_my = release_my  # Moment release at attachment point about y-axis
        self.release_mz = release_mz
        self.release_xp = release_xp
        self.release_xn = release_xn
        self.release_yp = release_yp
        self.release_yn = release_yn
        self.release_zp = release_zp
        self.release_zn = release_zn

        self.free_dofs = []
        self.constrained_dofs = []

        # Stiffness Matrix Parameters
        self.C = None  # Constraint Matrix (global DOFs to local DOFs)
        self.B = np.array([[1, 0, 0, 0, 0, 0],  # Static Conversion Matrix (dofs to connection point)
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, (self.zc - self.z0), -(self.yc - self.y0), 1, 0, 0],
                           [-(self.zc - self.z0), 0, (self.xc - self.x0), 0, 1, 0],
                           [(self.yc - self.y0), -(self.xc - self.x0), 0, 0, 0, 1]])

        # self.ka_template = None  # ka with "kz" value set to 1
        # self.ka = None  # Stiffness sub-matrix for base_anchors
        # self.kb = None  # Stiffness sub-matrix for bearing areas

        # State Parameters
        self.cz_result = {}  # Dictionary of compression zone geometries and resultants
        self.anchor_result = {}  # Dictionary of anchor force resultants
        self.nodal_forces = None
        self.connection_forces = None

    def set_dof_constraints(self, n_dof, dof_map):
        """Initializes static (displacement-independent) matrices given the total global dofs and dof_map.
         dof_map is a six element array indicating the index of the global DOFs at each local dofs"""

        """ Returns the constraint matrix (6 x n_dof) relating global DOFs to 6 element DOFs"""

        ''' Presence of vertical releases in both z+ and z- directions results in uncoupling of base plate vertical
        translation as a unique degree of freedom.
        Shear releases do not result in unique degrees of freedom, but rather are handled by modifying the shear
        of the anchors so that resulting plots will show floor plates kinematically constrained to the unit without
        imparting stiffness.'''

        self.C = np.zeros((6, n_dof))
        self.C[0, 0:6] = [1, 0, 0, 0, self.z0, -self.y0]
        self.C[1, 0:6] = [0, 1, 0, -self.z0, 0, self.x0]
        if dof_map[2] != 2:
            self.C[2, dof_map[2]] = 1

        else:
            self.C[2, 0:6] = [0, 0, 1, self.yc, -self.xc, 0]
        # todo: verify and revise. There should be coupling for rotation dofs when vertical dof is free on baseplate
        self.C[3, dof_map[3]] = 1
        self.C[4, dof_map[4]] = 1
        self.C[5, dof_map[5]] = 1

        self.free_dofs = [i for i, dof in enumerate(dof_map) if i != dof]
        self.constrained_dofs = [i for i, dof in enumerate(dof_map) if i == dof]

    def set_anchor_properties(self, anchors_object):
        """ Updates the anchor stiffness array [kx, ky, kz], and base material properties"""
        self.anchor_shear_stiffness = anchors_object.Kv
        self.anchor_tension_stiffness = anchors_object.K

    def set_sms_attachment_UNUSED(self, xy_anchors, ):
        self.connection = SMSAnchors()
        self.connection.set_sms_properties()  # todo [Attachments, pass actual equipment values here]

    def update_anchor_stiffness_matrix(self, u, initial=False):
        """ Computes and returns the anchor stiffness matrix without modifying self. """

        # Initialize stiffness values
        kx = np.full(self.n_anchor, self.anchor_shear_stiffness)
        ky = np.full(self.n_anchor, self.anchor_shear_stiffness)
        kz = np.full(self.n_anchor, self.anchor_tension_stiffness)

        if not initial:
            # Compute displacements
            xyz_anchors = np.column_stack((self.xy_anchors, np.zeros(self.n_anchor)))
            delta = Utilities.compute_point_displacements(xyz_anchors, self.C @ u, x0=self.x0, y0=self.y0, z0=self.z0)

            # Vectorized release conditions
            kx *= ~((self.release_xp & (delta[:, 0] >= 0)) | (self.release_xn & (delta[:, 0] < 0)))
            ky *= ~((self.release_yp & (delta[:, 1] >= 0)) | (self.release_yn & (delta[:, 1] < 0)))
            if not(self.release_zn and self.release_zp):  # both releasees triggers independent vertical plate dof
                kz *= (not self.release_zp) & (delta[:, 2] >= 0)
            else:
                kz *= (delta[:, 2] >= 0)

        # Vectorized matrix assembly
        x, y = self.xy_anchors[:, 0], self.xy_anchors[:, 1]
        ka = np.zeros((6, 6, self.n_anchor))  # Local variable instead of modifying self.ka

        ka[0, 0, :] = kx
        ka[0, 4, :] = ka[4, 0, :] = -kx * self.z0
        ka[0, 5, :] = ka[5, 0, :] = -kx * (y - self.y0)

        ka[1, 1, :] = ky
        ka[1, 3, :] = ka[3, 1, :] = ky * self.z0
        ka[1, 5, :] = ka[5, 1, :] = ky * (x - self.x0)

        ka[2, 2, :] = kz
        ka[2, 3, :] = ka[3, 2, :] = kz * (y - self.y0)
        ka[2, 4, :] = ka[4, 2, :] = -kz * (x - self.x0)

        ka[3, 3, :] = kz * (y - self.y0) ** 2 + ky * self.z0 ** 2
        ka[3, 4, :] = ka[4, 3, :] = -kz * (x - self.x0) * (y - self.y0)
        ka[3, 5, :] = ka[5, 3, :] = ky * (x - self.x0) * self.z0

        ka[4, 4, :] = kz * (x - self.x0) ** 2 + kx * self.z0 ** 2
        ka[4, 5, :] = ka[5, 4, :] = kx * (y - self.y0) * self.z0

        ka[5, 5, :] = ky * (x - self.x0) ** 2 + kx * (y - self.y0) ** 2

        return ka

    def get_compression_zone_properties(self, u, full_bearing=False):
        """Returns the geometric properties of the compression zones for given dof values, u"""
        # Identify area of bearing elements that are in compression
        # Loop trough each bearing boundary, then loop through each compression area and assemble list.

        # Case 1, no moment releases present:
            # Turn off all brearing
        # Case 2, moment releases are present:
            # Turn off bearing only if node is negatively displaced

        if (self.release_zn and not self.release_zp) and not (self.release_mx or self.release_my):
            release_bearing = True
        elif (self.release_zn and not self.release_zp) and (Utilities.vertical_point_displacements(np.array([[self.x0,self.y0,self.z0]]), u)[0]<=0):
            release_bearing = True
        else:
            release_bearing = False

        if release_bearing:
            compression_boundaries = []
        elif full_bearing:
            u0 = np.array([0, 0, -1e-8, 0, 0, 0])
            compression_boundaries = [
                compression_boundary
                for boundary in self.bearing_boundaries
                for compression_boundary in
                Utilities.bearing_area_in_compression(boundary, u0, x0=self.x0, y0=self.y0)]
        else:
            compression_boundaries = [
                compression_boundary
                for boundary in self.bearing_boundaries
                for compression_boundary in
                Utilities.bearing_area_in_compression(boundary, self.C @ u, x0=self.x0, y0=self.y0)]

        # Get stiffness and inertial properties for compression areas
        n_boundaries = len(compression_boundaries)
        areas = np.empty(n_boundaries)
        centroids = np.empty((n_boundaries, 2))
        Ixx_list = np.empty(n_boundaries)
        Iyy_list = np.empty(n_boundaries)
        Ixy_list = np.empty(n_boundaries)
        beta_list = np.empty(n_boundaries)

        for i, vertices in enumerate(compression_boundaries):
            area, centroid, Ixx, Iyy, Ixy = Utilities.polygon_properties(vertices)
            beta = Utilities.effective_indenter_stiffness(area, self.E_base, self.poisson)

            areas[i] = area
            centroids[i] = centroid
            Ixx_list[i] = Ixx
            Iyy_list[i] = Iyy
            Ixy_list[i] = Ixy
            beta_list[
                i] = beta if not full_bearing else beta / 10  # heuristic to reduce stiffness for initial stiffness matrix

        self.cz_result = {'compression_boundaries': compression_boundaries,
                          'areas': areas,
                          'centroids': centroids,
                          'Ixx': Ixx_list,
                          'Iyy': Iyy_list,
                          'Ixy': Ixy_list,
                          'beta': beta_list}

    def update_bearing_stiffness_matrix(self, u, initial=False):
        """ Computes and returns the bearing stiffness matrix without modifying self. """

        # Get compression zones (consider caching this if u doesn't change significantly)
        if initial:
            self.get_compression_zone_properties(None, full_bearing=True)
        else:
            self.get_compression_zone_properties(u)

        # If there are no compression zones, return a zero matrix
        if len(self.cz_result['areas']) == 0:
            return np.zeros((6, 6))

        # Precompute compression zone stiffness matrices
        kb_list = [
            self.compression_zone_matrix(A, x_bar, y_bar, Ixx, Iyy, Ixy, beta, self.x0, self.y0)
            for A, (x_bar, y_bar), Ixx, Iyy, Ixy, beta in zip(
                self.cz_result['areas'],
                self.cz_result['centroids'],
                self.cz_result['Ixx'],
                self.cz_result['Iyy'],
                self.cz_result['Ixy'],
                self.cz_result['beta'])
        ]

        # Efficiently sum all stiffness matrices
        kb = np.sum(kb_list, axis=0)

        return kb

    @staticmethod
    def compression_zone_matrix(A, x_bar, y_bar, Ixx, Iyy, Ixy, beta, x0=0.0, y0=0.0):
        """Returns the local bearing stiffness matrix for a single compression zone"""
        k_cz = np.zeros((6, 6))

        # Assign values according to the LaTeX expressions (note the zero-indexing in Python)
        k_cz[2, 2] = beta * A
        k_cz[2, 3] = k_cz[3, 2] = beta * (y_bar - y0) * A
        k_cz[2, 4] = k_cz[4, 2] = -beta * (x_bar - x0) * A
        k_cz[3, 3] = beta * (Ixx - 2 * y0 * y_bar * A + y0 ** 2 * A)
        k_cz[3, 4] = k_cz[4, 3] = -beta * (Ixy - y0 * x_bar * A - x0 * y_bar * A + x0 * y0 * A)
        k_cz[4, 4] = beta * (Iyy - 2 * x0 * x_bar * A + x0 ** 2 * A)

        return k_cz

    def get_element_stiffness_matrix_OLD(self, u):
        """ Assembles the element stiffness matrix from anchor and bearing matrices,
        Here, this refers to the stiffness matrix relating global dofs to global forces"""
        has_anchors = len(self.xy_anchors) > 0
        has_bearing = len(self.bearing_boundaries) > 0

        if len(self.xy_anchors) > 0 and len(self.bearing_boundaries) > 0:
            self.update_anchor_stiffness_matrix(u)
            self.update_bearing_stiffness_matrix(u)
            k_element = self.C.T @ (np.sum(self.ka, axis=2) + self.kb) @ self.C
        elif len(self.bearing_boundaries) > 0:
            self.update_bearing_stiffness_matrix(u)
            k_element = self.C.T @ self.kb @ self.C
        elif len(self.xy_anchors) > 0:
            self.update_anchor_stiffness_matrix(u)
            k_element = self.C.T @ np.sum(self.ka, axis=2) @ self.C
        else:
            ndof = self.C.shape[1]
            k_element = np.zeros((ndof, ndof))

        return k_element

    def get_element_stiffness_matrix(self, u, initial=False):
        """ Efficiently assembles the element stiffness matrix without modifying self. """
        has_anchors = len(self.xy_anchors) > 0
        ka = self.update_anchor_stiffness_matrix(u, initial=initial) if has_anchors else np.zeros((6, 6, self.n_anchor))
        kb = self.update_bearing_stiffness_matrix(u, initial=initial)

        # Precompute anchor stiffness sum
        k_anchor = np.sum(ka, axis=2) if has_anchors else np.zeros((6, 6))

        # Use matrix multiplication only once
        k_element = self.C.T @ (k_anchor + kb) @ self.C

        return k_element

    def get_compression_resultants(self, u):
        """Computes the compression zone resultant forces and centroids"""
        self.update_bearing_stiffness_matrix(u)

        if len(self.cz_result['compression_boundaries']) != 0:

            x0 = self.x0
            y0 = self.y0

            fz = np.zeros(len(self.cz_result['compression_boundaries']))
            mx = np.zeros(len(self.cz_result['compression_boundaries']))
            my = np.zeros(len(self.cz_result['compression_boundaries']))

            # get compression zones
            for i, (A, (x_bar, y_bar), Ixx, Iyy, Ixy, beta) in enumerate(zip(
                    self.cz_result['areas'],
                    self.cz_result['centroids'],
                    self.cz_result['Ixx'],
                    self.cz_result['Iyy'],
                    self.cz_result['Ixy'],
                    self.cz_result['beta'])):
                kb_cz = self.compression_zone_matrix(A, x_bar, y_bar, Ixx, Iyy, Ixy, beta, x0, y0)

                p = kb_cz @ self.C @ u

                fz[i] = p[2]
                mx[i] = p[3]
                my[i] = p[4]

            self.cz_result['fz'] = fz
            self.cz_result['resultant_centroids'] = np.column_stack((-my / fz + x0, mx / fz + y0))
        else:
            self.cz_result['fz'] = []
            self.cz_result['resultant_centroids'] = []

    def get_anchor_resultants(self, u):
        """Computes anchor forces"""
        if self.n_anchor > 0:
            ka = self.update_anchor_stiffness_matrix(u)
            p = np.einsum('ijk,jl,l->ik', ka, self.C, u)
            vx = p[0, :]
            vy = p[1, :]
            t = p[2, :]
            tension_resultant = t.sum()
            mx = p[3, :]
            my = p[4, :]

            if not np.isclose(tension_resultant, 0):
                resultant_centroid = np.array([-my.sum() / tension_resultant, mx.sum() / tension_resultant])
            else:
                resultant_centroid = np.nan
        else:
            vx = None
            vy = None
            t = None
            tension_resultant = None
            resultant_centroid = None

        self.anchor_result = {
            'vx': vx,
            'vy': vy,
            'tension': t,
            'tension_resultant': tension_resultant,
            'resultant_centroid': resultant_centroid}

    def get_nodal_forces_OLD(self, u):
        """ Returns the forces at the attachment point of the floor plate element"""
        # Assemble element stiffness matrix (relating global dofs to element basic forces
        if len(self.xy_anchors) > 0 and len(self.bearing_boundaries) > 0:
            self.update_anchor_stiffness_matrix(u)
            self.update_bearing_stiffness_matrix(u)
            k_element = (np.sum(self.ka, axis=2) + self.kb) @ self.C
        elif len(self.bearing_boundaries) > 0:
            self.update_bearing_stiffness_matrix(u)
            k_element = self.kb @ self.C
        elif len(self.xy_anchors) > 0:
            self.update_anchor_stiffness_matrix(u)
            k_element = np.sum(self.ka, axis=2) @ self.C
        else:
            ndof = self.C.shape[1]
            k_element = np.zeros((6, ndof))

        self.nodal_forces = k_element @ u

    def get_nodal_forces(self, u):
        """ Computes and returns the forces at the attachment point of the floor plate element """

        # Check if anchors and/or bearings exist
        has_anchors = len(self.xy_anchors) > 0
        has_bearings = len(self.bearing_boundaries) > 0

        # Compute anchor and bearing stiffness matrices
        ka = self.update_anchor_stiffness_matrix(u) if has_anchors else np.zeros((6, 6, self.n_anchor))
        kb = self.update_bearing_stiffness_matrix(u) if has_bearings else np.zeros((6, 6))

        # Precompute anchor stiffness sum
        k_anchor = np.sum(ka, axis=2) if has_anchors else np.zeros((6, 6))

        # Compute element stiffness matrix
        if has_anchors and has_bearings:
            k_element = (k_anchor + kb) @ self.C
        elif has_bearings:
            k_element = kb @ self.C
        elif has_anchors:
            k_element = k_anchor @ self.C
        else:
            ndof = self.C.shape[1]
            k_element = np.zeros((6, ndof))

        # Compute nodal forces
        self.nodal_forces = k_element @ u

        return

    def get_connection_forces(self, u):
        self.get_nodal_forces(u)

        # Sum nodal forces at connection point
        self.connection_forces = self.B @ self.nodal_forces

    def check_prying_thickness(self):
        """ update element resultants should be called before calling this method"""
        if self.release_mx == 'Check Prying':
            Mu = self.nodal_forces[3]
        elif self.release_my == 'Check Prying':
            Mu = self.nodal_forces[4]
        else:
            return False

        phi = 0.9
        self.tnp = (4*Mu/(phi*self.yield_width*self.fu))
        self.t

    def released_dof_residual(self, u_free, u_global, dof_map):
        # Assemble dof vector
        u = u_global
        u[dof_map[self.free_dofs]] = u_free

        # Update stiffness matrix
        ke = self.get_element_stiffness_matrix(u)

        # Return free-dof residual
        residual = ke[self.free_dofs,:] @ u
        return residual

    def solve_released_dof_equilibrium(self, u_global, dof_map, u_free_init=None, verbose=False):
        if u_free_init is None:
                u_free_init = np.zeros(len(self.free_dofs))

        methods = ['hybr']
        for method in methods:
            res = root(self.released_dof_residual, u_free_init, args=(u_global, dof_map), method=method, #xtol=1e-8,
                       options={'maxfev': 200,'xtol':1e-6})
            if res.success:
                # print(f'Success with {res.nfev} function calls')
                break
        sol = res.x
        if verbose:
            print(res)
        return sol, res.success

if __name__ == '__main__':
    pass


class WallBracketElement:
    def __init__(self, supporting_wall, xyz_equipment: np.array, xyz_wall: np.array,
                 normal_unit_vector, wall_flexibility,
                 horizontal_stiffness=None, vertical_stiffness=None, releases=None,
                 backing_offset_x=0, backing_offset_y=0, backing_offset_z=0):

        # Geometry Attributes in Global Coordinates
        self.xyz_equipment = xyz_equipment  # Coordinate at attachment to equipment
        self.xyz_wall = xyz_wall  # Coordinate at wall along line normal to wall at xyz_equipment
        self.xyz_backing = self.xyz_wall + (backing_offset_x, backing_offset_y, backing_offset_z)
        self.normal = normal_unit_vector

        self.supporting_wall = supporting_wall

        self.length = math.sqrt(sum((f - i) ** 2 for f, i in zip(xyz_equipment, xyz_wall)))

        # Bracket Hardware Properties
        self.bracket_id = None
        self.f_wall = wall_flexibility
        self.f_brace = None
        self.kp = horizontal_stiffness
        self.kz = vertical_stiffness
        self.capacity_method = None
        self.capacity_to_equipment = None
        self.bracket_capacity = None
        self.capacity_to_backing = None
        self.shear_capacity = None
        self.connection = None
        # self.e_cxn = e_cxn  # Eccentricity of bracket center-line normal to connection faying surface

        # Element Analysis Properties
        self.releases = releases

        x, y, z = self.xyz_equipment
        self.C = np.array([[1, 0, 0, 0, z, -y],
                           [0, 1, 0, -z, 0, x],
                           [0, 0, 1, y, -x, 0]])

        nx, ny, nz = self.normal
        self.G = np.array([[nx, ny, 0],
                           [-ny, nx, 0],
                           [0, 0, 1]])

        # Results Properties
        self.connection_forces = None
        self.bracket_forces = {}
        self.reactions_backing = None  # Reactions at backing in global coordinates (fx, fy, fz, mx, my, mz)
        self.reactions_equipment = None  # Reactions at connection in global coordinates (fx, fy, fz, mx, my, mz)
        self.tension_dcr = None

    def set_dof_constraints(self, n_dof):
        pass

    def set_bracket_properties(self, bracket_data, asd_lrfd_ratio):
        self.bracket_id = bracket_data['bracket_id']
        self.f_brace = bracket_data['bracket_flexibility']
        self.kp = bracket_data['kp']
        self.kz = bracket_data['kz']
        self.capacity_to_equipment = bracket_data['capacity_to_equipment']
        self.bracket_capacity = bracket_data['bracket_capacity']
        self.capacity_to_backing = bracket_data['capacity_to_backing']
        self.shear_capacity = bracket_data['shear_capacity']
        self.capacity_method = bracket_data['capacity_method']
        self.asd_lrfd_ratio = asd_lrfd_ratio


    def get_element_stiffness_matrix(self, u=None):
        """ Returns a 6x6 stiffness matrix for the primary degrees of freedom without modifying self. """

        # Compute initial stiffness values
        kn = 1 / (self.f_wall + self.f_brace)
        kp = self.kp
        kz = self.kz

        # If releases are defined and u is provided, check for displacement conditions
        if any(self.releases) and (u is not None):
            delta = self.G @ self.C @ u[0:6]  # Efficient matrix multiplication

            kn *= not ((delta[0] > 0 and self.releases[0]) or (delta[0] < 0 and self.releases[1]))
            kp *= not ((delta[1] > 0 and self.releases[2]) or (delta[1] < 0 and self.releases[3]))
            kz *= not ((delta[2] > 0 and self.releases[4]) or (delta[2] < 0 and self.releases[5]))

        # Construct local stiffness matrix
        k_br = np.diag([kn, kp, kz])

        # Compute global stiffness matrix without modifying self
        K = self.C.T @ self.G.T @ k_br @ self.G @ self.C
        return K, k_br

    def get_element_forces(self, u):
        """Given the"""

        # Compute the axial and shear element basic forces
        u = u[0:6]
        _, k_br = self.get_element_stiffness_matrix(u=u)
        npz_forces = np.linalg.multi_dot((k_br, self.G, self.C, u))
        fn, fp, fz = npz_forces
        self.bracket_forces = {'fn': fn,  # Normal to wall
                               'fp': fp,  # Parallel to Wall (Horizontal)
                               'fz': fz}  # Vertical

        # Compute the end reaction forces in Global Coordinates
        backing_forces = np.dot(self.G.T, npz_forces)
        equipment_forces = -1*backing_forces
        r_backing_to_equipment = self.xyz_equipment - self.xyz_wall
        moment_reactions = np.cross(r_backing_to_equipment,backing_forces)

        self.reactions_equipment = np.concatenate((equipment_forces, moment_reactions))
        self.reactions_backing = np.concatenate((backing_forces, moment_reactions))

        return

    def check_brackets(self):        
        capacities = [self.capacity_to_equipment, self.bracket_capacity, self.capacity_to_backing]
        governing_capacity = min([item for item in capacities if isinstance(item, (int, float))], default=None)
        if self.capacity_method == 'ASD':
            demand = self.bracket_forces['fn']*self.asd_lrfd_ratio
        else:
            demand = self.bracket_forces['fn']
        self.tension_dcr = demand / governing_capacity if governing_capacity else 'OK by inspection'


class RigidHardwareElement:
    """ Represents a rigid, rectangular hardware element with fasteners"""

    def __init__(self, w, h, pz_anchors, xyz_centroid=(0, 0, 0),
                 local_x=(0,-1,0), local_y=(0, 0, 1), local_z=(1, 0, 0)):
        # Boundary
        self.w = w
        self.h = h

        # Geometry in Global Coordinates
        self.centroid = xyz_centroid  # Centroid in global coordinates
        self.local_x = local_x
        self.local_y = local_y
        self.local_z = local_z
        self.global_to_local_transformation = np.array((local_x,local_y,local_z))

        # Geometry in Local Coordinates
        self.pz_anchors = pz_anchors  # Anchor coordinates in parallel-vertical axes.
        self.n_anchors = pz_anchors.shape[0]

        # Bolt Group Properites
        self.Ixx = None
        self.Iyy = None
        self.Ixy = None
        self.Ip = None

        # Nodal Forces
        self.N = None
        self.Vx = None
        self.Vy = None
        self.Mx = None
        self.My = None
        self.T = None

        self.get_anchor_group_properties()

    def get_anchor_group_properties(self):

        x_bar, y_bar = np.mean(self.pz_anchors, axis=0)
        edge_dist_y = min(self.h/2-self.pz_anchors[:,1].max(),
                          abs(-self.h/2 - self.pz_anchors[:,1].min()))
        edge_dist_x = min(self.w/2-self.pz_anchors[:,0].max(),
                          abs(-self.w/2 - self.pz_anchors[:,0].min()))
        # dx = self.pz_anchors[:, 0] - x_bar
        # dy = self.pz_anchors[:, 1] - y_bar
        dx = self.pz_anchors[:, 0] - self.w/2
        dy = self.pz_anchors[:, 1] - self.h/2
        self.Ixx = sum(dy ** 2)
        self.Iyy = sum(dx ** 2)
        # self.Ixy = sum(dx * dy)
        self.Ixy = 0
        self.Ip = self.Ixx + self.Iyy

    def set_centroid_forces(self, Vx, Vy, N, Mx, My, T, convert_to_local=False):
        if convert_to_local:
            converted_forces = self.global_to_local_transformation @ np.array([Vx, Vy, N])
            self.Vx, self.Vy,self.N = converted_forces

            converted_moments = self.global_to_local_transformation @ np.array([Mx, My, T])
            self.Mx, self.My, self.T = converted_moments
        else:
            self.N, self.Vx, self.Vy, self.Mx, self.My, self.T = N, Vx, Vy, Mx, My, T

    def get_anchor_forces(self):
        N, Vx, Vy, Mx, My, T = self.N, self.Vx, self.Vy, self.Mx, self.My, self.T
        Ixx = self.Ixx
        Iyy = self.Iyy
        Ixy = self.Ixy
        x = self.w/2 - self.pz_anchors[:, 0]  # This sums moments about edge of bounding rectangle, I think.
        y = self.h/2 - self.pz_anchors[:, 1]

        normal_term = N / self.n_anchors

        if Ixx == 0 and Ixy == 0:
            mx_term = np.zeros(y.shape)
        elif Ixy == 0:
            mx_term = y * (Mx / Ixx)
        else:
            mx_term = y * (Mx * Iyy - My * Ixy) / (Ixx * Iyy - Ixy ** 2)

        if Iyy == 0 and Ixy == 0:
            my_term = np.zeros(x.shape)
        elif Ixy == 0:
            my_term = -x * (My / Iyy)
        else:
            my_term = -x * (My * Ixx - Mx * Ixy) / (Ixx * Iyy - Ixy ** 2)

        n = normal_term + mx_term + my_term

        Ip = Ixx + Iyy
        if Ip == 0:
            vp = np.ones(self.n_anchors) * Vx / self.n_anchors
            vz = np.ones(self.n_anchors) * Vy / self.n_anchors
        else:
            vp = np.ones(self.n_anchors) * Vx / self.n_anchors - T*y/Ip
            vz = np.ones(self.n_anchors) * Vy / self.n_anchors + T*x/Ip

        return np.column_stack((n, vp, vz))


class WallBackingElement(RigidHardwareElement):
    def __init__(self, w, h, d, pz_anchors, xy_brackets, bracket_indices, supporting_wall,
                 backing_type='Flat', centroid=(0, 0, 0),
                 local_x=None, local_y=None, local_z=None, fy=None, t_steel=None):
        if local_z is None:
            local_z = EquipmentModel.WALL_NORMAL_VECTORS[supporting_wall]
        super().__init__(w, h, pz_anchors, xyz_centroid=centroid, local_x=local_x,
                         local_y=local_y,local_z=local_z)

        # Backing Hardware Properties
        self.d = d
        self.fy = fy
        self.t_steel = t_steel
        self.backing_type = backing_type
        self.supporting_wall = supporting_wall

        # Bracket Properties
        self.bracket_indices = bracket_indices
        self.xy_brackets = xy_brackets  # Bracket attachment locations in local coordinates
        self.x_bar, self.y_bar = np.mean(self.xy_brackets, axis=0)  # Centroid of brackets in local coordinates


        # Results Properties
        self.anchor_forces = None
        self.bracket_forces = None
        self.anchors_obj = None


    def get_centroid_forces(self, bracket_list):

        Vx = 0
        Vy = 0
        N = 0
        Mx = 0
        My = 0
        T = 0
        bracket_forces = np.zeros((len(bracket_list), 3))
        for i, bracket in enumerate(bracket_list):
            bracket_forces[i, :] = (bracket.bracket_forces['fn'],
                                    bracket.bracket_forces['fp'],
                                    bracket.bracket_forces['fz'])

            Vx += bracket.reactions_backing[0]
            Vy += bracket.reactions_backing[1]
            N += bracket.reactions_backing[2]
            r = bracket.xyz_backing - self.centroid
            dMx, dMy, dMz = np.cross(r,bracket.reactions_backing[0:3])

            Mx += bracket.reactions_backing[3] + dMx
            My += bracket.reactions_backing[4] + dMy
            T += bracket.reactions_backing[5] + dMz
        self.bracket_forces = bracket_forces
        self.set_centroid_forces(Vx, Vy, N, Mx, My, T, convert_to_local=True)

    def get_anchor_forces(self, bracket_list):
        self.get_centroid_forces(bracket_list)
        self.anchor_forces = super().get_anchor_forces()


class SMSHardwareAttachment(RigidHardwareElement):
    def __init__(self, w, h, pz_anchors, df_sms, centroid=(0, 0, 0),
                 local_x=None,local_y=None, local_z=None):
        super().__init__(w, h, pz_anchors, xyz_centroid=centroid,
                         local_x=local_x,local_y=local_y, local_z=local_z)
        self.anchors_obj = SMSAnchors(df_sms=df_sms)

    def get_anchor_forces(self, Vx, Vy, N, Mx, My, T, convert_to_local=False):
        self.set_centroid_forces(Vx, Vy, N, Mx, My, T, convert_to_local=convert_to_local)
        self.anchors_obj.anchor_forces = super().get_anchor_forces()


class SMSAnchors:
    def __init__(self, wall_data=None, backing_type=None, df_sms=None, condition_x = 'Condition 1',
                 condition_y = 'Condition 1'):
        # SMS capacity table
        self.df_sms = df_sms

        # Sheet Metal Properties
        self.fy = None
        self.gauge = None

        # Screw Properties
        self.screw_size = None

        # Attachment (Capacity) Condition
        self.condition_x = condition_x  # Steel to steel, One layer gyp, Two layer gyp, Prying
        self.condition_y = condition_y

        # Anchor Demands
        self.anchor_forces = None
        self.Tu_max = None
        self.DCR = None
        self.results = {}

        if wall_data is not None:
            self.set_sms_properties(gauge=wall_data['stud_gauge'], fy=wall_data['stud_fy'],
                                    num_gyp=wall_data['num_gyp'], backing_type=backing_type)

        self.conditions = {'Condition 1': {'Label': 'Steel-to-Steel Connection',
                                           'Table': 'Table 1'},
                           'Condition 2': {'Label': 'Single-Layer Gyp. Board (Non-Prying)',
                                           'Table': 'Table 2'},
                           'Condition 3': {'Label': 'Two-layer Gyp. Board (Non-Prying)',
                                           'Table': 'Table 3'},
                           'Condition 4': {'Label': 'Prying Condition',
                                           'Table': 'Table 4'}}

    def set_sms_properties(self, gauge=18, fy=33, num_gyp=0, backing_type=None):
        self.gauge = gauge
        self.fy = fy

    def set_screw_size(self, screw_size):
        self.screw_size = screw_size

    def check_anchors(self, asd_lrfd_ratio):
        # Convert Anchor Forces to ASD
        anchor_forces = self.anchor_forces * asd_lrfd_ratio

        # Determine if anchor_forces is (n x t x 3) or (n x 3)
        if anchor_forces.ndim == 3:
            # Get Maximum Forces for (n x t x 3) case
            tension_forces = anchor_forces[:, :, 0]
            idx_anchor, idx_theta = get_max_demand(tension_forces)
            tension_demand = tension_forces[idx_anchor, idx_theta]
            shear_demand = ((anchor_forces[idx_anchor, idx_theta, 1] ** 2 + anchor_forces[idx_anchor, idx_theta, 2] ** 2) ** 0.5)
            shear_x_demand = np.abs(anchor_forces[idx_anchor,idx_theta,1]).max()
            shear_y_demand = np.abs(anchor_forces[idx_anchor,idx_theta,2]).max()
        elif anchor_forces.ndim == 2:
            # Get Maximum Forces for (n x 3) case
            tension_demand = max(anchor_forces[:, 0].max(),0)
            shear_demand = ((anchor_forces[:, 1] ** 2 + anchor_forces[:, 2] ** 2) ** 0.5).max()
            shear_x_demand = np.abs(anchor_forces[:, 1]).max()
            shear_y_demand = np.abs(anchor_forces[:, 2]).max()
        else:
            raise ValueError("anchor_forces must be either an (n x t x 3) or (n x 3) array.")

        # Store the results
        self.results['Tension Demand'] = tension_demand
        self.results['Shear Demand'] = shear_demand
        self.results['Shear X Demand'] = shear_x_demand
        self.results['Shear Y Demand'] = shear_y_demand
        self.Tu_max = tension_demand
        self.Vu_max = shear_demand

        filtered_dfx = self.df_sms[
            (self.df_sms['sms_size'] == self.screw_size) &
            (self.df_sms['condition'] == self.condition_x) &
            (self.df_sms['fy'] == self.fy) &
            (self.df_sms['gauge'] == self.gauge)
            ]

        filtered_dfy = self.df_sms[
            (self.df_sms['sms_size'] == self.screw_size) &
            (self.df_sms['condition'] == self.condition_y) &
            (self.df_sms['fy'] == self.fy) &
            (self.df_sms['gauge'] == self.gauge)]

        if not filtered_dfx.empty:
            if np.isnan(filtered_dfx['shear'].values[0]) or np.isnan(filtered_dfx['tension'].values[0]) or np.isnan(filtered_dfy['shear'].values[0]):
                self.results['Shear X Capacity'] = "NA"
                self.results['Shear Y Capacity'] = "NA"
                self.results['Tension Capacity'] = "NA"
                self.results['Shear X DCR'] = "NG"
                self.results['Shear Y DCR'] = "NG"
                self.results['Shear DCR'] = "NG"
                self.results['Tension DCR'] = "NG"
                self.results['OK'] = False
                self.DCR = np.inf
            else:
                self.results['Shear X Capacity'] = filtered_dfx['shear'].values[0]
                self.results['Shear Y Capacity'] = filtered_dfy['shear'].values[0]
                self.results['Tension Capacity'] = filtered_dfx['tension'].values[0]
                self.results['Shear X DCR'] = self.results['Shear X Demand'] / self.results['Shear X Capacity']
                self.results['Shear Y DCR'] = self.results['Shear Y Demand'] / self.results['Shear Y Capacity']
                self.results['Shear DCR'] = (self.results['Shear X DCR']**2+self.results['Shear Y DCR']**2)**0.5
                self.results['Tension DCR'] = self.results['Tension Demand'] / self.results['Tension Capacity']
                self.DCR = self.results['Tension DCR'] + self.results['Shear DCR']
                self.results['OK'] = self.DCR < 1

        else:
            self.results['Shear X Capacity'] = 'NA'
            self.results['Shear Y Capacity'] = 'NA'
            self.results['Tension Capacity'] = 'NA'
            self.results['Shear X DCR'] = 'NG'
            self.results['Shear Y DCR'] = 'NG'
            self.results['Shear DCR'] = "NG"
            self.results['Tension DCR'] = 'NG'
            self.results['OK'] = False
            self.DCR = np.inf

    def reset_results(self):
        self.results = {}

    def max_dcr(self):
        """For use in comparing dcrs of multiple SMSAnchors objects.
        Returns the max of shear or tension DCR, or inf if either is NG."""
        if self.results == {}:
            return np.inf
        shear_x = np.inf if self.results['Shear X DCR'] == 'NG' else self.results['Shear X DCR']
        shear_y = np.inf if self.results['Shear Y DCR'] == 'NG' else self.results['Shear Y DCR']
        tension = np.inf if self.results['Tension DCR'] == 'NG' else self.results['Tension DCR']
        return max(shear_x, shear_y, tension)







        # self.DCR =

class BraceElement:
    def __init__(self, xyz_i, xyz_j):
        """ Assumes the brace element attaches the equipment object to a hardware object
        (for now, floor plate element)"""

        self.xyz_i = xyz_i
        self.xyz_j = xyz_j
        self.ks = None
        self.tension_only = False
        self.bracket_id = None
        self.capacity_to_equipment = None
        self.brace_capacity = None
        self.capacity_to_backing = None
        self.M = None
        self.k_element = None
        self.brace_force = None
        self.tension_dcr = None
        self.capacity_method = None

    def set_brace_properties(self, bracket_data):
        self.ks = bracket_data['bracket_stiffness']
        self.tension_only = bool(bracket_data['tension_only'])
        self.bracket_id = bracket_data['bracket_id']
        self.capacity_to_equipment = bracket_data['capacity_to_equipment']
        self.brace_capacity = bracket_data['bracket_capacity']
        self.capacity_to_backing = bracket_data['capacity_to_backing']
        self.capacity_method = bracket_data['capacity_method']

    def get_element_deformation(self, u):
        return self.M @ u

    def get_element_stiffness_matrix(self, u, initial=False):
        if self.tension_only and not initial and (self.get_element_deformation(u) < 0):
            return np.zeros_like(self.k_element)
        else:
            return self.k_element

    def get_brace_force(self, u):
        force = self.ks * self.M @ u
        if self.tension_only:
            self.brace_force = max([force, 0])
        else:
            self.brace_force = force

    def check_brace(self, u, asd_lrfd_ratio=1):
        self.get_brace_force(u)
        if self.capacity_method == 'ASD':
            brace_force = self.brace_force * asd_lrfd_ratio
        else:
            brace_force = self.brace_force

        capacities = [self.capacity_to_equipment, self.brace_capacity, self.capacity_to_backing]
        governing_capacity = min([item for item in capacities if isinstance(item, (int, float))], default=None)
        self.tension_dcr = brace_force / governing_capacity if governing_capacity else 'OK by inspection'


class BaseStrap(BraceElement):
    def __init__(self, xyz_i, xyz_j, base_plate):
        super().__init__(xyz_i, xyz_j)
        self.base_plate = base_plate

    def pre_compute_matrices(self):
        (dx, dy, dz) = (j - i for j, i in zip(self.xyz_j, self.xyz_i))
        L = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

        A = np.array([-dx / L, -dy / L, -dz / L, dx / L, dy / L, dz / L])

        (xi, yi, zi) = self.xyz_i
        (xj, yj, zj) = self.xyz_j
        x0 = self.base_plate.x0
        y0 = self.base_plate.y0
        z0 = self.base_plate.z0

        ci = np.array([[1, 0, 0, 0, zi, -yi],
                       [0, 1, 0, -zi, 0, xi],
                       [0, 0, 1, yi, -xi, 0]])
        cj = np.array([[1, 0, 0, 0, zj - z0, -(yj - y0)],
                       [0, 1, 0, -(zj - z0), 0, (xj - x0)],
                       [0, 0, 1, (yj - y0), -(xj - x0), 0]])

        local_constraints = np.block([
            [ci, np.zeros((3, 6))],
            [np.zeros((3, 6)), cj]])

        c_element = self.base_plate.C
        c_equip = np.eye(*c_element.shape)

        global_constraints = np.vstack((c_equip, c_element))

        self.M = A @ local_constraints @ global_constraints
        self.k_element = self.ks * np.outer(self.M.T, self.M)
        return
