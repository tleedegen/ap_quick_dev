from jedi.plugins.django import mapping

from anchor_pro.elements.wall_bracket import BracketReleases
from anchor_pro.model import WallOffsets
from anchor_pro.project_controller.excel_importer import ExcelTablesImporter
from anchor_pro.ap_types import (
    SeriesOrDict, WallPositions, WallNormalVecs
)
import anchor_pro.model as m
import anchor_pro.elements.base_plates as bp
import anchor_pro.elements.fastener_connection as cxn
import anchor_pro.elements.sms as sms
import anchor_pro.elements.concrete_anchors as conc
import anchor_pro.elements.wall_bracket as wbkt
import anchor_pro.elements.wall_backing as wbing

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from typing import Tuple, List, Optional, Literal

from anchor_pro.utilities import (
    transform_points,
    transform_vectors,
    compute_backing_xy_points
)

import json
import warnings

'''Factories'''
def model_factory(equipment_data: dict, excel_tables:ExcelTablesImporter) ->Tuple[m.EquipmentModel, bool]:

    # 1) Extract Inputs
    project_info = excel_tables.project_info
    df_fasteners = excel_tables.df_fasteners
    df_sms = excel_tables.df_sms

    # 2) Create Properties Objects
    # Create Model Info
    equipment_info = m.EquipmentInfo(
        equipment_id=equipment_data['equipment_id'],
        equipment_type=equipment_data['equipment_type']
    )

    # Create Equipment Properties
    ex = equipment_data['ex']
    ey = equipment_data['ey']
    if ex is None or np.isnan(ex):
        ex = 0
    if ey is None or np.isnan(ey):
        ey = 0

    eprops = m.EquipmentProps(
        Wp=equipment_data['Wp'],
        Bx=equipment_data['Bx'],
        By=equipment_data['By'],
        H=equipment_data['H'],
        zCG=equipment_data['zCG'],
        ex=ex,
        ey=ey,
        gauge=equipment_data.get('gauge'),
        fy=equipment_data.get('fy')
    )

    # Installation Info
    install = m.InstallationInfo(
        installation_type=equipment_data['installation_type'],
        base_material=equipment_data['base_material'],
        wall_material=equipment_data['wall_type']
    )

    code_pars = code_pars_factory(
        code_edition=project_info['code_edition'],
        equipment_data=equipment_data,
        project_info=project_info
    )

    # 3) Create Elements Lists with Factory Functions
    # Base Plate Elements, Connections, Anchors
    base_geometry_json = equipment_data['Pattern Definition_base']
    if not isinstance(base_geometry_json, str):
        raise KeyError(f"{equipment_info.equipment_id} base geometry {equipment_data['base_geometry']} not found.")

    E_base = None
    poisson = None

    if install.base_material == m.BaseMaterial.concrete:
        E_base = equipment_data['Ec_base']
        poisson = equipment_data['poisson_base']
    elif install.base_material == m.BaseMaterial.wood:
        E_base = equipment_data['E_wood_base']
        poisson = E_base / (2 * equipment_data[
            'G_wood']) - 1  # todo: this is wrong. G wood is specific gravity, not shear modulus
    else:
        raise Exception(f'Specified base material {install.base_material} not supported.')

    (base_plates,
     bp_connections,
     sms_fasteners,
     bp_to_cxn,
     cxn_to_sms) = base_elements_factory(base_geometry_json, eprops, E_base, poisson, df_fasteners, df_sms)

    base_anchors = base_anchors_factory(
        install, base_plates, eprops, equipment_data
    )

    # Wall Brackets, Connections, CXN SMS, Backing, Wall Anchors
    wall_geometry_json = equipment_data['Pattern Definition_wall']
    wall_height = equipment_data['wall_height']
    E_wall = equipment_data['E_wall']
    I_wall = equipment_data['I_wall']
    if not isinstance(wall_geometry_json, str):
        raise KeyError(f'{equipment_info.equipment_id} wall geometry {equipment_data["wall_geometry"]} not found.')
    (wall_brackets,
        wall_backing,
        bracket_connections,
        bracket_sms,
        bracket_to_backing, backing_to_brackets,
        wall_offsets,
        omit_bracket_output) = wall_elements_factory(wall_geometry_json, eprops, wall_height, E_wall,I_wall,df_fasteners)

    wall_anchors, backing_to_anchors, anchors_to_backing = wall_anchors_factory(
        install, wall_backing, eprops, equipment_data
    )

    # Create Elements Dataframe Object
    elements = m.Elements(
        base_plates=base_plates,
        base_anchors=base_anchors,
        base_plate_connections=bp_connections,
        base_plate_fasteners=sms_fasteners,
        bp_to_cxn=bp_to_cxn,
        cxn_to_sms=cxn_to_sms,
        bp_to_anchors=[0] * len(base_plates),
        # todo: This assigns all base plates to single anchors object. Modify for Wood Fasteners
        wall_brackets=wall_brackets,
        wall_backing=wall_backing,
        wall_anchors=wall_anchors,
        wall_bracket_connections=bracket_connections,
        wall_bracket_fasteners=bracket_sms,
        bracket_to_backing=bracket_to_backing,
        backing_to_brackets=backing_to_brackets,
        backing_to_anchors=backing_to_anchors,
        anchors_to_backing=anchors_to_backing
    )

    return m.EquipmentModel(
        equipment_info=equipment_info,
        install=install,
        code_pars=code_pars,
        equipment_props=eprops,
        elements=elements,
        wall_offsets=wall_offsets
    ), omit_bracket_output

def base_elements_factory(
    base_geometry_json_string: str,
    eprops: m.EquipmentProps,
    E_base: float,
    poisson: float,
    df_fasteners: pd.DataFrame,
    df_sms: pd.DataFrame,   # currently unused; keep for symmetry or remove
    ) -> Tuple:
    """
    Creates base-plate elements from the user-input JSON pattern.

    Returns a bundle with flat element lists and index maps:
        bp_to_cxn[i]     -> index of connection for base_plate[i], or None
        cxn_to_sms[j]    -> index of SMS fastener set for connection[j], or None
    """

    pattern = json.loads(base_geometry_json_string)

    # NOTE: base_strap not yet used
    selected_straps = {
        item['element']['straps']['strap']
        for item in pattern
        if item['element']['straps']['strap'] != 'null'
    }
    if len(selected_straps) == 0:
        base_strap = None
    elif len(selected_straps) > 1:
        warnings.warn(
            "User specified different strap types within one geometry pattern; "
            "proceeding with an arbitrary one."
        )
        base_strap = selected_straps.pop()
    else:
        base_strap = selected_straps.pop()

    base_plates: List[bp.BasePlateElement] = []
    connections: List[cxn.FastenerConnection] = []
    sms_fasteners: List[sms.SMSAnchors] = []
    bp_to_cxn: List[Optional[int]] = []
    cxn_to_sms: List[Optional[int]] = []

    for item in pattern:
        element = item['element']
        shape   = element['shape']
        anchors = element['anchors']
        straps  = element['straps']
        layout  = item['layout']

        # Parse/expand parametric shape & anchor points in local plate coords
        shape_points = np.asarray([
            [float(x) + float(xr) * eprops.Bx, float(y) + float(yr) * eprops.By]
            for x, y, xr, yr in zip(shape['X'], shape['Y'], shape['Xr'], shape['Yr'])
        ], dtype=float)

        anchor_points = np.asarray([
            [float(x) + float(xr) * eprops.Bx, float(y) + float(yr) * eprops.By]
            for x, y, xr, yr in zip(anchors['X'], anchors['Y'], anchors['Xr'], anchors['Yr'])
        ], dtype=float)

        # todo [BASE STRAPS]:
        # strap_geometry = [[float(x), float(y), float(z), float(dx), float(dy), float(dz)] for x, y, z, dx, dy, dz in
        #                   zip(straps['X'], straps['Y'], straps['Z'], straps['DX'], straps['DY'], straps['DZ']) if
        #                   self.base_strap]

        # Any releases present?
        release_keys = ['release_mx','release_my','release_mz',
                        'release_xp','release_xn','release_yp','release_yn','release_zp','release_zn']
        releases_present = any(value for key in release_keys for value in layout.get(key, []))

        split_mode = releases_present or element.get('check_fasteners', False)

        bearing_boundaries = []
        aggregated_xy = []

        zips = zip(layout["X"], layout["Y"], layout["Xr"], layout["Yr"],
                   layout["Rotation"], layout["Reflection"],
                   layout.get("release_mx", []), layout.get("release_my", []), layout.get("release_mz", []),
                   layout.get("release_xp", []), layout.get("release_xn", []),
                   layout.get("release_yp", []), layout.get("release_yn", []),
                   layout.get("release_zp", []), layout.get("release_zn", []))

        # Iterate and Create Base Plate Elements
        for (x, y, xr, yr, rot, refl,
             r_mx, r_my, r_mz, r_xp, r_xn, r_yp, r_yn, r_zp, r_zn) in zips:

            translation = np.array([float(x) + float(xr) * eprops.Bx,
                                    float(y) + float(yr) * eprops.By], dtype=float)

            boundary = transform_points(shape_points, translation, rot, refl)     # (n_b, 2)
            anchors_t = transform_points(anchor_points, translation, rot, refl)  # (n_a, 2)

            if split_mode:
                # Create Individual Base Plate Elements
                (x0, y0) = transform_points(np.array([[element['x0'], element['y0']]], float),
                                            translation, rot, refl)[0]
                z0 = float(element['z0'])
                (xc, yc) = transform_points(np.array([[element['xc'], element['yc']]], float),
                                            translation, rot, refl)[0]
                zc = float(element['zc'])

                releases = bp.BasePlateReleases(mx=bool(r_mx), my=bool(r_my), mz=bool(r_mz),
                                                xp=bool(r_xp), xn=bool(r_xn),
                                                yp=bool(r_yp), yn=bool(r_yn),
                                                zp=bool(r_zp), zn=bool(r_zn))

                props = bp.BasePlateProps(
                    bearing_boundaries=[boundary],
                    E_base=float(E_base),
                    poisson=float(poisson),
                    xy_anchors=anchors_t if anchors_t.size else np.empty((0, 2), dtype=float),
                    x0=float(x0), y0=float(y0), z0=float(z0),
                    xc=float(xc), yc=float(yc), zc=float(zc),
                    releases=releases,
                )
                base_plates.append(bp.BasePlateElement(props=props))

                # Create Connections
                if element.get('check_fasteners', False):
                    # Lookup fastener pattern
                    fastener_pattern = element['fastener_geometry_name']
                    if fastener_pattern not in df_fasteners.index:
                        raise KeyError(f"Fastener geometry '{fastener_pattern}' not found.")
                    data = df_fasteners.loc[fastener_pattern]
                    orientation = element['fastener_orientation']
                    plate_centroid_XYZ = np.array([xc, yc, zc], dtype=float)
                    local_x, local_y, local_z = base_connection_local_axes(orientation,rot,refl)
                    connection, sms_fastener = _sms_connection_factory(
                        fastener_pattern_data=data,
                        local_x=local_x,
                        local_y=local_y,
                        local_z=local_z,
                        eprops=eprops,
                        plate_centroid_XYZ=plate_centroid_XYZ)

                    connections.append(connection)
                    sms_fasteners.append(sms_fastener)
                    bp_to_cxn.append(len(connections) - 1)
                    cxn_to_sms.append(len(sms_fasteners) - 1)

                # todo [BASE STRAPS]
                # for strap_pts in strap_geometry:
                #     (x_eq, y_eq,) = Utilities.transform_points([[strap_pts[0] + strap_pts[3],
                #                                                  strap_pts[1] + strap_pts[4]]],
                #                                                translation, rotation_angle, reflection_angle)[0]
                #     z_eq = strap_pts[2] + strap_pts[5]
                #
                #     (x_pl, y_pl,) = Utilities.transform_points([[strap_pts[0],
                #                                                  strap_pts[1]]],
                #                                                translation, rotation_angle, reflection_angle)[0]
                #     z_pl = strap_pts[2]
                #
                #     self.base_straps.append(BaseStrap((x_eq, y_eq, z_eq), (x_pl, y_pl, z_pl), plate))

            else:  # If not split_mode, aggregate boundaries and anchor_xy points
                bearing_boundaries.append(boundary)
                if anchors_t.size:
                    aggregated_xy.extend(anchors_t.tolist())

        # Finally, Create single multi-boundary element if applicable
        if not split_mode:
            xy_anchors_arr = (
                np.empty((0, 2), dtype=float)
                if len(aggregated_xy) == 0
                else np.asarray(aggregated_xy, dtype=float)
            )

            props = bp.BasePlateProps(
                bearing_boundaries=bearing_boundaries,
                E_base=float(E_base),
                poisson=float(poisson),
                xy_anchors=xy_anchors_arr,
                x0=0.0, y0=0.0, z0=float(element['z0']),
                xc=0.0, yc=0.0, zc=float(element['z0']),
                releases=bp.BasePlateReleases(),  # no releases
            )
            base_plates.append(bp.BasePlateElement(props=props))
            bp_to_cxn.append(None)

    return (base_plates,
            connections,
            sms_fasteners,
            bp_to_cxn,
            cxn_to_sms)

def _sms_connection_factory(
        fastener_pattern_data,
        local_x,
        local_y,
        local_z,
        eprops,
        plate_centroid_XYZ
    ):
    ORIENT = {'X+': 0.0, 'X-': np.pi, 'Y+': np.pi / 2, 'Y-': 3 * np.pi / 2}

    data=fastener_pattern_data

    # Parametric width/height
    if np.isclose(local_z[0], 0.0):
        B = float(eprops.By)
    elif np.isclose(local_z[1], 0.0):
        B = float(eprops.Bx)
    else:
        B = 0.0

    w = float(data['W']) + float(data['Wr']) * B
    h = float(data['H']) + float(data['Hr']) * float(eprops.H)
    if np.isclose(w, 0.0):
        raise ValueError(
            "For base-plate attachments with fasteners, 'W' + 'Wr'*B must be nonzero "
            "or plates must be orthogonal to the ref box."
        )

    L_horiz = w - 2.0 * float(data['X Edge'])
    L_vert = h - 2.0 * float(data['Y Edge'])
    y_off = float(data['Y Offset'])
    x_off = float(data['X Offset'])
    place_x = data['X Placement']
    place_y = data['Y Placement']

    xy_points = compute_backing_xy_points(
        int(data['X Number']), int(data['Y Number']),
        L_horiz, L_vert, x_off, y_off,
        place_by_horiz=place_x, place_by_vert=place_y
    )  # -> (n,2) ndarray

    bg_props = cxn.ElasticBoltGroupProps(
        w=w, h=h,
        xy_anchors=np.asarray(xy_points, dtype=float),
        plate_centroid_XYZ=plate_centroid_XYZ,
        local_x=np.asarray(local_x, dtype=float),
        local_y=np.asarray(local_y, dtype=float),
        local_z=np.asarray(local_z, dtype=float),
    )
    connection = cxn.FastenerConnection(bolt_group_props=bg_props)

    # SMS props (fallbacks)
    gauge = int(eprops.gauge) if eprops.gauge is not None else 18
    fy = float(eprops.fy) if eprops.fy is not None else 33.0
    sms_props = sms.SMSProps(
        fy=fy, gauge=gauge,
        condition_x=sms.SMSCondition.METAL_ON_METAL,
        condition_y=sms.SMSCondition.METAL_ON_METAL,
    )
    sms_fastener = sms.SMSAnchors(props=sms_props)

    return connection, sms_fastener

def base_anchors_factory(
        install: m.InstallationInfo,
        base_plates_list: List[bp.BasePlateElement],
        eprops: m.EquipmentProps,
        equipment_data: dict)->List[m.BASE_ANCHOR_ELEMENTS]:

    base_anchors = []
    if install.base_material == m.BaseMaterial.concrete:
        xy_anchors_list = [plate.props.xy_anchors for plate in base_plates_list if plate.props.xy_anchors.size > 0]
        if xy_anchors_list:  # Ensure list is not empty before concatenation
            xy_anchors = np.concatenate(xy_anchors_list, axis=0)
            cx_pos = equipment_data['cx_pos_base']
            cx_neg = equipment_data['cx_neg_base']
            cy_neg = equipment_data['cy_pos_base']
            cy_pos = equipment_data['cy_neg_base']
            cx_pos = np.inf if np.isnan(cx_pos) or (cx_pos is None) else cx_pos
            cx_neg = np.inf if np.isnan(cx_neg) or (cx_neg is None) else cx_neg
            cy_pos = np.inf if np.isnan(cy_pos) or (cy_pos is None) else cy_pos
            cy_neg = np.inf if np.isnan(cy_neg) or (cy_neg is None) else cy_neg
            geo_props = conc.GeoProps(
                xy_anchors=xy_anchors,
                Bx=eprops.Bx,
                By=eprops.By,
                cx_pos=cx_pos,
                cx_neg=cx_neg,
                cy_neg=cy_pos,
                cy_pos=cy_neg,
            )
            concrete_props = conc.ConcreteProps(
                weight_classification=equipment_data['weight_classification_base'],
                profile=equipment_data['profile_base'],
                fc=equipment_data['fc_base'],
                lw_factor=equipment_data['lw_factor_base'],
                cracked_concrete=bool(equipment_data['cracked_concrete_base']),
                poisson=equipment_data['poisson_base'],
                t_slab=equipment_data['t_slab_base']
            )
            base_anchors = [conc.ConcreteAnchors(
                geo_props=geo_props,
                concrete_props=concrete_props)]
        else:
            base_anchors = []

    elif install.base_material == m.BaseMaterial.wood:
        raise NotImplementedError('Need to implement separate anchor objects for wood base anchors')
    else:
        raise KeyError(f'Selected base material {install.base_material} is invalid.')
    return base_anchors

def wall_elements_factory(
        wall_geometry_json_string: str,
        eprops: m.EquipmentProps,
        wall_height: float,
        E_wall: float,
        I_wall: float,
        df_fasteners: pd.DataFrame
)->Tuple:
    # Data unpacking and validation
    json_data = json.loads(wall_geometry_json_string)
    omit_bracket_output = json_data['omit_bracket_output']

    bracket_locations = json_data['bracket_locations']
    backing_groups = json_data['backing_groups']
    df_bracket_locations = pd.DataFrame(bracket_locations)
    df_backing_groups = pd.DataFrame.from_dict(backing_groups, orient='columns')
    df_brackets = df_bracket_locations.merge(df_backing_groups, left_on='Backing Group', right_on='group_number',
                                             how='left')
    df_brackets = df_brackets.merge(df_fasteners, left_on='backing_pattern', right_on='Pattern Name', how='left')
    df_brackets = df_brackets.rename(columns={'Supporting Wall': 'supporting_wall',
                                              'Plate X Offset': 'plate_x_offset',
                                              'Plate Y Offset': 'plate_y_offset',
                                              'N+': 'NP', 'N-': 'NN', 'P+': 'PP', 'P-': 'PN', 'Z+': 'ZP', 'Z-': 'ZN',
                                              'Backing Group':'backing_group',
                                              'Attachment Normal': 'attachment_normal'})

    # Wall Offsets
    wall_offsets = WallOffsets(
        XP=json_data['wall_offsets']['X+'],
        XN=json_data['wall_offsets']['X-'],
        YP=json_data['wall_offsets']['Y+'],
        YN=json_data['wall_offsets']['Y-'])

    # Wall Properties
    L = wall_height
    E = E_wall
    I = I_wall

    # Wall Brackets
    (wall_brackets,
     bracket_to_backing,
     backing_to_brackets) = _wall_brackets_factory(
        wall_offsets,
        df_brackets,
        E, I, L, eprops)

    # Wall Backing
    wall_backing = _wall_backing_factory(
        eprops=eprops,
        wall_brackets=wall_brackets,
        backing_to_brackets=backing_to_brackets,
        df_backing=df_brackets
    )

    # Connections
    bracket_connections = []
    bracket_fasteners = []
    if json_data['check_fasteners']:
        # Lookup fastener pattern
        fastener_pattern = json_data['fastener_geometry']
        if fastener_pattern not in df_fasteners.index:
            raise KeyError(f"Fastener geometry '{fastener_pattern}' not found.")
        data = df_fasteners.loc[fastener_pattern]
        for bracket_idx, bracket in enumerate(wall_brackets):
            orientation = json_data['bracket_locations']['Attachment Normal'][bracket_idx]
            local_x, local_y, local_z = bracket_connection_local_axes(orientation,bracket.geo_props.normal_unit_vector)
            plate_centroid_XYZ = bracket.geo_props.xyz_equipment
            connection, sms_fastener = _sms_connection_factory(
                fastener_pattern_data=data,
                local_x=local_x,
                local_y=local_y,
                local_z=local_z,
                eprops=eprops,
                plate_centroid_XYZ=plate_centroid_XYZ)
            bracket_connections.append(connection)
            bracket_fasteners.append(sms_fastener)

    return(
        wall_brackets,
        wall_backing,
        bracket_connections,
        bracket_fasteners,
        bracket_to_backing, backing_to_brackets,
        wall_offsets,
        omit_bracket_output)

def _wall_brackets_factory(
        wall_offsets: WallOffsets,
        df_brackets: pd.DataFrame,
        E, I, L,
        eprops
    ) -> Tuple[List[wbkt.WallBracketElement],List[int], List[List[int]]]:

    wall_brackets: List[wbkt.WallBracketElement] = []
    bracket_to_backing = []
    # Extract bracket_locations to DataFrame


    backing_counter = 0
    backing_group_label_to_index = {}


    for bracket in df_brackets.itertuples(index=False):
        # Attachment Point
        x0 = bracket.X + bracket.Xr * eprops.Bx
        y0 = bracket.Y + bracket.Yr * eprops.By
        z0 = bracket.Z + bracket.Zr * eprops.H
        xyz_equipment = np.array((x0, y0, z0))

        # Wall Normal Vector
        supporting_wall = bracket.supporting_wall
        normal_vec = np.asarray(WallNormalVecs[WallPositions(supporting_wall).name].value)
        backing_depth = bracket.D
        # Calculate Bracket Centerline Point And Connection Offset Point

        plate_x_offset = bracket.plate_x_offset
        z_offset = bracket.plate_y_offset

        if supporting_wall == 'X+':
            wall_gap = 0 if not wall_offsets.XP else wall_offsets.XP
            x_offset = 0
            y_offset = -plate_x_offset
        elif supporting_wall == 'X-':
            wall_gap = 0 if not wall_offsets.XN else wall_offsets.XN
            x_offset = 0
            y_offset = plate_x_offset
        elif supporting_wall == 'Y+':
            wall_gap = 0 if not wall_offsets.YP else wall_offsets.YP
            x_offset = plate_x_offset
            y_offset = 0
        elif supporting_wall == 'Y-':
            wall_gap = 0 if not wall_offsets.YN else wall_offsets.YN
            x_offset = -plate_x_offset
            y_offset = 0
        else:
            raise Exception('Supporting Wall Incorrectly Defined')

        xyz_wall = xyz_equipment - wall_gap * normal_vec
        xyz_backing = xyz_equipment + np.array((x_offset, y_offset, z_offset))

        releases = BracketReleases(
            NP=bracket.NP,
            NN=bracket.NN,
            PP=bracket.PP,
            PN=bracket.PN,
            ZP=bracket.ZP,
            ZN=bracket.ZN
        )

        geo_props = wbkt.GeometryProps(
            xyz_equipment=xyz_equipment,
            xyz_wall=xyz_wall,
            xyz_backing=xyz_backing,
            supporting_wall=WallPositions(supporting_wall),
            normal_unit_vector=normal_vec,
            E=E, I=I, L=L,
            releases=releases)

        wall_brackets.append(wbkt.WallBracketElement(geo_props=geo_props))

        # Create Mapping Lists
        backing_group = bracket.backing_group

        if backing_group == 0:
            backing_index = backing_counter
            backing_counter += 1
        elif backing_group not in backing_group_label_to_index:
            backing_group_label_to_index[backing_group] = backing_counter
            backing_counter += 1
            backing_index = backing_group_label_to_index[backing_group]
        else:
            backing_index = backing_group_label_to_index[backing_group]
        bracket_to_backing.append(backing_index)

    backing_to_brackets = []
    for backing_idx in range(backing_counter):
        backing_to_brackets.append([bracket_idx for bracket_idx, backing in enumerate(bracket_to_backing) if backing==backing_idx])



    return wall_brackets, bracket_to_backing, backing_to_brackets

def _wall_backing_factory(
        eprops: m.EquipmentProps,
        wall_brackets: List[wbkt.WallBracketElement],
        backing_to_brackets: List[List[int]],
        df_backing: pd.DataFrame,
        )->List[wbing.WallBackingElement]:

    b_dimension = {'X+': eprops.By,
                   'X-': eprops.By,
                   'Y+': eprops.Bx,
                   'Y-': eprops.Bx}
    wall_backing = []
    for bracket_indices in backing_to_brackets:
        backing_data = df_backing.iloc[bracket_indices[0]].to_dict()
        brackets = [wall_brackets[idx] for idx in bracket_indices]

        if len(set([bracket.geo_props.supporting_wall for bracket in brackets])) > 1:
            raise Exception("Input Error: All brackets in a shared backing group must be attached to the same wall")
        supporting_wall = brackets[0].geo_props.supporting_wall
        backing_type = backing_data['Connection Type']

        b = b_dimension[supporting_wall]  # Width of unit parallel to supporting wall
        wb = backing_data['W'] + backing_data['Wr'] * b  # Width of bracket
        hb = backing_data['H'] + backing_data['Hr'] * eprops.H  # Height of bracket
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
        manual_points = json.loads(manual_points_json)
        manual_x = manual_points['x']
        manual_y = manual_points['y']

        xy_anchor_points = compute_backing_xy_points(backing_data['X Number'], backing_data['Y Number'],
                                                               L_horizontal, L_vert, x_offset, y_offset,
                                                               place_by_horiz=place_by_horiz,
                                                               place_by_vert=place_by_vert,
                                                               manual_x=manual_x,
                                                               manual_y=manual_y)

        # Convert Bracket locations from global XYZ coordinates to local NPZ Coordinates
        xyz_brackets = np.array([bracket.geo_props.xyz_backing for bracket in brackets])
        npz_brackets = np.array([bracket.geo_props.G @ bracket.geo_props.xyz_backing for bracket in brackets])
        pz_brackets = npz_brackets[:, [1, 2]]
        # pz_cent = np.mean(pz_brackets, axis=0)
        # xy_brackets_local = pz_brackets - pz_cent
        centroid_in_global_coordinates = np.mean(np.array([bracket.geo_props.xyz_backing for bracket in brackets]), axis=0)
        local_z = np.asarray(WallNormalVecs[WallPositions(supporting_wall).name].value)
        local_y = (0, 0, 1)
        local_x = np.cross(local_y, local_z)

        props = wbing.WallBackingProps(
            d=db,
            backing_type = backing_type,
            supporting_wall=supporting_wall,
            pz_brackets=pz_brackets,
            xyz_brackets=xyz_brackets,
            # x_bar = x_bar,
            # y_bar = y_bar,
            fy= fy,
            t_steel=t_steel
        )

        bg_props = cxn.ElasticBoltGroupProps(
            w=wb, h=hb,
            xy_anchors=np.asarray(xy_anchor_points, dtype=float),
            plate_centroid_XYZ=centroid_in_global_coordinates,
            local_x=np.asarray(local_x, dtype=float),
            local_y=np.asarray(local_y, dtype=float),
            local_z=np.asarray(local_z, dtype=float),
        )

        wall_backing.append(
            wbing.WallBackingElement(props=props, bolt_group_props=bg_props))
    return wall_backing

def wall_anchors_factory(
        install: m.InstallationInfo,
        wall_backing_list: List[wbing.WallBackingElement],
        eprops: m.EquipmentProps,
        equipment_data: dict)->Tuple[List[m.WALL_ANCHOR_ELEMENTS], List[int], List[List[int]]]:

    wall_anchors = []
    backing_to_anchors = []


    if install.wall_material == m.WallMaterial.concrete:
        walls_with_anchors = set([backing.props.supporting_wall for backing in wall_backing_list])
        supporting_wall_idx = {}
        # Create a single anchors object for each wall to which anchors are attached
        for wall_idx, wall in enumerate(walls_with_anchors):
            supporting_wall_idx[wall] = wall_idx
            xy_anchors_list = [backing.bolt_group_props.xy_anchors
                               for backing in wall_backing_list
                               if backing.props.supporting_wall == wall]
            xy_anchors = np.concatenate(xy_anchors_list, axis=0)
            geo_props = conc.GeoProps(
                xy_anchors=xy_anchors,
                Bx=eprops.Bx,
                By=eprops.By,
                cx_pos=np.inf, #todo: read user-specified edge distances for walls
                cx_neg=np.inf,
                cy_neg=np.inf,
                cy_pos=np.inf,
                supporting_wall=wall
            )
            concrete_props = conc.ConcreteProps(
                weight_classification=equipment_data['weight_classification_wall'],
                profile=equipment_data['profile_wall'],
                fc=equipment_data['fc_wall'],
                lw_factor=equipment_data['lw_factor_wall'],
                cracked_concrete=bool(equipment_data['cracked_concrete_wall']),
                poisson=equipment_data['poisson_wall'],
                t_slab=equipment_data['t_slab_wall']
            )
            wall_anchors.append(conc.ConcreteAnchors(geo_props=geo_props,concrete_props=concrete_props))

        backing_to_anchors = [supporting_wall_idx[backing.props.supporting_wall] for backing in wall_backing_list]

    #todo: Wall SMS and Wood Screws

    anchors_to_backing = []
    for anchor_idx in range(len(wall_anchors)):
        anchors_to_backing.append(
            [backing_idx for backing_idx, anchor in enumerate(backing_to_anchors) if anchor == anchor_idx])

    return wall_anchors, backing_to_anchors, anchors_to_backing

def code_pars_factory(
    code_edition: m.Codes,
    equipment_data: dict,
    project_info: dict
) -> m.CODE_PARS:
    if code_edition == m.Codes.cbc98:
        return m.CBC_98Pars(
            Ip=project_info['Ip'],
            Z=project_info['Z'],
            Cp=equipment_data['Cp'],                     # CBC 1998, 16B parameter
            cp_amplification=equipment_data['cp_amplification'],
            cp_category=equipment_data['cp_category'],
            below_grade=equipment_data['below_grade'],
            grade_factor=1.0 if not equipment_data['below_grade'] else (2.0/3.0),
            Cp_eff=None
        )

    elif code_edition == m.Codes.asce7_16:
        return m.ASCE7_16Pars(
            ap=equipment_data['ap'],
            Rp=equipment_data['Rp'],
            Ip=equipment_data['Ip'],
            sds=project_info['sds'],
            z=equipment_data['z'],
            h=equipment_data['building_height'],
            omega=equipment_data['omega'],
            use_dynamic=False,   # todo: [Future Feature] add a use dynamic toggle in workbook
            ai=None,
            Ax=None
        )

    elif code_edition == m.Codes.asce7_22_OPM:
        return m.ASCE7_22_OPMPars(
            Cpm=equipment_data['Cpm'],
            Cv=equipment_data['Cv'],
            omega=equipment_data['omega_opm']
        )

    else:
        raise NotImplementedError(f"Selected code edition ({code_edition}) not supported.")

def _get(data: SeriesOrDict, key: str, default: object = None):
    """Access helper that works for dict or pandas.Series (prefers .at for speed)."""
    if hasattr(data, "at"):
        try:
            return data.at[key]
        except KeyError:
            return default
    return data.get(key, default)  # type: ignore[attr-defined]

def mechanical_anchor_props_factory(
    anchor_data: SeriesOrDict,
    anchor_obj: conc.ConcreteAnchors,
    is_seismic: bool=True,
    return_bp_stiffness: bool = False
) -> Tuple[conc.MechanicalAnchorProps,bp.AnchorStiffness,bool] | Tuple[conc.MechanicalAnchorProps, bool]:


    cp = anchor_obj.concrete_props
    geo = anchor_obj.geo_props

    # Check Anchor Applicability
    if cp.profile in (conc.Profiles.slab, conc.Profiles.wall):
        anchor_position_ok = bool(_get(anchor_data, "slab_ok", True))
    elif cp.profile == conc.Profiles.deck and geo.anchor_position == conc.AnchorPosition.top:
        anchor_position_ok = bool(_get(anchor_data, "deck_top_ok", True))
    elif cp.profile == conc.Profiles.deck and geo.anchor_position == conc.AnchorPosition.soffit:
        anchor_position_ok = bool(_get(anchor_data, "deck_soffit_ok", True))
    else:
        anchor_position_ok = True  # sensible default

    # Interpolate Spacing Limits by Thickness
    if cp.profile == conc.Profiles.deck:
        suffix = '_deck'
    else:
        suffix = '_slab'

    hmin1 = float(_get(anchor_data, "hmin1"+suffix))
    hmin2 = float(_get(anchor_data, "hmin2"+suffix))
    h_vals  = [hmin1, hmin2]
    c1_vals = [float(_get(anchor_data, "c11"+suffix)), float(_get(anchor_data, "c21"+suffix))]
    c2_vals = [float(_get(anchor_data, "c12"+suffix)), float(_get(anchor_data, "c22"+suffix))]
    s1_vals = [float(_get(anchor_data, "s11"+suffix)), float(_get(anchor_data, "s21"+suffix))]
    s2_vals = [float(_get(anchor_data, "s12"+suffix)), float(_get(anchor_data, "s22"+suffix))]
    cac_vals= [float(_get(anchor_data, "cac1"+suffix)), float(_get(anchor_data, "cac2"+suffix))]

    # Interpolate c1, s1, c2, s2, cac at t_slab
    c1 = float(np.interp(cp.t_slab, h_vals, c1_vals))
    s1 = float(np.interp(cp.t_slab, h_vals, s1_vals))
    c2 = float(np.interp(cp.t_slab, h_vals, c2_vals))
    s2 = float(np.interp(cp.t_slab, h_vals, s2_vals))
    cac = float(np.interp(cp.t_slab, h_vals, cac_vals))
    hmin = hmin1

    # Used Cracked/Uncracked Properties
    kc_cr = float(_get(anchor_data, "kc_cr"))
    kc_uncr = float(_get(anchor_data, "kc_uncr"))
    Np_cr = _get(anchor_data, "Np_cr")
    Np_uncr = _get(anchor_data, "Np_uncr")
    K_cr = float(_get(anchor_data, "K_cr"))
    K_uncr = float(_get(anchor_data, "K_uncr"))

    if cp.cracked_concrete:
        kc_sel = kc_cr
        Np_sel = Np_cr
        K_sel = K_cr
    else:
        kc_sel = kc_uncr
        Np_sel = Np_uncr
        K_sel = K_uncr

    # Use Seismic/Non-Seismic Properties
    Vsa = _get(anchor_data, "Vsa_default")
    Vsa_eq = _get(anchor_data, "Vsa_eq")
    Np_eq = _get(anchor_data, "Np_eq")

    if is_seismic:
        Vsa = float(Vsa_eq) if Vsa_eq is not None else Vsa
        Np_sel = float(Np_eq) if Np_eq is not None else Np_sel


    # Create Anchor Info Object
    info = conc.AnchorBasicInfo(
        anchor_id = _get(anchor_data,'anchor_id'),
        installation_method = _get(anchor_data,'installation_method'),
        anchor_type = _get(anchor_data, 'anchor_type'),
        manufacturer= _get(anchor_data, 'manufacturer'),
        product = _get(anchor_data, 'product'),
        product_type=_get(anchor_data, 'product_type'),
        esr= _get(anchor_data, 'esr'),
        cost_rank= _get(anchor_data, 'cost_rank')
    )

    fya = float(_get(anchor_data, "fya"))
    fua = float(_get(anchor_data, "fua"))
    Nsa = float(_get(anchor_data, "Nsa"))
    le = float(_get(anchor_data, "le"))
    da = float(_get(anchor_data, "da"))
    esr = str(_get(anchor_data, "esr"))
    hef_default = float(_get(anchor_data, "hef_default"))
    Kv = float(_get(anchor_data, "Kv"))
    # Optional
    abrg_val = _get(anchor_data, "abrg", None)
    abrg = float(abrg_val) if abrg_val is not None else None

    #Create Properties Object
    props = conc.MechanicalAnchorProps(
        info=info,
        fya=fya,
        fua=fua,
        Nsa=Nsa,
        Np=Np_sel,
        kc = kc_sel,
        kc_uncr = kc_uncr,
        kc_cr = kc_cr,
        le = le,
        da = da,
        cac = cac,
        esr = esr,
        hef_default = hef_default,
        Vsa = Vsa,
        K = K_sel,
        K_cr = K_cr,
        K_uncr = K_uncr,
        Kv = Kv,
        hmin = hmin,
        c1 = c1, s1 = s1, c2 = c2, s2 = s2,
        abrg = abrg,
        phi = anchor_phi_factors_factory(anchor_data))

    if not return_bp_stiffness:
        return props, anchor_position_ok

    bp_stiffness = bp.AnchorStiffness(
        shear = props.Kv,
        tension= props.K
    )
    return props, bp_stiffness, anchor_position_ok

def anchor_phi_factors_factory(anchor_data: SeriesOrDict) -> conc.Phi:
    return conc.Phi(
        saN=_get(anchor_data, 'phi_saN'),
        pN=_get(anchor_data, 'phi_pN'),
        cN=_get(anchor_data, 'phi_cN'),
        cV=_get(anchor_data, 'phi_cV'),
        saV=_get(anchor_data, 'phi_saV'),
        cpV=_get(anchor_data, 'phi_cpV'),
        eqV=_get(anchor_data, 'phi_eqV'),
        eqN=_get(anchor_data, 'phi_eqN'),
        seismic=0.7,
        aN=_get(anchor_data, 'phi_aN', None)  # Optional, default to None if missing
    )

def wall_bracket_props_factory(bracket_id: str, bracket_data, wall_flexibility:float)->wbkt.BracketProps:

    kn = 1.0 / (wall_flexibility + bracket_data['bracket_flexibility'])
    return wbkt.BracketProps(
        bracket_id=bracket_id,
        brace_flexibility=bracket_data['bracket_flexibility'],
        kn=kn,
        kp=bracket_data['kp'],
        kz=bracket_data['kz'],
        capacity_to_equipment=bracket_data['capacity_to_equipment'],
        bracket_capacity=bracket_data['bracket_capacity'],
        capacity_to_backing=bracket_data['capacity_to_backing'],
        shear_capacity=bracket_data['shear_capacity'],
        capacity_method=bracket_data['capacity_method']
    )

def base_connection_local_axes(
        orientation: Literal['X+','X-','Y+','Y-'],
        rot: float, refl: float)->Tuple[NDArray,NDArray,NDArray]:

    ORIENT = {'X+': 0.0, 'X-': np.pi, 'Y+': np.pi / 2, 'Y-': 3 * np.pi / 2}
    # Local faying direction (in plate plane)
    f_angle = ORIENT[orientation]
    f_local = (np.cos(f_angle), np.sin(f_angle))
    f_global = transform_vectors([f_local], rot, refl)[0]  # (fx, fy)
    local_z = np.array([f_global[0], f_global[1], 0.0], dtype=float)
    local_y = np.array([0.0, 0.0, 1.0], dtype=float)  # connection is vertical
    local_x = np.cross(local_y, local_z)

    return local_x, local_y, local_z

def bracket_connection_local_axes(
        cxn_normal_direction: Literal['X+','X-','Y+','Y-', 'Z+','Z-'],
        bracket_normal: NDArray):
    """ Current default is for bracket connection local x to align with bracket normal-to-wall direction."""
    local_z_dict = {'X+': np.array((1,0,0)),
                    'X-': np.array((-1,0,0)),
                    'Y+': np.array((0, 1, 0)),
                    'Y-': np.array((0, -1, 0)),
                    'Z+': np.array((0, 0, 1)),
                    'Z-': np.array((0, 0, -1))
                    }

    local_x = -bracket_normal
    local_z = local_z_dict[cxn_normal_direction]
    local_y = np.cross(local_z, local_x)
    return local_x, local_y, local_z