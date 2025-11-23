import matplotlib

from anchor_pro.ap_types import FactorMethod, WallPositions
from anchor_pro.model import EquipmentModel

# Force a non-interactive backend
matplotlib.use("agg")


import matplotlib.pyplot as plt
matplotlib.rcParams["text.usetex"] = False
plt.rcParams.update({  # Use times font for latex annotations
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    "text.usetex": False,
})

import matplotlib.patches as patches
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import vtk

import numpy as np
import math
from adjustText import adjust_text  # https://adjusttext.readthedocs.io/en/latest/Examples.html

import anchor_pro.model as m
import anchor_pro.elements.concrete_anchors as conc
import anchor_pro.elements.base_plates as bp
import anchor_pro.elements.sms as sms
import anchor_pro.elements.fastener_connection as cxn
from anchor_pro.project_controller.project_classes import ModelRecord
from anchor_pro.utilities import get_governing_result
from numpy.typing import NDArray
from typing import List, Union, Optional


from pylatex.utils import make_temp_dir
import posixpath
import uuid
import shutil
import os


TEMP_DIR = make_temp_dir()


# TEMP_DIR = r"C:\Users\djmiller\Desktop\AnchorPro Local\Suir Testing\Graphics"  #todo: EXE COMPILE


def plt_save(*args, filename=None, extension="pdf", **kwargs):
    """Save the plot.

    Returns
    -------
    str
        The basename with which the plot has been saved.
    """
    if not filename:
        filename = "{}.{}".format(str(uuid.uuid4()), extension.strip("."))
    else:
        filename = "{}.{}".format(filename, extension.strip("."))
    filepath = posixpath.join(TEMP_DIR, filename)
    filepath = filepath.replace('\\', '/')
    plt.savefig(filepath, *args, **kwargs)
    plt.close()
    return filepath


def plt_save_png(*args, filename=None, extension="png", **kwargs):
    """Save the plot.

    Returns
    -------
    str
        The basename with which the plot has been saved.
    """
    if not filename:
        filename = "{}.{}".format(str(uuid.uuid4()), extension.strip("."))
    else:
        filename = "{}.{}".format(filename, extension.strip("."))
    filepath = posixpath.join(TEMP_DIR, filename)
    filepath = filepath.replace('\\', '/')
    plt.savefig(filepath, *args, format='png', **kwargs)
    return filepath


def plt_save_pgf(*args, extension="pgf", **kwargs):
    """Save the plot efficiently in a vector format."""
    filename = f"{uuid.uuid4()}.{extension.strip('.')}"
    temp_path = os.path.join(TEMP_DIR, filename)

    plt.savefig(temp_path, *args, format="pgf", bbox_inches="tight")

    # Copy the PGF to the working directory for stability
    stable_path = os.path.join(os.getcwd(), filename)
    shutil.copy(temp_path, stable_path)
    return stable_path


def vtk_save(window_to_image_filter, filename=None):
    """Save VTK render output to a file."""
    if not filename:
        filename = f"{str(uuid.uuid4())}.png"
    else:
        filename = f"{filename}.png"

    filepath = posixpath.join(TEMP_DIR, filename)
    filepath = filepath.replace('\\', '/')

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filepath)
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.Write()

    # Explicitly release VTK objects to free memory
    writer.SetInputConnection(None)  # Disconnect input
    writer = None  # Allow garbage collection
    window_to_image_filter = None  # Allow garbage collection

    return filepath


def equipment_3d_view_vtk(model_record: ModelRecord,_, filename=None):
    model = model_record.model
    eprops = model.equipment_props
    install = model.install

    # Create a renderer and render window
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)

    # Extract Bounding Box
    H = eprops.H
    Bx = eprops.Bx
    By = eprops.By

    # Extract Wall/Edge-of-slab Coordinates
    wall_coordinate = {}
    unit_edge = {'X+': 0.5 * eprops.Bx,
                 'X-': -0.5 * eprops.Bx,
                 'Y+': 0.5 * eprops.By,
                 'Y-': -0.5 * eprops.By}
    if model.elements.wall_brackets:
        wall_coordinate = _get_wall_coordinates(model)
        wall_thickness = 4  # todo: update graphic to pull wall thickness from wall anchors object
        wall_height = 120

    # Create Floor Object
    #todo: fix for wood anchors
    if install.installation_type == m.InstallType.base and isinstance(model.elements.base_anchors[0],conc.ConcreteAnchors):
        xyz_min = [-0.6*Bx, -0.6*By, -model.elements.base_anchors[0].concrete_props.t_slab]
        xyz_max = [0.6*Bx, 0.6*By, 0]
        actor = _create_vtk_box(xyz_min, xyz_max, type='filled', opacity=0.35)
        renderer.AddActor(actor)
    elif install.installation_type == m.InstallType.brace:
        xyz_min = [wall_coordinate['X-'] - wall_thickness, wall_coordinate['Y-'] - wall_thickness,
                   -model.base_anchors.t_slab if model.elements.base_anchors else -2]
        xyz_max = [wall_coordinate['X+'] + wall_thickness, wall_coordinate['Y+'] + wall_thickness, 0]
        actor = _create_vtk_box(xyz_min, xyz_max, type='filled', opacity=0.35)
        renderer.AddActor(actor)

    # Create Wall Objects
    if install.installation_type in ['Wall Brackets', 'Wall Mounted']:
        wall_z_min = 0 if install.installation_type == 'Wall Brackets' else -model.H * 0.1
        wall_z_max = 1.1 * model.equipment_props.H
        wall_xyz_min = {'X+': [wall_coordinate['X+'], wall_coordinate['Y-'] - wall_thickness, wall_z_min],
                        'X-': [wall_coordinate['X-'] - wall_thickness, wall_coordinate['Y-'] - wall_thickness,
                               wall_z_min],
                        'Y+': [wall_coordinate['X-'] - wall_thickness, wall_coordinate['Y+'], wall_z_min],
                        'Y-': [wall_coordinate['X-'] - wall_thickness, wall_coordinate['Y-'] - wall_thickness,
                               wall_z_min]}
        wall_xyz_max = {
            'X+': [wall_coordinate['X+'] + wall_thickness, wall_coordinate['Y+'] + wall_thickness, wall_z_max],
            'X-': [wall_coordinate['X-'], wall_coordinate['Y+'] + wall_thickness, wall_z_max],
            'Y+': [wall_coordinate['X+'] + wall_thickness, wall_coordinate['Y+'] + wall_thickness, wall_z_max],
            'Y-': [wall_coordinate['X+'] + wall_thickness, wall_coordinate['Y-'], wall_z_max]}
        for anchors in model.elements.wall_anchors:
            if anchors:
                actor = _create_vtk_box(wall_xyz_min[anchors.geo_props.supporting_wall], wall_xyz_max[anchors.geo_props.supporting_wall], type='filled', opacity=0.35)
                renderer.AddActor(actor)

    # Create Equipment Bounding Box
    xyz_min = [-0.5 * Bx, -0.5 * By, 0]
    xyz_max = [0.5 * Bx, 0.5 * By, H]
    wireframe_actor = _create_vtk_box(xyz_min, xyz_max, type='wireframe')
    renderer.AddActor(wireframe_actor)

    # Create anchor cylinders
    for ba in model.elements.base_anchors:
        for px, py in ba.geo_props.xy_anchors:
            cylinder = vtk.vtkCylinderSource()
            cylinder.SetRadius(0.25)
            cylinder.SetHeight(ba.anchor_props.hef_default + 1)
            cylinder.SetResolution(50)

            # Apply transformation to orient the cylinder vertically
            transform = vtk.vtkTransform()
            transform.RotateX(90)  # Rotate around the X-axis to make it vertical

            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetTransform(transform)
            transform_filter.SetInputConnection(cylinder.GetOutputPort())
            transform_filter.Update()

            cylinder_actor = vtk.vtkActor()
            cylinder_mapper = vtk.vtkPolyDataMapper()
            cylinder_mapper.SetInputConnection(transform_filter.GetOutputPort())
            cylinder_actor.SetMapper(cylinder_mapper)
            cylinder_actor.SetPosition(px, py, -ba.anchor_props.hef_default / 2 + 0.5)
            cylinder_actor.GetProperty().SetColor(0, 0, 0)
            renderer.AddActor(cylinder_actor)

    # Create floor plates
    for plate in model.elements.base_plates:
        for boundary in plate.props.bearing_boundaries:
            # Create vtkPoints from boundary points
            points = vtk.vtkPoints()
            for x, y in boundary:
                points.InsertNextPoint(x, y, 0)

            # Triangulate the polygon if necessary
            delaunay = vtk.vtkDelaunay2D()
            delaunay.SetInputData(vtk.vtkPolyData())
            delaunay.GetInput().SetPoints(points)
            delaunay.Update()

            # Create a mapper and actor for the floor plate
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(delaunay.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.5, 0.5, 0.5)
            actor.GetProperty().SetOpacity(0.7)
            renderer.AddActor(actor)

    # Create Connection Elements
    for cxn in model.elements.base_plate_connections:
        cxn_props = cxn.bolt_group_props
        d = 0
        npz_min = [0, -0.5 * cxn_props.w, -0.5 * cxn_props.h]
        npz_max = [d, 0.5 * cxn_props.w, 0.5 * cxn_props.h]
        xyz_min = [a + b for a, b in zip(_npz_to_xyz(npz_min, normal_vector=cxn_props.local_z), cxn_props.plate_centroid_XYZ)]
        xyz_max = [a + b for a, b in zip(_npz_to_xyz(npz_max, normal_vector=cxn_props.local_z), cxn_props.plate_centroid_XYZ)]
        actor = _create_vtk_box(xyz_min, xyz_max, type="filled", facecolor=(0.75, 0.75, 0.75))
        renderer.AddActor(actor)

        for pz_anchor in cxn_props.xy_anchors:
            l_anchor = d + 1
            npz_anchor = [-0.5, pz_anchor[0], pz_anchor[1]]
            xyz_anchor = [a + b for a, b in
                          zip(_npz_to_xyz(npz_anchor, normal_vector=cxn_props.local_z), cxn_props.plate_centroid_XYZ)]
            rotate_z = np.degrees(np.arccos(np.dot(cxn_props.local_z, (0, 1, 0))))
            actor = _create_vtk_cylinder(xyz_anchor, l_anchor, rotate_z=rotate_z, radius=0.2, color=(1, 0, 0))
            renderer.AddActor(actor)

    # Create Base Strap Elements
    # for strap in equipment_obj.base_straps:
    #     xyz_i = np.array(strap.xyz_i)
    #     xyz_j = np.array(strap.xyz_j)
    #
    #     # Create a line source
    #     line_source = vtk.vtkLineSource()
    #     line_source.SetPoint1(*xyz_i)  # Start point
    #     line_source.SetPoint2(*xyz_j)  # End point
    #
    #     # Create mapper and actor
    #     line_mapper = vtk.vtkPolyDataMapper()
    #     line_mapper.SetInputConnection(line_source.GetOutputPort())
    #
    #     line_actor = vtk.vtkActor()
    #     line_actor.SetMapper(line_mapper)
    #
    #     # Set line width for visibility
    #     line_actor.GetProperty().SetLineWidth(15)  # Adjust thickness as needed
    #     line_actor.GetProperty().SetColor(0.25, 0.25, 0.25)  # Gray color
    #
    #     # Add to the renderer
    #     renderer.AddActor(line_actor)

    # Create Bracket Elements
    # if not equipment_obj.omit_bracket_output:
    #     for bracket in equipment_obj.wall_brackets:
    #         xyz_0 = np.array(bracket.xyz_equipment)
    #         xyz_f = np.array(bracket.xyz_wall)
    #         height = np.linalg.norm(xyz_0 - xyz_f)
    #         if bracket.supporting_wall[0] == 'X':
    #             rotate_z = 90
    #         else:
    #             rotate_z = 0
    #         xyz = np.mean((xyz_0, xyz_f), axis=0)
    #         cylinder_actor = _create_vtk_cylinder(xyz, height, radius=0.5, rotate_z=rotate_z, color=(0.5, 0.5, 0.5))
    #         renderer.AddActor(cylinder_actor)

    # Create Backing Elements
    # n_coord = {k: -v for k, v in wall_coordinate.items()}
    #
    # for backing in equipment_obj.wall_backing:
    #     xyz_cent = backing.centroid
    #     npz_cent = _xyz_to_npz(xyz_cent,backing.supporting_wall)
    #     d = backing.d
    #     npz_min = [0 + d, backing.x_bar - backing.w / 2, backing.y_bar - backing.h / 2] \
    #         + npz_cent
    #     npz_max = [0, backing.x_bar + backing.w / 2, backing.y_bar + backing.h / 2] \
    #         + npz_cent
    #     xyz_min = _npz_to_xyz(npz_min, backing.supporting_wall)
    #     xyz_max = _npz_to_xyz(npz_max, backing.supporting_wall)
    #     actor = _create_vtk_box(xyz_min, xyz_max, type="filled", facecolor=(0.75, 0.75, 0.75))
    #     renderer.AddActor(actor)
    #
    #     for anchor in backing.pz_anchors:
    #         l_anchor = d + 1.5
    #         npz_anchor = [0 + d - 0.5, anchor[0] + backing.x_bar, anchor[1] + backing.y_bar] + npz_cent
    #         xyz_anchor = _npz_to_xyz(npz_anchor, backing.supporting_wall)
    #         if backing.supporting_wall[0] == 'X':
    #             rotate_z = 90
    #         else:
    #             rotate_z = 0
    #         actor = _create_vtk_cylinder(xyz_anchor, l_anchor, rotate_z=rotate_z, radius=0.25, color=(1, 0, 0))
    #         renderer.AddActor(actor)

    # Create axis arrows
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(max(H, Bx, By) / 10,
                        max(H, Bx, By) / 10,
                        max(H, Bx, By) / 10)
    renderer.AddActor(axes)
    axes.SetXAxisLabelText('')
    axes.SetYAxisLabelText('')
    axes.SetZAxisLabelText('')

    # todo: figure out a way to add labels to the axes. Lots of GPT debugging and couldn't get it to work

    # axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0, 0, 0)  # Black color for X axis
    # axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0, 0, 0)  # Black color for Y axis
    # axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0, 0, 0)  # Black color for Z axis
    #
    # axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(6)  # Larger font size
    # axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(100)
    # axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(100)

    # # Add custom text actors for labels
    # x_text = vtk.vtkTextActor3D()
    # x_text.SetInput("X Axis")
    # x_text.GetTextProperty().SetFontSize(12)  # Set the desired font size
    # x_text.GetTextProperty().SetColor(0, 0, 0)  # Set color to black
    # x_text.SetPosition(1.2 * max(H, Bx, By) / 10, 0, 0)  # Position near X-axis
    #
    # y_text = vtk.vtkTextActor3D()
    # y_text.SetInput("Y Axis")
    # y_text.GetTextProperty().SetFontSize(12)  # Set the desired font size
    # y_text.GetTextProperty().SetColor(0, 0, 0)  # Set color to black
    # y_text.SetPosition(0, 1.2 * max(H, Bx, By) / 10, 0)  # Position near Y-axis
    #
    # z_text = vtk.vtkTextActor3D()
    # z_text.SetInput("Z Axis")
    # z_text.GetTextProperty().SetFontSize(12)  # Set the desired font size
    # z_text.GetTextProperty().SetColor(0, 0, 0)  # Set color to black
    # z_text.SetPosition(0, 0, 1.2 * max(H, Bx, By) / 10)  # Position near Z-axis
    #
    # renderer.AddActor2D(x_text)
    # renderer.AddActor2D(y_text)
    # renderer.AddActor2D(z_text)

    # Set up a custom camera
    camera = vtk.vtkCamera()
    camera.SetPosition(-0.75 * Bx, -4 * By, 1.25 * H)
    camera.SetFocalPoint(0, 0, 0.5 * H)
    camera.SetViewUp(0, 0, 1)
    camera.Zoom(0.5)
    renderer.SetActiveCamera(camera)

    render_window = vtk.vtkRenderWindow()
    figw = 3  # inches
    figh = 3  # inches
    dpi = 600  # dots per inch
    render_window.SetSize(int(figw * dpi), int(figh * dpi))
    # render_window.SetSize(900, 900)
    render_window.OffScreenRenderingOn()
    render_window.AddRenderer(renderer)

    # Render and capture the image
    render_window.Render()
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    # Save the image to a file >>>>> Now a separate function vtk_save
    # writer = vtk.vtkPNGWriter()
    # if not filename:
    #     filename = "{}.png".format(str(uuid.uuid4()))
    # else:
    #     filename = f'{filename}.png'
    # filepath = posixpath.join(TEMP_DIR, filename)
    # filepath = filepath.replace('\\', '/')
    # writer.SetFileName(filepath)
    # writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    # writer.Write()

    return window_to_image_filter, figw


def equipment_plan_view(model_record: ModelRecord, _):
    equipment_obj = model_record.model
    width = 3
    height = 3

    margin = 0.125

    wratio = (width - margin) / width
    hratio = (height - margin) / height
    # Plotting
    fig = plt.figure(figsize=(width, height))
    ax_plan = fig.add_subplot(121)
    ax_plan.set_position([(1 - wratio), (1 - hratio), wratio, hratio])  # left, bottom, width, height

    # Plot bearing boundaries as solid black outlines
    for plate in equipment_obj.elements.base_plates:
        for boundary in plate.props.bearing_boundaries:
            if len(boundary) > 0:
                polygon = patches.Polygon(boundary, closed=True, fill=False, edgecolor='black')
                ax_plan.add_patch(polygon)

    # Plot anchor points as solid circles
    if equipment_obj.elements.base_anchors:
        for ba in equipment_obj.elements.base_anchors:
            for point in ba.geo_props.xy_anchors:
                circle = patches.Circle(point, radius=0.25, color='black')
                ax_plan.add_patch(circle)

    # Plot Center of Gravity
    ax_plan.plot(equipment_obj.equipment_props.ex, equipment_obj.equipment_props.ey, 'ok', markersize=8, markerfacecolor='w')
    ax_plan.plot(equipment_obj.equipment_props.ex, equipment_obj.equipment_props.ey, '+k', markersize=12)
    ax_plan.text(equipment_obj.equipment_props.ex + 0.05 * equipment_obj.equipment_props.Bx,
                 equipment_obj.equipment_props.ey + 0.05 * equipment_obj.equipment_props.By,
                 f'C.G.', ha='left', va='bottom')

    # Draw a semi-opaque box centered around the origin with extents W, D
    Bx, By = (equipment_obj.equipment_props.Bx, equipment_obj.equipment_props.By)
    rectangle = patches.Rectangle((-Bx / 2, -By / 2), Bx, By, linewidth=3, edgecolor='gray', facecolor='none',
                                  alpha=0.5)
    ax_plan.add_patch(rectangle)

    # Dimension line: horizontal (Bx)
    y_offset = -By / 2 - 1.0  # below the bounding box
    ax_plan.annotate(
        '', xy=(-Bx / 2, y_offset), xytext=(Bx / 2, y_offset),
        arrowprops=dict(arrowstyle='<->', color='black')
    )
    ax_plan.text(0, y_offset - 0.2, f'$B_x$ = {Bx:.2f}', ha='center', va='top')

    # Dimension line: vertical (By)
    x_offset = -Bx / 2 - 1.0  # to the left of the bounding box
    ax_plan.annotate(
        '', xy=(x_offset, -By / 2), xytext=(x_offset, By / 2),
        arrowprops=dict(arrowstyle='<->', color='black')
    )
    ax_plan.text(x_offset - 0.2, 0, f'$B_y$ = {By:.2f}', ha='right', va='center', rotation='vertical')

    # Set equal aspect ratio
    ax_plan.spines['top'].set_visible(False)
    ax_plan.spines['right'].set_visible(False)
    ax_plan.spines['bottom'].set_position(('data', 0))
    # ax.spines['bottom'].set_color('lightgray')
    ax_plan.spines['bottom'].set_zorder(0)
    ax_plan.spines['left'].set_position(('data', 0))
    # ax.spines['left'].set_color('lightgray')
    ax_plan.spines['left'].set_zorder(0)
    ax_plan.autoscale_view()
    ax_plan.set_aspect('equal')
    return fig, width


def equipment_elevation_view(model_record: ModelRecord, _):
    equipment_obj = model_record.model
    eprops = equipment_obj.equipment_props
    width = 3
    height = 3

    margin = 0.25

    wratio = (width - margin) / width
    hratio = (height - margin) / height

    # Plotting
    fig = plt.figure(figsize=(width, height))
    ax_elev = fig.add_subplot(121)
    ax_elev.set_position(((1 - wratio), (1 - hratio), wratio, hratio))  # left, bottom, width, height

    Bx = eprops.Bx
    H = eprops.H
    ex = eprops.ex
    zCG = eprops.zCG

    # Draw the elevation bounding box
    rectangle = patches.Rectangle((-Bx / 2, 0), Bx, H, linewidth=3, edgecolor='gray', facecolor='none', alpha=0.5)
    ax_elev.add_patch(rectangle)

    # Plot the Center of Gravity
    ax_elev.plot(ex, zCG, 'ok', markersize=8, markerfacecolor='w')
    ax_elev.plot(ex, zCG, '+k', markersize=12)
    ax_elev.text(ex + 0.05 * Bx, zCG + 0.05 * H, 'C.G.', ha='left', va='bottom')

    # Dimension line: horizontal (Bx)
    y_offset = -0.05 * eprops.H  # slightly below zero
    ax_elev.annotate(
        '', xy=(-Bx / 2, y_offset), xytext=(Bx / 2, y_offset),
        arrowprops=dict(arrowstyle='<->', color='black')
    )
    ax_elev.text(0, y_offset - .1, f'$B_x$ = {Bx:.2f}', ha='center', va='top')

    # Dimension line: vertical (H) on left
    x_offset_left = -Bx / 2 - 1.0
    ax_elev.annotate(
        '', xy=(x_offset_left, 0), xytext=(x_offset_left, H),
        arrowprops=dict(arrowstyle='<->', color='black')
    )
    ax_elev.text(x_offset_left - 0.2, H / 2, f'$H$ = {H:.2f}', ha='right', va='center', rotation='vertical')

    # Dimension line: vertical (zCG) on right
    x_offset_right = Bx / 2 + 1.0
    ax_elev.annotate(
        '', xy=(x_offset_right, 0), xytext=(x_offset_right, zCG),
        arrowprops=dict(arrowstyle='<->', color='black')
    )
    ax_elev.text(x_offset_right + 0.2, zCG / 2, f'$z_{{CG}}$ = {zCG:.2f}', ha='left', va='center', rotation='vertical')

    # Set axes appearance
    # ax_elev.spines['top'].set_visible(False)
    # ax_elev.spines['right'].set_visible(False)
    # ax_elev.spines['bottom'].set_position(('data', 0))
    # ax_elev.spines['bottom'].set_zorder(0)
    # ax_elev.spines['left'].set_position(('data', 0))
    # ax_elev.spines['left'].set_zorder(0)
    ax_elev.set_xlim(-Bx / 2 - 2, Bx / 2 + 2)
    ax_elev.set_ylim(y_offset, H + 1)
    ax_elev.set_aspect('equal')
    ax_elev.autoscale_view()
    ax_elev.axis('off')
    return fig, width


def base_equilibrium(model_record: ModelRecord):
    """Wrapper function to plot base equilibrium for goverining anchor tension condition"""
    model = model_record.model
    run_idx = model_record.governing_run
    run = model_record.analysis_runs[run_idx]
    ba_res, ba_idx = get_governing_result(run.results.base_anchors)

    method = model.elements.base_anchors[ba_idx].factor_method
    theta_idx = ba_res.governing_theta_idx

    theta_z = run.solutions[method].theta_z[theta_idx]
    cz_states = [res.compression_zones[theta_idx] for res in run.results.base_plates]
    anchor_forces = [res.anchor_forces[:,:,theta_idx] for res in run.results.base_plates]

    return _equilibrium_plan_view(model, cz_states, anchor_forces, theta_z)


def _equilibrium_plan_view(model: EquipmentModel,
                           cz_states: List[bp.CompressionZoneState],
                           anchor_forces: List[NDArray],
                           theta_z):
    By = model.equipment_props.By
    Bx = model.equipment_props.Bx

    width = 3.5
    min_h_to_w = 0.5

    if Bx == By:
        height = width
    elif Bx > By:
        width = width
        height = width * max(By / Bx, min_h_to_w)
    else:
        height = 3
        width = height * max(Bx / By, 1 / min_h_to_w)

    margin = 0

    wratio = (width - margin) / width
    hratio = (height - margin) / height
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    ax.set_position(((1 - wratio), (1 - hratio), wratio, hratio))  # left, bottom, width, height

    texts = []

    text_offset = 1

    max_length = 0.25 * max([Bx, By])  # Maximum arrow length for applied load
    sf = max_length / model.factored_loads.lrfd.Fh

    # Plot compression zones
    i = 0
    for cz in cz_states:
        for boundary, load, centroid in zip(cz.compression_boundaries,
                                            cz.fz,
                                            cz.resultant_centroids):
            # Shade area in compression
            polygon = patches.Polygon(boundary, closed=True, facecolor='blue', alpha=0.5)
            ax.add_patch(polygon)

            # Mark and label compression centroid
            ax.plot(centroid[0], centroid[1], '+k')
            label = ax.text(centroid[0], centroid[1], f'({load:.0f} lbs)', color='blue', verticalalignment='center',
                            horizontalalignment='left')
            texts.append(label)

    # Plot bearing boundaries as solid black outlines
    for plate in model.elements.base_plates:
        for boundary in plate.props.bearing_boundaries:
            if len(boundary) > 0:
                polygon = patches.Polygon(boundary, closed=True, fill=False, edgecolor='black')
                ax.add_patch(polygon)

    # Plot Anchor Points and Forces
    for plate, forces in zip(model.elements.base_plates, anchor_forces):
        for vxvyt, xy in zip(forces, plate.props.xy_anchors):
            vx, vy, ten = vxvyt
            ax.arrow(xy[0], xy[1], -vx * sf, -vy * sf, head_width=0.2, color='green', linewidth=1.5)

            if ten > 0:
                circle = patches.Circle(xy, radius=0.25, color='red')
                label = ax.text(xy[0], xy[1], f'{ten:.0f} lbs', color='red', verticalalignment='center',
                                horizontalalignment='left')
                texts.append(label)
            else:
                circle = patches.Circle(xy, radius=0.25, color='black')
            ax.add_patch(circle)

    # # Plot anchor points as solid circles
    # max_length = 0.25 * max([equipment_obj.Bx, equipment_obj.By])  # Maximum arrow length for applied load
    # if equipment_obj.base_anchors:
    #
    #     # Plot arrows for anchor forces
    #     sf = max_length / equipment_obj.Fh
    #
    #     for i, (point, load) in enumerate(zip(equipment_obj.base_anchors.xy_anchors,
    #                                           [(el.anchor_result['vx'],
    #                                             el.anchor_result['vx'],
    #                                             el.anchor_result['tension'])
    #                                             for el in equipment_obj.floor_plates if el.n_anchor > 0])):
    #
    #         ax.arrow(point[0], point[1], -load[1] * sf, -load[2] * sf, head_width=0.2, color='green', linewidth=1.5)
    #
    #         if load[0] > 0:
    #             circle = patches.Circle(point, radius=0.25, color='red')
    #
    #             label = ax.text(point[0], point[1], f'{load[0]:.0f} lbs', color='red', verticalalignment='center',
    #                             horizontalalignment='left')
    #
    #             texts.append(label)
    #
    #         else:
    #             circle = patches.Circle(point, radius=0.25, color='black')
    #         ax.add_patch(circle)
            # label = ax.text(point[0] + text_offset, point[1] - text_offset, f'{i + 1:.0f}', color='red', fontsize=10,
            #                 ha='left',
            #                 va='center',
            #                 bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.1', linewidth=0.5),
            #                 zorder=1)
            # texts.append(label)

    # Draw a semi-opaque box centered around the origin with extents Bx, By
    rectangle = patches.Rectangle((-Bx / 2, -By / 2), Bx, By, linewidth=3, edgecolor='gray', facecolor='none',
                                  alpha=0.5)
    ax.add_patch(rectangle)

    # Draw Arrow for Loading Direction
    l = max_length  # max(equipment_obj.Bx, equipment_obj.By) * 0.25
    ax.annotate('',
                xy=(model.equipment_props.ex + l * np.cos(theta_z),
                    model.equipment_props.ey + l * np.sin(theta_z)),
                xycoords='data',
                xytext=(model.equipment_props.ex, model.equipment_props.ey),
                textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.05, width=3, headwidth=9, headlength=12))

    ax.autoscale_view()
    # if width > height:
    #     xmin, xmax = ax.get_xlim()
    #     xrange = xmax - xmin
    #     yrange = (height/width)*xrange
    #     ax.set_ylim(-0.5*yrange,0.5*yrange)
    # else:
    #     ymin, ymax = ax.get_ylim()
    #     yrange = ymax - ymin
    #     xrange = (width/height)*yrange
    #     ax.set_xlim(-0.5*xrange, 0.5 * xrange)

    # Set equal aspect ratio
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['bottom'].set_zorder(0)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['left'].set_zorder(0)

    ax.set_aspect('equal')

    # Adjust the annotations to avoid overlap and keep them within the figure bounds
    adjust_text(texts, ax=ax, expand=(1.5, 2), arrowprops=dict(arrowstyle="->", color='black', lw=0.5),
                min_arrow_len=20)

    # plt.show()
    return fig, width


def base_displaced_shape(model_record: ModelRecord):
    """ plots displaced shape for governing base anchor solution """
    model = model_record.model
    run_idx = model_record.governing_run
    run = model_record.analysis_runs[run_idx]
    ba_res, ba_idx = get_governing_result(run.results.base_anchors)

    method = model.elements.base_anchors[ba_idx].factor_method
    theta_idx = ba_res.governing_theta_idx

    sol = run.solutions[method].equilibrium_solutions[:,theta_idx]
    theta_z = run.solutions[method].theta_z[theta_idx]
    fh, fv = model.factored_loads.get(method)
    return _displaced_shape(model, sol, theta_z, fh, fv)


def bracket_displaced_shape(equipment_obj):
    """ plots displaced shape for governing bracket solution """
    return _displaced_shape(equipment_obj,
                            equipment_obj.governing_solutions['wall_bracket_tension']['sol'],
                            equipment_obj.governing_solutions['wall_bracket_tension']['theta_z'])


def wall_displaced_shape(equipment_obj, _):
    """ plots displaced shape for governing wall anchor solution"""
    return _displaced_shape(equipment_obj,
                            equipment_obj.governing_solutions['wall_anchor_tension']['sol'],
                            equipment_obj.governing_solutions['wall_anchor_tension']['theta_z'])


def _displaced_shape(model: EquipmentModel,
                     sol, theta_z, fh, fv, sf=None, width=2.75):

    if not sf:
        sf = _get_scale_factor(model, sol)

    width = width
    fig = plt.figure(figsize=(width, width))
    ax = fig.add_subplot(111, projection='3d')

    bx, by, h = model.equipment_props.Bx, model.equipment_props.By, model.equipment_props.H
    color_ud = 'darkgray'  # Undispalced Shape color
    lw_ud = 0.75  # Undisplaced linewidth
    ls_ud = '-'  # Undisplaced line style
    zorder_ud = 1

    color_disp = 'blue'
    lw_disp = 0.75
    ls_disp = '-'
    zorder_disp = 1

    '''Plot the Undisplaced Shape'''
    # Bounding Box
    box_vertices = np.array([
        [-bx / 2, -by / 2, 0],
        [bx / 2, -by / 2, 0],
        [bx / 2, by / 2, 0],
        [-bx / 2, by / 2, 0],
        [-bx / 2, -by / 2, h],
        [bx / 2, -by / 2, h],
        [bx / 2, by / 2, h],
        [-bx / 2, by / 2, h]
    ])

    _plot_wireframe_box(ax, box_vertices, color=color_ud, linewidth=lw_ud, linestyle=ls_ud)

    # Base Plate Elements
    for plate in model.elements.base_plates:
        for boundary in plate.props.bearing_boundaries:
            if len(boundary) > 0:
                boundary_closed = np.vstack((boundary, boundary[0, :]))
                ax.plot(boundary_closed[:, 0], boundary_closed[:, 1], np.zeros_like(boundary_closed[:, 0]),
                        color=color_ud,
                        linewidth=lw_ud, linestyle=ls_ud, solid_capstyle='round')
        # todo:[Plotting Hardware Connection] if plate.connection: plot connection
        # Create Connection Elements
        for cxn in model.elements.base_plate_connections:
            d = 0  # No offset in the NPZ X-direction for plate connection

            # Compute bounding box in NPZ coordinates
            npz_min = [0, -0.5 * cxn.bolt_group_props.w, -0.5 * cxn.bolt_group_props.h]
            npz_max = [d, 0.5 * cxn.bolt_group_props.w, 0.5 * cxn.bolt_group_props.h]

            # Convert to XYZ and offset by centroid
            xyz_min = [a + b for a, b in zip(_npz_to_xyz(npz_min, normal_vector=cxn.bolt_group_props.local_z), cxn.bolt_group_props.plate_centroid_XYZ)]
            xyz_max = [a + b for a, b in zip(_npz_to_xyz(npz_max, normal_vector=cxn.bolt_group_props.local_z), cxn.bolt_group_props.plate_centroid_XYZ)]

            # Build wireframe vertices of the box
            xmin, ymin, zmin = xyz_min
            xmax, ymax, zmax = xyz_max
            vertices = np.array([
                [xmin, ymin, zmin],
                [xmax, ymin, zmin],
                [xmax, ymax, zmin],
                [xmin, ymax, zmin],
                [xmin, ymin, zmax],
                [xmax, ymin, zmax],
                [xmax, ymax, zmax],
                [xmin, ymax, zmax]
            ])
            _plot_wireframe_box(ax, vertices, color=color_ud, linewidth=lw_ud, linestyle=ls_ud)

            # Plot anchors
            for anchor in cxn.bolt_group_props.xy_anchors:
                l_anchor = 0.5  # Length of anchor projection
                npz_anchor_0 = [-0.5, anchor[0], anchor[1]]  # start
                npz_anchor_f = [-0.5 - l_anchor, anchor[0], anchor[1]]  # end (projecting into normal vector direction)

                xyz_anchor_0 = [a + b for a, b in
                                zip(_npz_to_xyz(npz_anchor_0, normal_vector=cxn.bolt_group_props.local_z), cxn.bolt_group_props.plate_centroid_XYZ)]
                xyz_anchor_f = [a + b for a, b in
                                zip(_npz_to_xyz(npz_anchor_f, normal_vector=cxn.bolt_group_props.local_z), cxn.bolt_group_props.plate_centroid_XYZ)]

                ax.plot([xyz_anchor_0[0], xyz_anchor_f[0]],
                        [xyz_anchor_0[1], xyz_anchor_f[1]],
                        [xyz_anchor_0[2], xyz_anchor_f[2]],
                        color=color_ud, marker='o', linestyle='-',
                        markersize=2, markeredgewidth=0)

    # Base Anchors
    for ba in model.elements.base_anchors:
        for px, py in ba.geo_props.xy_anchors:
            ax.plot([px, px], [py, py], [0, -ba.anchor_props.hef_default], color=color_ud, marker='o', linestyle='-',
                    markersize=2, markeredgewidth=0)

    # Base Straps
    # todo: base_straps
    # for strap in model.elements.base_straps:
    #     xyz_0 = np.array(strap.xyz_i)
    #     xyz_f = np.array(strap.xyz_j)
    #     ax.plot([xyz_0[0], xyz_f[0]], [xyz_0[1], xyz_f[1]], [xyz_0[2], xyz_f[2]], color=color_ud, marker='o',
    #             linestyle='-',
    #             markersize=2, markeredgewidth=0)

    # # Wall Brackets
    # wall_coordinate = _get_wall_coordinates(model)
    # if not model.omit_bracket_output:
    #     for bracket in model.elements.wall_brackets:
    #         xyz_0 = np.array(bracket.xyz_equipment)
    #         xyz_f = np.array(bracket.xyz_wall)
    #         ax.plot([xyz_0[0], xyz_f[0]], [xyz_0[1], xyz_f[1]], [xyz_0[2], xyz_f[2]], color=color_ud, marker='o',
    #                 linestyle='-',
    #                 markersize=4, markeredgewidth=0, linewidth=3)
    #
    # # Wall Backing
    # n_coord = {k: -v for k, v in wall_coordinate.items()}
    #
    # for backing in model.wall_backing:
    #     xyz_cent = backing.centroid
    #     npz_cent = _xyz_to_npz(xyz_cent, backing.supporting_wall)
    #     d = backing.d
    #
    #     for anchor in backing.pz_anchors:
    #         l_anchor = 3
    #         npz_anchor_0 = [0, anchor[0] + backing.x_bar, anchor[1] + backing.y_bar] + npz_cent
    #
    #         npz_anchor_f = [- l_anchor, anchor[0] + backing.x_bar, anchor[1] + backing.y_bar] + npz_cent
    #
    #         xyz_anchor_0 = _npz_to_xyz(npz_anchor_0, backing.supporting_wall)
    #         xyz_anchor_f = _npz_to_xyz(npz_anchor_f, backing.supporting_wall)
    #
    #         ax.plot([xyz_anchor_0[0], xyz_anchor_f[0]],
    #                 [xyz_anchor_0[1], xyz_anchor_f[1]],
    #                 [xyz_anchor_0[2], xyz_anchor_f[2]], color=color_ud, marker='o', linestyle='-',
    #                 markersize=2, markeredgewidth=0)
    #
    #     npz_min = [0 + d, backing.x_bar - backing.w / 2, backing.y_bar - backing.h / 2] \
    #         + npz_cent
    #     npz_max = [0, backing.x_bar + backing.w / 2, backing.y_bar + backing.h / 2] \
    #         + npz_cent
    #     xyz_min = _npz_to_xyz(npz_min, backing.supporting_wall)
    #     xyz_max = _npz_to_xyz(npz_max, backing.supporting_wall)
    #     xmin, ymin, zmin = xyz_min
    #     xmax, ymax, zmax = xyz_max
    #     vertices = np.array([[xmin, ymin, zmin],
    #                          [xmax, ymin, zmin],
    #                          [xmax, ymax, zmin],
    #                          [xmin, ymax, zmin],
    #                          [xmin, ymin, zmax],
    #                          [xmax, ymin, zmax],
    #                          [xmax, ymax, zmax],
    #                          [xmin, ymax, zmax]])
    #     _plot_wireframe_box(ax, vertices, color=color_ud, linewidth=lw_ud, linestyle=ls_ud)

    '''Plot the Displaced Shape'''
    # Bounding Box
    local_dofs = sol[0:6]
    displaced_vertices = box_vertices + _get_coordinate_displacements(box_vertices, local_dofs, sf=sf)
    _plot_wireframe_box(ax, displaced_vertices, color=color_disp, linewidth=lw_disp, linestyle=ls_disp)

    # Base Plate Elements


    for plate in model.elements.base_plates:
        local_dofs = plate.dof_props.C @ sol
        for boundary in plate.props.bearing_boundaries:
            if len(boundary) > 0:
                boundary_closed = np.vstack((boundary, boundary[0, :]))
                boundary_closed = np.hstack([boundary_closed, np.zeros([len(boundary_closed[:, 0]), 1])])
                boundary_local = boundary_closed - np.array([plate.props.x0, plate.props.y0, plate.props.z0])
                disp_bound = boundary_closed + _get_coordinate_displacements(boundary_local, local_dofs, sf=sf)
                ax.plot(disp_bound[:, 0], disp_bound[:, 1], disp_bound[:, 2], color=color_disp, linewidth=lw_disp,
                        linestyle=ls_disp, solid_capstyle='round')

            # Base Anchors
            if plate.props.xy_anchors.size > 0:
                anchor = np.hstack([plate.props.xy_anchors, np.zeros([len(plate.props.xy_anchors[:, 0]), 1])])
                anchor_local = anchor - np.array([plate.props.x0, plate.props.y0, plate.props.z0])
                local_dofs = plate.dof_props.C @ sol
                anchor_disp = anchor + _get_coordinate_displacements(anchor_local, local_dofs, sf=sf)
                for (dx, dy, dz), (x, y, z) in zip(anchor_disp, anchor):
                    ax.plot([dx, x], [dy, y], [dz, -model.elements.base_anchors[0].anchor_props.hef_default], color=color_disp, marker='o',
                            linestyle=ls_ud,
                            markersize=2, markeredgewidth=0)

    #todo: switch to for ele in elements loops for cxn and anchors
    for cxn in model.elements.base_plate_connections:
        cxn_props = cxn.bolt_group_props
        d = 0  # No offset in the NPZ X-direction for plate connection

        # Compute bounding box in NPZ coordinates
        npz_min = [0, -0.5 * cxn_props.w, -0.5 * cxn_props.h]
        npz_max = [d, 0.5 * cxn_props.w, 0.5 * cxn_props.h]

        # Convert to XYZ and offset by centroid
        xyz_min = [a + b for a, b in zip(_npz_to_xyz(npz_min, normal_vector=cxn_props.local_z), cxn_props.plate_centroid_XYZ)]
        xyz_max = [a + b for a, b in zip(_npz_to_xyz(npz_max, normal_vector=cxn_props.local_z), cxn_props.plate_centroid_XYZ)]

        # xyz_min += _get_coordinate_displacements([xyz_min], sol[0:6], sf=sf)[0]
        # xyz_max += _get_coordinate_displacements([xyz_max], sol[0:6], sf=sf)[0]

        # Build wireframe vertices of the box
        xmin, ymin, zmin = xyz_min
        xmax, ymax, zmax = xyz_max
        vertices = np.array([
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax]
        ])
        vertices += _get_coordinate_displacements(vertices, sol[0:6], sf=sf)
        _plot_wireframe_box(ax, vertices, color=color_disp, linewidth=lw_disp, linestyle=ls_disp)

        # Plot anchors
        for anchor in cxn.bolt_group_props.xy_anchors:
            l_anchor = 0.5  # Length of anchor projection
            npz_anchor_0 = [-0.5, anchor[0], anchor[1]]  # start
            npz_anchor_f = [-0.5 - l_anchor, anchor[0], anchor[1]]  # end (projecting into normal vector direction)

            xyz_anchor_0 = [a + b for a, b in
                            zip(_npz_to_xyz(npz_anchor_0, normal_vector=cxn_props.local_z), cxn_props.plate_centroid_XYZ)]
            xyz_anchor_f = [a + b for a, b in
                            zip(_npz_to_xyz(npz_anchor_f, normal_vector=cxn_props.local_z), cxn_props.plate_centroid_XYZ)]

            xyz_anchor_0 += _get_coordinate_displacements([xyz_anchor_0], sol[0:6], sf=sf)[0]
            xyz_anchor_f += _get_coordinate_displacements([xyz_anchor_f], sol[0:6], sf=sf)[0]

            ax.plot([xyz_anchor_0[0], xyz_anchor_f[0]],
                    [xyz_anchor_0[1], xyz_anchor_f[1]],
                    [xyz_anchor_0[2], xyz_anchor_f[2]],
                    color=color_disp, marker='o', linestyle=ls_disp,
                    markersize=2, markeredgewidth=0)

    # # Base Straps #todo continue here
    # for strap in model.base_straps:
    #     xyz_0 = np.array(strap.xyz_i)
    #     xyz_0 += _get_coordinate_displacements([xyz_0], sol[0:6], sf=sf)[0]
    #     xyz_f = np.array(strap.xyz_j)
    #     xyz_0p = np.array([strap.base_plate.x0, strap.base_plate.y0, strap.base_plate.z0])
    #     xyz_f += _get_coordinate_displacements(xyz_f - xyz_0p, strap.base_plate.C @ sol, sf=sf)[0]
    #     ax.plot([xyz_0[0], xyz_f[0]], [xyz_0[1], xyz_f[1]], [xyz_0[2], xyz_f[2]], color=color_disp, marker='o',
    #             linestyle='-',
    #             markersize=2, markeredgewidth=0)
    #
    # # Wall Bracket Elements
    # if not model.omit_bracket_output:
    #     for bracket in model.wall_brackets:
    #         xyz_0 = np.array(bracket.xyz_equipment)
    #         xyz_f = np.array(bracket.xyz_wall)
    #         xyz_0 = xyz_0 + _get_coordinate_displacements([xyz_0], sol[0:6], sf=sf)
    #         xyz_0 = xyz_0[0]
    #         ax.plot([xyz_0[0], xyz_f[0]], [xyz_0[1], xyz_f[1]], [xyz_0[2], xyz_f[2]], color=color_disp, marker='o',
    #                 linestyle='-',
    #                 markersize=4, markeredgewidth=0, linewidth=3)

    # Load Arrows
    loads = model.factored_loads.lrfd
    fv = loads.Fv_max  #todo: logic to select which force to use
    max_arrow_length = (model.equipment_props.Bx + model.equipment_props.By) / 2
    fh_arrow_length = loads.Fh * max_arrow_length / max(abs(loads.Fh), abs(fv))
    fv_arrow_length = fv * max_arrow_length / max(abs(loads.Fh), abs(fv))

    ax.quiver(model.equipment_props.ex, model.equipment_props.ey, model.equipment_props.zCG,
              0, 0, fv_arrow_length,
              color='darkblue', arrow_length_ratio=0.2, capstyle='round')

    ax.quiver(model.equipment_props.ex, model.equipment_props.ey, model.equipment_props.zCG,
              fh_arrow_length * np.cos(theta_z),
              fh_arrow_length * np.sin(theta_z),
              0, color='r', arrow_length_ratio=0.2, capstyle='round')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_aspect('equal')

    # Turn off the gridlines
    ax.grid(False)

    # Show only the gray background on the xy plane
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0))

    shade = (0.9, 0.9, 0.9, 1)
    azim = -60
    if model.elements.base_plates:
        ax.zaxis.set_pane_color(shade)
        azim = -60
    # if model.wall_anchors['X+'] is not None:
    #     azim = -120
    #     ax.xaxis.set_pane_color(shade)
    # if model.wall_anchors['X-'] is not None:
    #     azim = -60
    #     ax.xaxis.set_pane_color(shade)
    # if model.wall_anchors['Y-'] is not None:
    #     azim = 60
    #     ax.yaxis.set_pane_color(shade)
    # if model.wall_anchors['Y+'] is not None:
    #     azim = -120
    #     ax.yaxis.set_pane_color(shade)

    ax.view_init(azim=azim)

    return fig, width


def base_anchors_vs_theta(model_record: ModelRecord):
    model = model_record.model
    run_idx = model_record.governing_run
    run = model_record.analysis_runs[run_idx]

    ba_res, ba_idx = get_governing_result(run.results.base_anchors)
    method = model.elements.base_anchors[ba_idx].factor_method
    converged = run.solutions[method].converged
    theta_z = run.solutions[method].theta_z
    matrix_t = ba_res.forces[:, 2, converged]
    matrix_v = (ba_res.forces[:, 0, converged] ** 2 +
                ba_res.forces[:, 1, converged] ** 2) ** 0.5
    matrix_unity = ba_res.unity_by_anchor[:,converged]
    governing_theta_index = theta_z[ba_res.governing_theta_idx]
    fig, width = _forces_vs_theta(theta_z[converged],
                                  [matrix_t, matrix_v, matrix_unity],
                                  [governing_theta_index, governing_theta_index, governing_theta_index],
                                  [r'Tension, $T$ (lbs)', r'Shear, $V$ (lbs)', 'Unity'],
                                  ['T', 'V', 'DCR'])
    return fig, width


def bracket_vs_theta(equipment_obj):
    matrix_n = equipment_obj.wall_bracket_forces[:, :, 0]
    matrix_p = equipment_obj.wall_bracket_forces[:, :, 1]
    matrix_z = equipment_obj.wall_bracket_forces[:, :, 2]
    fig, width = _forces_vs_theta(equipment_obj.theta_z,
                                  [matrix_n, matrix_p, matrix_z],
                                  [equipment_obj.governing_solutions['wall_bracket_tension'][
                                       'theta_z'],
                                   equipment_obj.governing_solutions['wall_bracket_shear'][
                                       'theta_z'],
                                   equipment_obj.governing_solutions['wall_bracket_shear'][
                                       'theta_z']],
                                  [r'Normal, $N$ (lbs)', r'In-Plane Shear, $V_p$ (lbs)',
                                   r'Vert. Shear, $V_z$ (lbs)'],
                                  ['N', 'V_p', 'V_z'])

    return fig, width


def wall_anchors_vs_theta(equipment_obj, _):
    anchors = [anchors for wall, anchors in equipment_obj.wall_anchors.items()]
    anchors += [backing.anchors_obj for backing in equipment_obj.wall_backing]
    all_anchor_forces = np.concatenate([a.anchor_forces for a in anchors if a is not None], axis=0)
    matrix_n = all_anchor_forces[:, :, 0]
    matrix_p = all_anchor_forces[:, :, 1]
    matrix_z = all_anchor_forces[:, :, 2]

    fig, width = _forces_vs_theta(equipment_obj.theta_z,
                                  [matrix_n, matrix_p, matrix_z],
                                  [equipment_obj.governing_solutions['wall_anchor_tension'][
                                       'theta_z'],
                                   equipment_obj.governing_solutions['wall_anchor_shear'][
                                       'theta_z'],
                                   equipment_obj.governing_solutions['wall_anchor_shear'][
                                       'theta_z']],
                                  [r'Normal, $N$ (lbs)', r'In-Plane Shear, $V_p$ (lbs)',
                                   r'Vert. Shear, $V_z$ (lbs)'],
                                  ['N', 'V_p', 'V_z'])

    return fig, width


def _forces_vs_theta(theta_z, matrix_list, theta_max_list, label_list, annotation_list):
    angles_degrees = np.rad2deg(theta_z)
    width = 3.5
    fig, axs = plt.subplots(len(matrix_list), 1, figsize=(width, 1.5 * len(matrix_list)))

    if len(matrix_list) == 1:
        axs = [axs]

    for idx, (matrix, theta_max, label, annotation) in enumerate(
            zip(matrix_list, theta_max_list, label_list, annotation_list)):
        ax = axs[idx]
        # Plot each line
        ax.plot(angles_degrees, matrix.T, '.-', alpha=0.5, color='b', linewidth=0.5, markersize=1)
        # for line, row in zip(lines, matrix):
        #     line.set_data(angles_degrees, row)

        # Calculate the envelope (max for each angle)
        envelope = np.max(matrix, axis=0)
        # Plot the envelope line
        ax.plot(angles_degrees, envelope, color='red', linewidth=1)

        # Find the max load and its corresponding angle
        max_load = envelope[theta_z == theta_max][0]
        max_angle = np.degrees(theta_max)

        # Add a data label for the maximum load
        ax.annotate(rf'${annotation} = {max_load:.0f}$ lbs at ${max_angle:.0f}^{{\circ}}$',
                    xy=(max_angle, max_load), xycoords='data', fontsize=8,
                    xytext=(max_angle + 45 if max_angle < 180 else max_angle - 45, max_load), textcoords='data',
                    arrowprops=dict(facecolor='black', shrink=0.05, width=3, headwidth=9, headlength=12),
                    horizontalalignment='left' if max_angle < 180 else 'right', verticalalignment='top')

        # Set the labels and title
        if idx == len(matrix_list) - 1:
            ax.set_xlabel(r'Angle, $\theta_z$ (degrees)')
        ax.set_ylabel(label)

        # Add grid for better readability
        ax.grid(True)
        ax.set_xlim(0, 360)
        ax.set_xticks(range(0, 361, 45))
        if np.min(matrix) > 0:
            ax.set_ylim(0, 1.1 * np.max(matrix))

    plt.tight_layout()
    return fig, width


def anchor_vs_theta_OLD(equipment_obj):
    angles = equipment_obj.theta_z
    matrix = equipment_obj.base_anchors.anchor_forces[:, :, 0]

    width = 6.5
    fig = plt.figure(figsize=(width, 4))
    ax = fig.add_subplot(211)
    # ax.set_position([0.1, 0.2, 0.8, 0.8])  # left, bottom, width, height
    # Convert angles from radians to degrees for plotting
    angles_degrees = np.rad2deg(angles)
    # Plot each line
    for row in matrix:
        ax.plot(angles_degrees, row, '.-', alpha=0.5, color='b')

    # Calculate the envelope (max for each angle)
    envelope = np.max(matrix, axis=0)
    # Plot the envelope line
    ax.plot(angles_degrees, envelope, color='red', linewidth=2)

    # Find the max load and its corresponding angle
    max_load = np.max(envelope)
    max_angle = math.degrees(equipment_obj.governing_solutions['base_anchor_tension']['theta_z'])

    # Add a data label for the maximum load
    ax.annotate(rf'$N_{{max}} = {max_load:.0f}$ lbs at ${max_angle:.0f}$ deg',
                xy=(max_angle, max_load), xycoords='data',
                xytext=(max_angle + 45 if max_angle < 180 else max_angle - 45, max_load), textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.05, width=3, headwidth=9, headlength=12),
                horizontalalignment='left' if max_angle < 180 else 'right', verticalalignment='top')

    # Set the labels and title
    # ax.set_xlabel(r'Angle, $\theta_z$ (degrees)')
    ax.set_ylabel('Reaction Loads, $N$ (lbs)')
    # ax.set_title('Load vs. Angle Plot')

    # Add grid for better readability
    ax.grid(True)
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 45))

    ''' Shear Plot'''
    matrix = (equipment_obj.base_anchors.anchor_forces[:, :, 1] ** 2 + equipment_obj.base_anchors.anchor_forces[:, :,
                                                                       2] ** 2) ** 0.5
    axv = fig.add_subplot(212)
    # axv.set_position([0.1, 0.2, 0.8, 0.8])  # left, bottom, width, height
    # Convert angles from radians to degrees for plotting
    angles_degrees = np.rad2deg(angles)
    # Plot each line
    for row in matrix:
        axv.plot(angles_degrees, row, '.-', alpha=0.5, color='b')

    # Calculate the envelope (max for each angle)
    envelope = np.max(matrix, axis=0)
    # Plot the envelope line
    axv.plot(angles_degrees, envelope, color='red', linewidth=2)

    # Find the max load and its corresponding angle
    max_load = np.max(envelope)
    max_angle = math.degrees(equipment_obj.governing_solutions['base_anchor_tension']['theta_z'])

    # Add a data label for the maximum load
    axv.annotate(rf'$V_{{max}} = {max_load:.0f}$ lbs at ${max_angle:.0f}$ deg',
                 xy=(max_angle, max_load), xycoords='data',
                 xytext=(max_angle + 45 if max_angle < 180 else max_angle - 45, max_load), textcoords='data',
                 arrowprops=dict(facecolor='black', shrink=0.05, width=3, headwidth=9, headlength=12),
                 horizontalalignment='left' if max_angle < 180 else 'right', verticalalignment='top')

    # Set the labels and title
    axv.set_xlabel(r'Angle, $\theta_z$ (degrees)')
    axv.set_ylabel('Reaction Loads, $V$ (lbs)')
    # ax.set_title('Load vs. Angle Plot')

    # Add grid for better readability
    axv.grid(True)
    axv.set_xlim(0, 360)
    axv.set_xticks(range(0, 361, 45))

    return fig, width


def anchor_tension_shear_interaction(_, anchors_obj: conc.ConcreteAnchors, results: conc.ConcreteAnchorResults):
    dcr_n = results.tension_unity_by_anchor[results.governing_anchor_idx,results.governing_theta_idx]
    dcr_v = results.shear_unity_by_anchor[results.governing_anchor_idx,results.governing_theta_idx]
    width = 2.5
    height = 2.5
    margin = 0.75

    wratio = (width - margin) / width
    hratio = (height - margin) / height
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    ax.set_position(((1 - wratio), (1 - hratio), wratio, hratio))  # left, bottom, width, height

    # Calculate corresponding y values
    x = np.linspace(0, 1, 500) ** (3 / 5)  # Raise to the power of 3/5 to distribute points more evenly
    y = (1 - x ** (5 / 3)) ** (3 / 5)

    # Plotting

    ax.plot(x, y, 'r-')
    ax.plot([0, dcr_v], [dcr_n, dcr_n], ':k')
    ax.plot([dcr_v, dcr_v], [0, dcr_n], ':k')
    ax.plot(dcr_v, dcr_n, 'ok')

    ax.set_xlabel(r'$V_u/\phi\phi_{seismic}V_n$', fontsize=10)
    ax.set_ylabel(r'$N_u/\phi\phi_{seismic}N_n$', fontsize=10)
    ax.grid(True)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major', labelsize=10)

    return fig, width


def anchor_basic(model_record: ModelRecord,
                 anchor_obj: conc.ConcreteAnchors,
                 anchor_results: conc.ConcreteAnchorResults):
    return _anchor_diagram_vtk(anchor_obj, anchor_results)


def anchor_N_breakout(model_record: ModelRecord,
                 anchor_obj: conc.ConcreteAnchors,
                 anchor_results: conc.ConcreteAnchorResults):
    return _anchor_diagram_vtk(anchor_obj, anchor_results, show_tension_breakout=True)


def anchor_N_pullout(model_record: ModelRecord,
                 anchor_obj: conc.ConcreteAnchors,
                 anchor_results: conc.ConcreteAnchorResults):
    pass


def anchor_V_breakout(model_record: ModelRecord,
                 anchor_obj: conc.ConcreteAnchors,
                 anchor_results: conc.ConcreteAnchorResults):
    return _anchor_diagram_vtk(anchor_obj, anchor_results, show_shear_breakout=True)


def _anchor_diagram_vtk(anchors: conc.ConcreteAnchors,
                        results: conc.ConcreteAnchorResults,
                        show_tension_breakout: bool = False,
                        show_shear_breakout: bool = False):
    a = anchors
    renderer = vtk.vtkRenderer()

    def plot_tension_and_shear_forces():
        max_arrow_length = 6  # a.hef_default
        max_force = max(results.governing_tension,results.governing_shear)
        max_force = max_force if max_force != 0 else 1

        f_normalize = results.forces[:,:,results.governing_theta_idx] / max_force * max_arrow_length  # Normalize to hef_default for the largest force

        for (x, y), (t, vx, vy) in zip(a.geo_props.xy_anchors, f_normalize):
            # Create arrow for tension force (Z-direction)
            arrow_source = vtk.vtkArrowSource()
            arrow_source.SetShaftRadius(0.05)  # Adjust the thickness as needed
            arrow_source.SetTipRadius(0.1)

            tension_transform = vtk.vtkTransform()
            tension_transform.Translate(x, y, 2.5)
            tension_transform.RotateY(-90)
            tension_transform.Scale(t, t, t)

            tension_transform_filter = vtk.vtkTransformPolyDataFilter()
            tension_transform_filter.SetTransform(tension_transform)
            tension_transform_filter.SetInputConnection(arrow_source.GetOutputPort())

            tension_mapper = vtk.vtkPolyDataMapper()
            tension_mapper.SetInputConnection(tension_transform_filter.GetOutputPort())

            tension_actor = vtk.vtkActor()
            tension_actor.SetMapper(tension_mapper)
            tension_actor.GetProperty().SetColor(1, 0, 0)  # Red color

            renderer.AddActor(tension_actor)

            # Create arrow for shear force in X-direction
            translation = [x + 1, y, 0] if vx > 0 else [x - 1, y, 0]
            shear_x_transform = vtk.vtkTransform()
            shear_x_transform.Translate(*translation)
            shear_x_transform.Scale(vx, vx, vx)
            shear_x_transform.RotateZ(0)  # Rotate to align with the X-axis

            shear_x_transform_filter = vtk.vtkTransformPolyDataFilter()
            shear_x_transform_filter.SetTransform(shear_x_transform)
            shear_x_transform_filter.SetInputConnection(arrow_source.GetOutputPort())

            shear_x_mapper = vtk.vtkPolyDataMapper()
            shear_x_mapper.SetInputConnection(shear_x_transform_filter.GetOutputPort())

            shear_x_actor = vtk.vtkActor()
            shear_x_actor.SetMapper(shear_x_mapper)
            shear_x_actor.GetProperty().SetColor(0, 0, 1)  # Blue color

            renderer.AddActor(shear_x_actor)

            # Create arrow for shear force in Y-direction
            translation = [x, y + 1, 0] if vy > 0 else [x, y - 1, 0]
            shear_y_transform = vtk.vtkTransform()
            shear_y_transform.Translate(*translation)
            shear_y_transform.Scale(vy, vy, vy)
            shear_y_transform.RotateZ(90)  # Default orientation for Y-axis

            shear_y_transform_filter = vtk.vtkTransformPolyDataFilter()
            shear_y_transform_filter.SetTransform(shear_y_transform)
            shear_y_transform_filter.SetInputConnection(arrow_source.GetOutputPort())

            shear_y_mapper = vtk.vtkPolyDataMapper()
            shear_y_mapper.SetInputConnection(shear_y_transform_filter.GetOutputPort())

            shear_y_actor = vtk.vtkActor()
            shear_y_actor.SetMapper(shear_y_mapper)
            shear_y_actor.GetProperty().SetColor(0, 0, 1)  # Blue color

            renderer.AddActor(shear_y_actor)

    def plot_tension_forces(tg: conc.TensionGroup, xy_group: NDArray):
        max_t = results.governing_tension
        max_t = max_t if max_t != 0 else 1
        max_arrow_length = a.anchor_props.hef_default  # Normalize to hef_default for the largest force
        t_normalize = results.forces[tg.anchor_indices,0,results.governing_theta_idx] / max_t * max_arrow_length

        for (x, y), t in zip(xy_group, t_normalize):
            # Create arrow for tension force (Z-direction)
            arrow_source = vtk.vtkArrowSource()
            arrow_source.SetShaftRadius(0.05)
            arrow_source.SetTipRadius(0.1)

            tension_transform = vtk.vtkTransform()
            tension_transform.Translate(x, y, 2.5)
            tension_transform.RotateY(-90)  # Orient along Z-axis
            tension_transform.Scale(t, t, t)

            tension_transform_filter = vtk.vtkTransformPolyDataFilter()
            tension_transform_filter.SetTransform(tension_transform)
            tension_transform_filter.SetInputConnection(arrow_source.GetOutputPort())

            tension_mapper = vtk.vtkPolyDataMapper()
            tension_mapper.SetInputConnection(tension_transform_filter.GetOutputPort())

            tension_actor = vtk.vtkActor()
            tension_actor.SetMapper(tension_mapper)
            tension_actor.GetProperty().SetColor(1, 0, 0)  # Red color

            renderer.AddActor(tension_actor)

    def plot_shear_forces(sg: conc.ShearGroup, xy_group: NDArray):
        force_component_idx = {conc.ShearDirLabel.XN: 1,
                               conc.ShearDirLabel.XP: 1,
                               conc.ShearDirLabel.YN: 2,
                               conc.ShearDirLabel.YP: 2}

        if sg.direction in (conc.ShearDirLabel.XN, conc.ShearDirLabel.XP):
            force_component_idx = 1
        else:
            force_component_idx = 2
        v = results.forces[sg.anchor_indices, force_component_idx, results.governing_theta_idx]
        max_v = v.max()
        max_v = max_v if max_v != 0 else 1

        max_v = np.max(np.abs(v))
        max_v = max_v if max_v != 0 else 1
        max_arrow_length = a.anchor_props.hef_default  # Normalize to hef_default for the largest force
        v_normalized = v / max_v * max_arrow_length

        for (x, y), vi in zip(xy_group, v_normalized):
            # Create arrow for shear force in X-direction
            arrow_source = vtk.vtkArrowSource()
            arrow_source.SetShaftRadius(0.05)
            arrow_source.SetTipRadius(0.1)

            translation = {conc.ShearDirLabel.XN: [x - 1, y, 0],
                           conc.ShearDirLabel.XP: [x + 1, y, 0],
                           conc.ShearDirLabel.YN: [x, y - 1, 0],
                           conc.ShearDirLabel.YP: [x, y + 1, 0]}

            shear_transform = vtk.vtkTransform()
            shear_transform.Translate(*translation[sg.direction])
            shear_transform.Scale(vi, vi, vi)

            rotations = {conc.ShearDirLabel.XN: 0,
                           conc.ShearDirLabel.XP: 0,
                           conc.ShearDirLabel.YN: 90,
                           conc.ShearDirLabel.YP: 90}

            shear_transform.RotateZ(rotations[sg.direction])  # Align with X-axis

            shear_transform_filter = vtk.vtkTransformPolyDataFilter()
            shear_transform_filter.SetTransform(shear_transform)
            shear_transform_filter.SetInputConnection(arrow_source.GetOutputPort())

            shear_mapper = vtk.vtkPolyDataMapper()
            shear_mapper.SetInputConnection(shear_transform_filter.GetOutputPort())

            shear_actor = vtk.vtkActor()
            shear_actor.SetMapper(shear_mapper)
            shear_actor.GetProperty().SetColor(0, 0, 1)  # Blue color

            renderer.AddActor(shear_actor)

    def plot_shear_xy(case='tension'):
        v = np.linalg.norm(a.max_group_forces[case][:, 1:3], axis=1)
        max_v = np.max(v)
        max_v = max_v if max_v != 0 else 1
        max_arrow_length = a.hef_default  # Normalize to hef_default for the largest force
        v_normalized = a.max_group_forces[case][:, 1:3] / max_v * max_arrow_length

        for (x, y), (vx, vy) in zip(a.xy_group, v_normalized):
            # Create arrow for shear force in XY-plane
            arrow_source = vtk.vtkArrowSource()
            arrow_source.SetShaftRadius(0.05)
            arrow_source.SetTipRadius(0.1)

            shear_xy_transform = vtk.vtkTransform()
            shear_xy_transform.Translate(x, y, 0)
            shear_xy_transform.Scale(np.sqrt(vx ** 2 + vy ** 2), np.sqrt(vx ** 2 + vy ** 2), 1)
            shear_xy_transform.RotateZ(np.degrees(np.arctan2(vy, vx)))  # Rotate to align with the XY-plane direction

            shear_xy_transform_filter = vtk.vtkTransformPolyDataFilter()
            shear_xy_transform_filter.SetTransform(shear_xy_transform)
            shear_xy_transform_filter.SetInputConnection(arrow_source.GetOutputPort())

            shear_xy_mapper = vtk.vtkPolyDataMapper()
            shear_xy_mapper.SetInputConnection(shear_xy_transform_filter.GetOutputPort())

            shear_xy_actor = vtk.vtkActor()
            shear_xy_actor.SetMapper(shear_xy_mapper)
            shear_xy_actor.GetProperty().SetColor(0, 0, 1)  # Blue color

            renderer.AddActor(shear_xy_actor)

    def plot_breakout_cone(renderer,
                           x1b, x2b, x3b, x4b,
                           x1t, x2t, x3t, x4t,
                           y1b, y2b, y3b, y4b,
                           y1t, y2t, y3t, y4t,
                           cone_color):
        cone_vertices = np.array([[x1b, y1b, z1b],  # Point 1 at the bottom
                                  [x2b, y2b, z2b],  # Point 2 at the bottom
                                  [x3b, y3b, z3b],  # Point 3 at the bottom
                                  [x4b, y4b, z4b],  # Point 4 at the bottom
                                  [x1t, y1t, z1t],  # Point 1 at the top
                                  [x2t, y2t, z2t],  # Point 2 at the top
                                  [x3t, y3t, z3t],  # Point 3 at the top
                                  [x4t, y4t, z4t]])  # Point 4 at the top

        points = vtk.vtkPoints()
        for v in cone_vertices:
            points.InsertNextPoint(v)

        faces_array = np.array([[0, 1, 2, 3],  # Bottom face
                                [4, 5, 6, 7],  # Top face
                                [0, 1, 5, 4],  # Side faces
                                [1, 2, 6, 5],
                                [2, 3, 7, 6],
                                [3, 0, 4, 7]])

        faces = vtk.vtkCellArray()
        for face in faces_array:
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(4)
            for i in range(4):
                polygon.GetPointIds().SetId(i, face[i])
            faces.InsertNextCell(polygon)

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(faces)

        cone_mapper = vtk.vtkPolyDataMapper()
        cone_mapper.SetInputData(poly_data)
        cone_actor = vtk.vtkActor()
        cone_actor.SetMapper(cone_mapper)
        cone_actor.GetProperty().SetColor(*cone_color)  # Red color
        cone_actor.GetProperty().SetOpacity(0.5)  # Semi-transparent

        renderer.AddActor(cone_actor)

    def plot_concrete(renderer, xmin, xmax, ymin, ymax, zmin, zmax):
        vertices = np.array([[xmin, ymin, zmin],
                             [xmax, ymin, zmin],
                             [xmax, ymax, zmin],
                             [xmin, ymax, zmin],
                             [xmin, ymin, zmax],
                             [xmax, ymin, zmax],
                             [xmax, ymax, zmax],
                             [xmin, ymax, zmax]])

        points = vtk.vtkPoints()
        for v in vertices:
            points.InsertNextPoint(v)

        # Plot Box
        faces = vtk.vtkCellArray()
        for face in [[0, 1, 2, 3], [4, 5, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 5, 4], [2, 3, 7, 6]]:
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(4)
            for i in range(4):
                polygon.GetPointIds().SetId(i, face[i])
            faces.InsertNextCell(polygon)

        # Create a vtkPolyData object for the concrete slab
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(faces)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.35)
        actor.GetProperty().SetColor(0.8, 0.8, 0.7)

        # Create the wireframe for the slab's edges
        edges_faces = {
            (0, 1): [0, 4],
            (1, 2): [0, 3],
            (2, 3): [0, 5],
            (3, 0): [0, 2],
            (4, 5): [1, 4],
            (5, 6): [1, 3],
            (6, 7): [1, 5],
            (7, 4): [1, 2],
            (0, 4): [2, 4],
            (1, 5): [3, 4],
            (2, 6): [3, 5],
            (3, 7): [2, 5],
        }

        wireframe_edges = vtk.vtkCellArray()
        for edge, faces in edges_faces.items():
            # if all([a.anchor_props.cax_neg < 3 * a.concrete_props.t_slab if face == 2 else
            #         a.cax_pos < 3 * a.t_slab if face == 3 else
            #         a.cay_neg < 3 * a.t_slab if face == 4 else
            #         a.cay_pos < 3 * a.t_slab if face == 5 else True for face in faces]):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, edge[0])
            line.GetPointIds().SetId(1, edge[1])
            wireframe_edges.InsertNextCell(line)

        wireframe_polydata = vtk.vtkPolyData()
        wireframe_polydata.SetPoints(points)
        wireframe_polydata.SetLines(wireframe_edges)

        wireframe_mapper = vtk.vtkPolyDataMapper()
        wireframe_mapper.SetInputData(wireframe_polydata)
        wireframe_actor = vtk.vtkActor()
        wireframe_actor.SetMapper(wireframe_mapper)
        wireframe_actor.GetProperty().SetColor(0, 0, 0)  # Black wireframe

        renderer.AddActor(actor)
        renderer.AddActor(wireframe_actor)

    # Extract geometry for plotting
    if show_shear_breakout and show_tension_breakout:
        # possible added option to show both cones concurrently
        raise NotImplementedError
    elif show_shear_breakout:
        sg = results.shear_groups[results.governing_shear_group]
        calc = results.shear_breakout_calcs[results.governing_shear_group]
        xy_group = a.geo_props.xy_anchors[sg.anchor_indices,:]
        sg_dir = sg.direction
        cone_color = (0, 0, 1)

        # Determine governing breakout cone coordinates
        # Order of points
        #      (4) - (3)
        # Y     |     |
        # |_ X (1) - (2)
        if sg_dir == conc.ShearDirLabel.XP:  # Positive X
            x1b = x4b = x1t = x4t = xy_group[:, 0].min()
            x2b = x3b = x2t = x3t = min(x1b, x4b) + sg.ca1

            y1b = y1t = xy_group[xy_group[:, 0] == x1b, 1].min()
            y2b = y2t = sg.ca2n_ordinate
            y4b = y4t = xy_group[xy_group[:, 0] == x1b, 1].max()
            y3b = y3t = sg.ca2p_ordinate

            z1b = z4b = z1t = z2t = z3t = z4t = 0
            z2b = z3b = -calc.ha_eff

        elif sg_dir == conc.ShearDirLabel.XN:  # Negative X Full
            x2b = x3b = x2t = x3t = xy_group[:, 0].max()
            x1b = x4b = x1t = x4t = max(x2b,x3b) - sg.ca1

            y2b = y2t = xy_group[xy_group[:, 0] == x2b, 1].min()
            y1b = y1t = sg.ca2n_ordinate
            y3b = y3t = xy_group[xy_group[:, 0] == x2b, 1].max()
            y4b = y4t = sg.ca2p_ordinate

            z2b = z3b = z1t = z2t = z3t = z4t = 0
            z1b = z4b = -calc.ha_eff

        elif sg_dir == conc.ShearDirLabel.YP:  # Positive Y Full
            y1b = y2b = y1t = y2t = xy_group[:, 1].min()
            y3b = y4b = y3t = y4t = min(y1b, y2b) + sg.ca1

            x1b = x1t = xy_group[xy_group[:, 1] == y1b, 0].min()
            x4b = x4t = sg.ca2n_ordinate
            x2b = x2t = xy_group[xy_group[:, 1] == y1b, 0].max()
            x3b = x3t = sg.ca2p_ordinate

            z1b = z2b = z1t = z2t = z3t = z4t = 0
            z3b = z4b = -calc.ha_eff

        else: #sg_dir == conc.ShearDirLabel.YN:  # Positive Y Full
            y3b = y4b = y3t = y4t = xy_group[:, 1].max()
            y1b = y2b = y1t = y2t = max(y3b, y4b) - sg.ca1

            x4b = x4t = xy_group[xy_group[:, 1] == y3b, 0].min()
            x1b = x1t = sg.ca2n_ordinate
            x3b = x3t = xy_group[xy_group[:, 1] == y3b, 0].max()
            x2b = x2t = sg.ca2p_ordinate

            z3b = z4b = z1t = z2t = z3t = z4t = 0
            z1b = z2b = -calc.ha_eff

        # Create vertices for the concrete slab box
        xmin = min(max(xy_group[:, 0].min() - 1.5 * a.concrete_props.t_slab, a.geo_props.edge_ordinates.XN), x1t, x4t)
        xmax = max(min(xy_group[:, 0].max() + 1.5 * a.concrete_props.t_slab, a.geo_props.edge_ordinates.XP), x2t, x3t)
        ymin = min(max(xy_group[:, 1].min() - 1.5 * a.concrete_props.t_slab, a.geo_props.edge_ordinates.YN), y1t, y2t)
        ymax = max(min(xy_group[:, 1].max() + 1.5 * a.concrete_props.t_slab, a.geo_props.edge_ordinates.YP), y3t, y4t)
        zmax = 0
        zmin = -a.concrete_props.t_slab

        plot_concrete(renderer, xmin, xmax, ymin, ymax, zmin, zmax)
        plot_breakout_cone(renderer,
                           x1b, x2b, x3b, x4b,
                           x1t, x2t, x3t, x4t,
                           y1b, y2b, y3b, y4b,
                           y1t, y2t, y3t, y4t,
                           cone_color)
        plot_shear_forces(sg, xy_group)

    elif show_tension_breakout:
        tg = results.tension_groups[results.governing_tension_group]
        calc = results.tension_breakout_calcs[results.governing_tension_group]
        xy_group = a.geo_props.xy_anchors[tg.anchor_indices, :]
        cone_color = (1, 0, 0)

        x1b = x4b = xy_group[:, 0].min()
        x2b = x3b = xy_group[:, 0].max()
        y1b = y2b = xy_group[:, 1].min()
        y3b = y4b = xy_group[:, 1].max()

        x1t = x4t = xy_group[:, 0].min() - min([tg.cax_neg, 1.5 * calc.hef])
        x2t = x3t = x1t + calc.bxN
        y1t = y2t = xy_group[:, 1].min() - min([tg.cay_neg, 1.5 * calc.hef])
        y3t = y4t = y1t + calc.byN

        z1b = z2b = z3b = z4b = -calc.hef
        z1t = z2t = z3t = z4t = 0  # Assuming zmax is the top surface of the slab

        # Create vertices for the concrete slab box
        slab_extension = 1.5*a.concrete_props.t_slab
        xmin = min(xy_group[:, 0].min() - min(tg.cax_neg, slab_extension), x1t, x2t, x3t, x4t)
        xmax = max(xy_group[:, 0].max() + min(tg.cax_pos, slab_extension), x1t, x2t, x3t, x4t)
        ymin = min(xy_group[:, 1].min() - min(tg.cay_neg, slab_extension), y1t, y2t, y3t, y4t)
        ymax = max(xy_group[:, 1].max() + min(tg.cay_pos, slab_extension), y1t, y2t, y3t, y4t)
        zmax = 0
        zmin = -a.concrete_props.t_slab

        plot_concrete(renderer,xmin, xmax, ymin, ymax, zmin, zmax)
        plot_breakout_cone(renderer,
                           x1b, x2b, x3b, x4b,
                           x1t, x2t, x3t, x4t,
                           y1b, y2b, y3b, y4b,
                           y1t, y2t, y3t, y4t,
                           cone_color)
        plot_tension_forces(tg, xy_group)

    else:  # Plot without Breakout Cones
        xy_group = a.geo_props.xy_anchors
        slab_extension = 1.5 * a.concrete_props.t_slab

        # Create vertices for the concrete slab box
        xmin = xy_group[:,0].min() - min(slab_extension, a.geo_props.cx_neg)
        xmax = xy_group[:, 0].max() + max(slab_extension, a.geo_props.cx_pos)
        ymin = xy_group[:, 1].min() - min(slab_extension, a.geo_props.cy_neg)
        ymax = xy_group[:, 1].max() + max(slab_extension, a.geo_props.cy_pos)

        zmax = 0
        zmin = -a.concrete_props.t_slab

        plot_concrete(renderer, xmin, xmax, ymin, ymax, zmin, zmax)
        plot_tension_and_shear_forces()

    renderer.SetBackground(1, 1, 1)

    # Draw Anchors
    for x, y in xy_group:
        cylinder = vtk.vtkCylinderSource()
        cylinder.SetRadius(a.anchor_props.da / 2)
        cylinder.SetHeight(a.anchor_props.hef_default + 2)
        cylinder.SetResolution(50)

        # Apply transformation to orient the cylinder vertically
        transform = vtk.vtkTransform()
        transform.RotateX(90)  # Rotate around the X-axis to make it vertical

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transform)
        transform_filter.SetInputConnection(cylinder.GetOutputPort())
        transform_filter.Update()

        cylinder_actor = vtk.vtkActor()
        cylinder_mapper = vtk.vtkPolyDataMapper()
        cylinder_mapper.SetInputConnection(transform_filter.GetOutputPort())
        cylinder_actor.SetMapper(cylinder_mapper)
        cylinder_actor.SetPosition(x, y, -a.anchor_props.hef_default / 2)
        cylinder_actor.GetProperty().SetColor(0, 0, 0)
        renderer.AddActor(cylinder_actor)

    # Setup camera view
    phi = 45  # Elevation angle
    azimuth = 260  # Azimuth angle

    # Compute bounds and related parameters
    bounds = np.array(renderer.ComputeVisiblePropBounds())  # [-x, +x, -y, +y, -z, +z]
    r = 1.1 * max(bounds[1::2] - bounds[::2]) * 3.5
    cen = (bounds[1::2] + bounds[::2]) * 0.5
    cen[2] = bounds[4] - (bounds[5] - bounds[4])
    # cen[2] = -a.t_slab

    # Convert azimuth and elevation angles to radians for trigonometric functions
    azimuth_rad = math.radians(azimuth)
    phi_rad = math.radians(phi)

    # Calculate camera position based on spherical coordinates
    camera_x = cen[0] + r * math.cos(phi_rad) * math.cos(azimuth_rad)
    camera_y = cen[1] + r * math.cos(phi_rad) * math.sin(azimuth_rad)
    camera_z = cen[2] + r * math.sin(phi_rad)

    # Set up the camera with the calculated position
    camera = vtk.vtkCamera()
    camera.SetPosition(camera_x, camera_y, camera_z)
    camera.SetFocalPoint(cen[0], cen[1], cen[2])
    camera.SetViewUp(0, 0, 1)
    camera.SetViewAngle(10)
    renderer.SetActiveCamera(camera)
    renderer.ResetCamera()  # Adjusts the camera to include all visible elements
    render_window = vtk.vtkRenderWindow()
    figw = 2.5  # inches
    figh = 2.5  # inches
    dpi = 600  # dots per inch
    render_window.SetSize(int(figw * dpi), int(figh * dpi))
    render_window.OffScreenRenderingOn()
    render_window.AddRenderer(renderer)

    # Render and capture the image
    render_window.Render()
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    return window_to_image_filter, figw


def anchor_spacing_criteria(_, anchors_obj: conc.ConcreteAnchors, anchor_results: conc.ConcreteAnchorResults):
    c1 = anchors_obj.anchor_props.c1
    s1 = anchors_obj.anchor_props.s1
    c2 = anchors_obj.anchor_props.c2
    s2 = anchors_obj.anchor_props.s2
    # c_min = anchors_obj.c_min
    ca = anchors_obj.geo_props.c_min  # minimum provided edge distance
    smin = anchors_obj.geo_props.s_min

    width = 2.25
    height = 2.25
    margin = 0.75

    wratio = (width - margin) / width
    hratio = (height - margin) / height
    # Plotting
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    ax.set_position(((1 - wratio), (1 - hratio), wratio, hratio))  # left, bottom, width, height

    ax_lim = max(2 * c2, 2 * s1)
    polygon = patches.Polygon(((0, 0), (ax_lim, 0), (ax_lim, s2), (c2, s2), (c1, s1), (c1, ax_lim), (0, ax_lim)),
                              closed=True, facecolor='gray', alpha=0.5)
    ax.add_patch(polygon)
    ax.plot((ax_lim, c2, c1, c1), (s2, s2, s1, ax_lim), '-k', linewidth=1)
    if (ca < ax_lim) and (smin < ax_lim):
        ax.plot((0, ca), (smin, smin), ':k')
        ax.plot((ca, ca), (0, smin), ':k')
        ax.plot(ca, smin, 'ok')
    else:
        x = min(ca, ax_lim)
        y = min(smin, ax_lim)
        L_arrow = 0.5 * min(c2, s1)
        L_point = ((ca - x) ** 2 + (smin - y) ** 2) ** 0.5
        if np.isclose(L_point, 0):
            dx = 0
            dy = 0
        elif np.isinf(L_point):
            dx = L_arrow
            dy = 0
        else:
            dx = (ca - x) * L_arrow / L_point
            dy = (smin - y) * L_arrow / L_point
        ax.arrow(x - dx, y - dy, dx, dy, width=.1, fc='black', ec='black', length_includes_head=True, zorder=3)

    ax.annotate("OK", (c2, s1), xytext=(c2, s1), horizontalalignment='left',
                verticalalignment='bottom')
    # ax.annotate(r"{\color{red}\textbf{\textsf{NG}}}", (c1, s2), xytext=(c1, s2), horizontalalignment='right',
    #             verticalalignment='top')
    ax.annotate("NG", (c1, s2), xytext=(c1, s2), horizontalalignment='right',
                verticalalignment='top')

    ax.set_xlabel('Edge Distance, $c$ (in)', fontsize=10)
    ax.set_ylabel('Anchor Spacing, $s$ (in)', fontsize=10)
    ax.set_xlim(0, ax_lim)
    ax.set_ylim(0, ax_lim)
    ax.grid(True)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major', labelsize=10)

    return fig, width


def sms_hardware_attachment(
        _,
        cxn_obj:cxn.FastenerConnection,
        cxn_res: cxn.ElasticBoltGroupResults ,
        sms_obj:sms.SMSAnchors,
        sms_res: sms.SMSResults,
        __):

    w = cxn_obj.bolt_group_props.w
    h = cxn_obj.bolt_group_props.h
    anchor_xy = cxn_obj.bolt_group_props.xy_anchors
    theta_idx = sms_res.governing_theta_idx
    anchor_forces = cxn_res.anchor_forces_loc[:,:,theta_idx]
    N = float(cxn_res.centroid_forces_loc[0, theta_idx])
    Vx = float(cxn_res.centroid_forces_loc[1, theta_idx])
    Vy = float(cxn_res.centroid_forces_loc[2, theta_idx])
    Mx = float(cxn_res.centroid_forces_loc[3, theta_idx])
    My = float(cxn_res.centroid_forces_loc[4, theta_idx])
    T = float(cxn_res.centroid_forces_loc[5, theta_idx])

    width = 3
    height = 2.25
    margin = 0

    wratio = (width - margin) / width
    hratio = (height - margin) / height
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(121)
    ax.set_position(((1 - wratio), (1 - hratio), wratio, hratio))  # left, bottom, width, height

    # Plot Wall Backing Outline
    rectangle = patches.Rectangle((-w / 2, -h / 2), w, h,
                                  linewidth=1, edgecolor='gray', facecolor='none', alpha=0.5, zorder=1)
    ax.add_patch(rectangle)

    text_offset = 0.5

    # Plot anchor points
    for i, (xy, force) in enumerate(zip(anchor_xy, anchor_forces)):
        if force[0] < 0:
            ax.plot(xy[0], xy[1], 'rx', markersize=3, zorder=3)  # Red "X"
        else:
            ax.plot(xy[0], xy[1], 'ro', markersize=3, zorder=3)  # Red dot
        ax.text(xy[0] + text_offset, xy[1] + text_offset, f'{i + 1:.0f}', color='red', fontsize=8, ha='left',
                va='center',
                bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.1', linewidth=0.5), zorder=1)

    # Plot arrows for anchor forces
    # max_length = 0.5 * min(w,
    #                        h,
    #                        anchor_xy[:, 0].max() - anchor_xy[:, 0].min(),
    #                        anchor_xy[:, 1].max() - anchor_xy[:, 1].min())  # Maximum arrow length for normalization

    max_length = 0.2 * max(w, h)

    anchor_resultants = np.linalg.norm(anchor_forces[:, 1:], axis=1)
    max_anchor = np.max(anchor_resultants, axis=0)

    max_resultant = max(max_anchor, abs(Vx), abs(Vy))
    max_moment = max(abs(Mx), abs(My))
    sf_moment = 0 if np.isclose(max_moment, 0) else max_length / max_moment
    sf = 0 if np.isclose(max_resultant, 0) else max_length / max_resultant

    # SMS Forces
    for xy, force in zip(anchor_xy, anchor_forces):
        if not np.isclose(np.linalg.norm(force[1:]), 0):
            ax.arrow(xy[0], xy[1], -force[1] * sf, -force[2] * sf, head_width=0.15, color='red', linewidth=2, zorder=3)

    # Vx
    if not np.isclose(Vx, 0):
        if Vx > 0:
            ax.arrow(0, 0, Vx * sf, 0, head_width=0.15, color='blue', linewidth=2, zorder=4)
        else:
            ax.arrow(abs(Vx*sf)+0.5*text_offset, 0, Vx * sf, 0, head_width=0.15, color='blue', linewidth=2, zorder=4)
        ax.text(text_offset, text_offset, f'$V_x$={Vx :.0f} lbs', color='blue',
                fontsize=8,
                ha='left', va='center', zorder=1)

    # Vy
    if not np.isclose(Vy, 0):
        if Vy  > 0:
            ax.arrow(0, 0, 0, Vy * sf, head_width=0.15, color='blue', linewidth=2, zorder=4)
        else:
            ax.arrow(0, abs(Vy*sf)+0.5*text_offset, 0, Vy * sf, head_width=0.15, color='blue', linewidth=2, zorder=4)
        ax.text(text_offset,
                3 * text_offset,
                f'$V_y$={Vy :.0f} lbs', color='blue', fontsize=8,
                ha='left', va='center', zorder=1)

    # N
    if not np.isclose(N, 0):
        if N < 0:
            ax.plot(0, 0, 'bx', markersize=4, zorder=4)  # Red "X"
        else:
            ax.plot(0, 0, 'bo', markersize=4, zorder=4)  # Red dot
        ax.text(text_offset, -text_offset, f'$N$={N:.0f} lbs', color='blue', fontsize=8,
                ha='left', va='center', zorder=1)

    # T
    if not np.isclose(T, 0):
        if T < 0:
            ax.text(0, 0, r'$\circlearrowleft$', color='green', fontsize=20,
                    ha='center', va='center', zorder=1)
        else:
            ax.text(0, 0, r'$\circlearrowright$', color='green', fontsize=20,
                    ha='center', va='center', zorder=1)
        ax.text(-text_offset, -3*text_offset, f'$T$={T:.0f} in-lbs', color='green', fontsize=8,
                ha='right', va='center', zorder=1)

    # Mx
    if not np.isclose(Mx, 0):
        if Mx > 0:
            ax.arrow(-Mx * sf_moment-0.5*text_offset, 0, Mx * sf_moment, 0, head_width=0.15, color='green',
                     linewidth=2, zorder=3)
        else:
            ax.arrow(0, 0, Mx * sf_moment, 0, head_width=0.15, color='green',
                     linewidth=2, zorder=3)
        ax.text(- text_offset, text_offset, f'$M_x$={Mx :.0f} in-lbs', color='green',
                fontsize=8,
                ha='right', va='center', zorder=1)

    # My
    if not np.isclose(My, 0):
        if My > 0:
            ax.arrow(0,-My * sf_moment-0.5*text_offset, 0, My * sf_moment, head_width=0.15, color='green',
                     linewidth=2, zorder=3)
        else:
            ax.arrow(0, 0, 0, My * sf_moment, head_width=0.15, color='green',
                     linewidth=2, zorder=3)
        ax.text(-text_offset, -text_offset,f'$M_y$={My :.0f} in-lbs',
                color='green',
                fontsize=8,
                ha='right', va='center', zorder=1)

    # My
    # Create legend handles for custom legend entries
    # anchor_legend = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=6,
    #                               label='Anchor Points')
    # bracket_legend = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=6,
    #                                label='Bracket Loads')
    # backing_legend = mlines.Line2D([], [], color='gray', linestyle='-', linewidth=1, label='Backing Element')
    #
    # # Add legend
    # ax.legend(handles=[anchor_legend, bracket_legend, backing_legend], frameon=False)

    # Formatting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if h > 2 * w:
        ax.spines['bottom'].set_visible(False)
    else:
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['bottom'].set_zorder(0)
    if w > 2 * h:
        ax.spines['left'].set_visible(False)
    else:
        ax.spines['left'].set_position(('data', 0))
        ax.spines['left'].set_zorder(0)
    datalim = max(np.ceil(1.1*w), np.ceil(1.1*h))
    ax.set_xlim(-datalim / 2, datalim / 2)
    ax.set_ylim(-datalim / 2, datalim / 2)
    ax.set_aspect('equal')
    return fig, width


def wall_backing(_, backing_obj):
    w = backing_obj.w
    h = backing_obj.h
    anchor_xy = backing_obj.pz_anchors
    bracket_xy = backing_obj.xy_brackets # - [backing_obj.x_bar, backing_obj.y_bar]
    bracket_forces = backing_obj.bracket_forces
    anchor_forces = backing_obj.anchor_forces

    width = 3
    height = 2.25
    margin = 0

    wratio = (width - margin) / width
    hratio = (height - margin) / height
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(121)
    ax.set_position(((1 - wratio), (1 - hratio), wratio, hratio))  # left, bottom, width, height

    # Plot Wall Backing Outline
    rectangle = patches.Rectangle((-w / 2, -h / 2), w, h, linewidth=1, edgecolor='gray', facecolor='none',
                                  alpha=0.5, zorder=1)
    ax.add_patch(rectangle)

    text_offset = 3

    # Plot bracket points
    for i, (xy, force) in enumerate(zip(bracket_xy, bracket_forces)):
        if force[0] < 0:
            ax.plot(xy[0], xy[1], 'bx', markersize=6, zorder=2)
        else:
            ax.plot(xy[0], xy[1], 'bo', markersize=6, zorder=2)
        # texts.append(ax.text(xy[0], xy[1], f'N: {force[0]:.0f}', color='blue', fontsize=8, ha='right'))

        ax.text(xy[0], xy[1] - text_offset, f'{i + 1:.0f}', color='blue', fontsize=8, ha='left', va='center',
                bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.1', linewidth=0.5), zorder=1)

    # Plot anchor points
    for i, (xy, force) in enumerate(zip(anchor_xy, anchor_forces)):
        if force[0] < 0:
            ax.plot(xy[0], xy[1], 'rx', markersize=3, zorder=3)  # Red "X"
        else:
            ax.plot(xy[0], xy[1], 'ro', markersize=3, zorder=3)  # Red dot
        ax.text(xy[0], xy[1] + text_offset, f'{i + 1:.0f}', color='red', fontsize=8, ha='left', va='center',
                bbox=dict(facecolor='w', edgecolor='black', boxstyle='round,pad=0.1', linewidth=0.5), zorder=1)

    # Plot arrows for bracket and anchor forces
    max_length = 0.25 * max(w, h)  # Maximum arrow length for normalization
    bracket_resultants = np.linalg.norm(bracket_forces[:, 1:], axis=1)
    anchor_resultants = np.linalg.norm(anchor_forces[:, 1:], axis=1)
    max_resultant = np.max(np.concatenate((bracket_resultants, anchor_resultants), axis=0), axis=0)
    sf = max_length / max_resultant
    for xy, force, resultant in zip(bracket_xy, bracket_forces, bracket_resultants):
        ax.arrow(xy[0], xy[1], force[1] * sf, force[2] * sf, head_width=0.15, color='blue', linewidth=2, zorder=2)

    for xy, force, resultant in zip(anchor_xy, anchor_forces, anchor_resultants):
        ax.arrow(xy[0], xy[1], -force[1] * sf, -force[2] * sf, head_width=0.15, color='red', linewidth=2, zorder=3)

    # Create legend handles for custom legend entries
    anchor_legend = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=6,
                                  label='Anchor Points')
    bracket_legend = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=6,
                                   label='Bracket Loads')
    backing_legend = mlines.Line2D([], [], color='gray', linestyle='-', linewidth=1, label='Backing Element')

    # Add legend
    ax.legend(handles=[anchor_legend, bracket_legend, backing_legend], frameon=False)

    # Formatting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if h > 2 * w:
        ax.spines['bottom'].set_visible(False)
    else:
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['bottom'].set_zorder(0)
    if w > 2 * h:
        ax.spines['left'].set_visible(False)
    else:
        ax.spines['left'].set_position(('data', 0))
        ax.spines['left'].set_zorder(0)
    datalim = max(np.ceil(1.1*w), np.ceil(1.1*h))
    ax.set_xlim(-datalim / 2, datalim / 2)
    ax.set_ylim(-datalim / 2, datalim / 2)
    ax.set_aspect('equal')
    return fig, width


def _create_vtk_box(xyz_min, xyz_max,
                    type='wireframe',
                    opacity=1.0,
                    facecolor=(0.8, 0.8, 0.7)):
    xmin, ymin, zmin = xyz_min
    xmax, ymax, zmax = xyz_max
    vertices = np.array([[xmin, ymin, zmin],
                         [xmax, ymin, zmin],
                         [xmax, ymax, zmin],
                         [xmin, ymax, zmin],
                         [xmin, ymin, zmax],
                         [xmax, ymin, zmax],
                         [xmax, ymax, zmax],
                         [xmin, ymax, zmax]])

    points = vtk.vtkPoints()
    for v in vertices:
        points.InsertNextPoint(v)

    if type == 'filled':
        faces = vtk.vtkCellArray()
        for face in [[0, 1, 2, 3], [4, 5, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 5, 4], [2, 3, 7, 6]]:
            polygon = vtk.vtkPolygon()
            polygon.GetPointIds().SetNumberOfIds(4)
            for i in range(4):
                polygon.GetPointIds().SetId(i, face[i])
            faces.InsertNextCell(polygon)

        # Create a vtkPolyData object
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(faces)

        # Create a mapper and actor for the box
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetColor(*facecolor)

    if type == 'wireframe':
        edges = vtk.vtkCellArray()
        for edge in [[0, 1], [1, 2], [2, 3], [3, 0],  # bottom edges
                     [4, 5], [5, 6], [6, 7], [7, 4],  # top edges
                     [0, 4], [1, 5], [2, 6], [3, 7]]:  # vertical edges
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, edge[0])
            line.GetPointIds().SetId(1, edge[1])
            edges.InsertNextCell(line)

        wireframe_polydata = vtk.vtkPolyData()
        wireframe_polydata.SetPoints(points)
        wireframe_polydata.SetLines(edges)

        wireframe_mapper = vtk.vtkPolyDataMapper()
        wireframe_mapper.SetInputData(wireframe_polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(wireframe_mapper)
        actor.GetProperty().SetColor(0.0, 0.0, 0.0)  # Black wireframe
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetLineWidth(2)  # Increase line width
    return actor


def _create_vtk_cylinder(xyz, height, radius=0.25, rotate_x=0, rotate_y=0, rotate_z=0, color=(0, 0, 0)):
    cylinder = vtk.vtkCylinderSource()
    cylinder.SetRadius(radius)
    cylinder.SetHeight(height)
    cylinder.SetResolution(50)

    # Apply transformation to orient the cylinder vertically
    transform = vtk.vtkTransform()
    transform.RotateX(rotate_x)
    transform.RotateY(rotate_y)
    transform.RotateZ(rotate_z)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputConnection(cylinder.GetOutputPort())
    transform_filter.Update()

    cylinder_actor = vtk.vtkActor()
    cylinder_mapper = vtk.vtkPolyDataMapper()
    cylinder_mapper.SetInputConnection(transform_filter.GetOutputPort())
    cylinder_actor.SetMapper(cylinder_mapper)
    cylinder_actor.SetPosition(*xyz)
    cylinder_actor.GetProperty().SetColor(*color)

    return cylinder_actor


def _npz_to_xyz(npz, supporting_wall=None, normal_vector=None):
    if supporting_wall is not None:
        normal_vectors = {'X+': (-1, 0, 0),
                          'X-': (1, 0, 0),
                          'Y+': (0, -1, 0),
                          'Y-': (0, 1, 0)}

        nx, ny, nz = normal_vectors[supporting_wall]
    elif normal_vector is not None:
        nx, ny, nz = normal_vector
    else:
        raise Exception("Must provide either supporting wall or normal_vector")

    G = np.array([[nx, ny, 0],
                  [-ny, nx, 0],
                  [0, 0, 1]])
    return np.dot(G.T, npz)

def _xyz_to_npz(xyz, supporting_wall=None, normal_vector=None):
    if supporting_wall is not None:
        normal_vectors = {'X+': (-1, 0, 0),
                          'X-': (1, 0, 0),
                          'Y+': (0, -1, 0),
                          'Y-': (0, 1, 0)}

        nx, ny, nz = normal_vectors[supporting_wall]
    elif normal_vector is not None:
        nx, ny, nz = normal_vector
    else:
        raise Exception("Must provide either supporting wall or normal_vector")

    G = np.array([[nx, ny, 0],
                  [-ny, nx, 0],
                  [0, 0, 1]])
    return np.dot(G, xyz)

def _plot_wireframe_box(ax, vertices, color='blue', linewidth=2, linestyle='-'):
    for i in range(4):
        # Bottom square
        ax.plot([vertices[i][0], vertices[(i + 1) % 4][0]], [vertices[i][1], vertices[(i + 1) % 4][1]],
                [vertices[i][2], vertices[(i + 1) % 4][2]], color=color, linewidth=linewidth, linestyle=linestyle,
                solid_capstyle='round')
        # Top square
        ax.plot([vertices[i + 4][0], vertices[(i + 1) % 4 + 4][0]],
                [vertices[i + 4][1], vertices[(i + 1) % 4 + 4][1]],
                [vertices[i + 4][2], vertices[(i + 1) % 4 + 4][2]], color=color, linewidth=linewidth,
                linestyle=linestyle, solid_capstyle='round')
        # Vertical lines
        ax.plot([vertices[i][0], vertices[i + 4][0]], [vertices[i][1], vertices[i + 4][1]],
                [vertices[i][2], vertices[i + 4][2]], color=color, linewidth=linewidth, linestyle=linestyle,
                solid_capstyle='round')


def _plot_wireframe_box_collection(ax, vertices, color='blue', linewidth=2, linestyle='-', zorder=0):
    """Efficiently plots a wireframe box using Line3DCollection instead of multiple ax.plot() calls."""
    # todo: I think delete this function. Was an attempt to improve performance, but does not give visuallly good results.
    # Define the 12 edges of the box using vertex indices
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face edges
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face edges
        (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical connecting edges
    ]

    # Create an array of line segments
    line_segments = [[vertices[i], vertices[j]] for i, j in edges]

    # Create the collection and add it to the plot
    line_collection = Line3DCollection(line_segments, colors=color, linewidths=linewidth, linestyles=linestyle)
    line_collection.set_sort_zpos(zorder)
    ax.add_collection3d(line_collection)


def _get_coordiante_displacements_OLD(local_xyz, local_dofs, sf=100):
    """Non-vectorized solution"""
    scale_factor = sf
    coordinate_displacements = np.zeros_like(local_xyz)
    for i, (x, y, z) in enumerate(local_xyz):
        u = scale_factor * np.array([[1, 0, 0, 0, z, -y],
                                     [0, 1, 0, -z, 0, x],
                                     [0, 0, 1, y, -x, 0]]) @ local_dofs
        coordinate_displacements[i] = u
    return coordinate_displacements


def _get_coordinate_displacements(local_xyz, local_dofs, sf=100):
    """Vectorized computation of coordinate displacements."""
    scale_factor = sf

    # Ensure local_xyz is always (N, 3)
    local_xyz = np.atleast_2d(local_xyz)  # Converts (3,) to (1, 3) if needed

    # Extract X, Y, Z as separate arrays
    x, y, z = local_xyz[:, 0], local_xyz[:, 1], local_xyz[:, 2]

    # Create transformation matrices for all points at once
    transform_matrices = np.stack([
        np.column_stack([np.ones_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), z, -y]),
        np.column_stack([np.zeros_like(y), np.ones_like(y), np.zeros_like(y), -z, np.zeros_like(y), x]),
        np.column_stack([np.zeros_like(z), np.zeros_like(z), np.ones_like(z), y, -x, np.zeros_like(z)])
    ], axis=1)  # Shape: (num_points, 3, 6)

    # Perform batch matrix multiplication: (num_points, 3, 6) @ (6,)
    coordinate_displacements = scale_factor * np.einsum('ijk,k->ij', transform_matrices, local_dofs)

    return coordinate_displacements


def _get_scale_factor(model: EquipmentModel, sol):
    eprops = model.equipment_props
    bx, by, h = eprops.Bx, eprops.By, eprops.H
    target_disp = 0.05 * max(bx, by, h)
    box_vertices = np.array([
        [-bx / 2, -by / 2, 0],
        [bx / 2, -by / 2, 0],
        [bx / 2, by / 2, 0],
        [-bx / 2, by / 2, 0],
        [-bx / 2, -by / 2, h],
        [bx / 2, -by / 2, h],
        [bx / 2, by / 2, h],
        [-bx / 2, by / 2, h]
    ])
    u_list = [_get_coordinate_displacements(box_vertices, sol[0:6], sf=1)]
    u_list += [_get_coordinate_displacements(
                np.hstack((boundary - np.array((plate.props.x0, plate.props.y0)), np.zeros((boundary.shape[0], 1)))),
                plate.dof_props.C @ sol,
                sf=1)
             for plate in model.elements.base_plates
             for boundary in plate.props.bearing_boundaries]

    max_disp = max([np.max(np.abs(u)) for u in u_list])
    sf = 0 if np.isclose(max_disp, 0, 1e-10) else target_disp / max_disp
    return sf  # SF will cause maximum box displacement to be 5% of largest dimension


def _get_wall_coordinates(
        model):  # todo:[Wall Bracket Connection] update to use equip.wall_offset[supporting_wall] instead of bracket.wall_gap
    # Extract Wall/Edge-of-slab Coordinates
    wall_coordinate = {'X+': None, 'X-': None, 'Y+': None, 'Y-': None}
    unit_edge = {'X+': 0.5 * model.equipment_props.Bx,
                 'X-': -0.5 * model.equipment_props.Bx,
                 'Y+': 0.5 * model.equipment_props.By,
                 'Y-': -0.5 * model.equipment_props.By}

    for supporting_wall in ['X+', 'X-', 'Y+', 'Y-']:

        # backing_d = [backing.d for backing in model.wall_backing if backing.supporting_wall == supporting_wall]
        gap = getattr(model.wall_offsets,WallPositions(supporting_wall).name,None)
        if gap:
            wall_coordinate[supporting_wall] = unit_edge[supporting_wall] + np.sign(unit_edge[supporting_wall]) * gap
        else:
            wall_coordinate[supporting_wall] = unit_edge[supporting_wall] * 1.2
    return wall_coordinate


VTK_PLOTS = [equipment_3d_view_vtk,
             anchor_basic,
             anchor_N_breakout,
             anchor_N_pullout,
             anchor_V_breakout]
