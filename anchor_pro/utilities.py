import numpy as np
from shapely.geometry import Polygon
from collections import defaultdict, deque  # used for Utilities.bearing_area_in_compression

from numpy.typing import NDArray
from typing import List, Optional, Tuple
from anchor_pro.ap_types import (RESULTS_LIKE)

def transform_points(points, translation, rotation_angle, reflection_angle):
    rotation_matrix = Utilities.rotation_matrix(rotation_angle)
    reflection_matrix = Utilities.reflection_matrix(reflection_angle)
    transformed_points = []
    for point in points:
        point = np.dot(rotation_matrix, point)
        point = np.dot(reflection_matrix, point)
        transformed_point = point + translation
        transformed_points.append(transformed_point)
    return np.array(transformed_points)

def transform_vectors(xy_components, rotation_angle, reflection_angle):
    """ OBSOLETE: Tranlates the directionality of a vector based on 'reflection' and 'rotation' axes.
    Was used previously for slotted-holes defined by a unit vector direction. This functionality has been replaced
    with directional releases."""

    rotation_matrix = Utilities.rotation_matrix(rotation_angle)
    reflection_matrix = Utilities.reflection_matrix(reflection_angle)
    transformed_vecs = []
    for vec in xy_components:
        vec = np.dot(rotation_matrix, vec)
        vec = np.dot(reflection_matrix, vec)
        transformed_vecs.append(vec)
    return np.array(transformed_vecs)

def compute_point_displacements(
    points: NDArray,
    u: NDArray,
    *,
    x0: float = 0.0,
    y0: float = 0.0,
    z0: float = 0.0,
    keep_dims:bool = False
) -> NDArray:
    """
    Compute rigid-body displacements at given points for one or many DOF states.

    Parameters
    ----------
    points : (n, 3) array_like
    u : (6,) or (6, n_theta) array_like
        DOFs (dx, dy, dz, rx, ry, rz). If (6, n_theta), displacements are
        computed for each column of u and returned with θ as the last axis.
    x0, y0, z0 : float
        Reference origin for rotations.

    Returns
    -------
    disp : (n, 3) or (n, 3, n_theta) ndarray
        Displacements at each point. If `u` is (6,), shape is (n, 3).
        If `u` is (6, n_theta), shape is (n, 3, n_theta) with θ last.
    """

    # Points
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must be shape (n, 3)")

    # DOFs → ensure shape (6, n_theta)
    U = np.asarray(u, dtype=float)
    if U.ndim == 1:
        if U.shape[0] != 6:
            raise ValueError("u must have length 6 when 1-D")
        U = U[:, None]  # (6, 1)
    elif U.ndim == 2:
        if U.shape[0] != 6:
            raise ValueError("u must be shape (6, n_theta) when 2-D")
    else:
        raise ValueError("u must be 1-D (6,) or 2-D (6, n_theta)")

    n_theta = U.shape[1]

    # Relative position vectors r = p - p0
    d = P - np.array([x0, y0, z0], dtype=float)  # (n, 3)
    dx = d[:, 0][:, None]  # (n, 1)
    dy = d[:, 1][:, None]
    dz = d[:, 2][:, None]

    # Translations and rotations, broadcast over points
    ux = U[0][None, :]  # (1, n_theta)
    uy = U[1][None, :]
    uz = U[2][None, :]
    rx = U[3][None, :]
    ry = U[4][None, :]
    rz = U[5][None, :]

    # Displacement field: t + ω × r
    # Cross components (broadcast: (n,1) vs (1,n_theta) → (n, n_theta))
    disp_x = ux + (ry * dz - rz * dy)
    disp_y = uy + (rz * dx - rx * dz)
    disp_z = uz + (rx * dy - ry * dx)

    disp = np.stack([disp_x, disp_y, disp_z], axis=1)  # (n, 3, n_theta)

    # If single θ, return (n,3) to match legacy behavior
    return disp[:, :, 0] if (n_theta == 1) and not keep_dims else disp

def vertical_point_displacements(points, u, x0=0.0, y0=0.0):
    """ points: array of x,y,z distances relative to the origin (0,0)
        u = (dx, dy, dz, rx, ry, rz): element kinematic variables for rigid body rotation
        returns: array of vertical displacement values"""
    return u[2] + u[3] * (points[:, 1] - y0) - u[4] * (points[:, 0] - x0)

def bearing_area_in_compression(xy_points, u, x0=0.0, y0=0.0):
    """
    Accepts vertices representing the bearing element boundary, along with the displacement values at those points.
    Returns the xy_points of the compression regions of the bearing element as separate polygons.
    """
    if len(xy_points) == 0:
        return []

    if not np.array_equal(xy_points[0], xy_points[-1]):
        xy_points = np.vstack([xy_points, xy_points[0]])

    z_values = vertical_point_displacements(xy_points, u, x0=x0, y0=y0)
    if all(z_values >= 0):  # No compression region
        return []

    n = len(xy_points)
    compression_edges = []  # Edges connecting points below the xy-plane
    intersection_points = []  # Store intersection points with the xy-plane
    graph = defaultdict(list)  # Graph of compression points and connections

    # Step 1: Traverse the polygon and add intersections
    for i in range(n-1):
        p1, p2 = xy_points[i], xy_points[(i + 1) % n]
        z1, z2 = z_values[i], z_values[(i + 1) % n]

        if z1 <= 0 and z2 <= 0:  # Both points are below the xy-plane
            compression_edges.append((tuple(p1), tuple(p2)))
            graph[tuple(p1)].append(tuple(p2))
            graph[tuple(p2)].append(tuple(p1))
        elif z1 * z2 <= 0:  # Edge crosses the xy-plane (or one vertex is on the xy_plane
            # Calculate intersection point
            t = -z1 / (z2 - z1)
            x_new = p1[0] + t * (p2[0] - p1[0])
            y_new = p1[1] + t * (p2[1] - p1[1])
            intersection_point = (x_new, y_new)
            intersection_points.append(intersection_point)

            # Add edges from below-plane points to the intersection
            if z1 < 0:
                compression_edges.append((tuple(p1), intersection_point))
                graph[tuple(p1)].append(intersection_point)
                graph[intersection_point].append(tuple(p1))
            else:
                compression_edges.append((intersection_point, tuple(p2)))
                graph[intersection_point].append(tuple(p2))
                graph[tuple(p2)].append(intersection_point)

    # Step 2: Identify the extreme intersection point
    intersection_points = sorted(intersection_points, key=lambda p: (p[0], p[1]))

    # Step 3: Add edges between intersection points
    for i in range(0, len(intersection_points) - 1, 2):
        p1, p2 = intersection_points[i], intersection_points[i + 1]

        # Add to the graph
        graph[p1].append(p2)
        graph[p2].append(p1)

        # Add to the compression_edges list
        compression_edges.append((p1, p2))

    # Step 4: Use graph traversal to find connected components
    def bfs(node, visited):
        """Perform BFS to find all connected nodes."""
        component = []
        queue = deque([node])
        visited.add(node)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return component

    visited = set()
    compression_regions = []
    for node in graph:
        if node not in visited:
            # Get all connected nodes in this component
            component = bfs(node, visited)
            # Reconstruct a polygon from the component
            polygon = reconstruct_polygon(component, compression_edges)
            if len(polygon) > 2:
                compression_regions.append(np.array(polygon))

    return compression_regions

def polygon_properties(vertices, datum=np.array([0, 0])):
    vertices = vertices - datum
    poly = Polygon(vertices)
    # area = poly.area
    centroid = (poly.centroid.x, poly.centroid.y)

    # Compute Moments of Intertia, relative to Origin
    '''https://en.wikipedia.org/wiki/Second_moment_of_area'''
    x, y = vertices[:, 0], vertices[:, 1]

    x = np.concatenate((x, [x[0]]))
    y = np.concatenate((y, [y[0]]))
    # x = np.array([x for x, y in vertices] + [vertices[0][0]])
    # y = np.array([y for x, y in vertices] + [vertices[0][1]])

    signed_area = (np.sum(x[0:-1] * y[1:]) - np.sum(x[1:] * y[0:-1])) / 2
    if signed_area < 0:
        area = -signed_area
        x = x[::-1]
        y = y[::-1]
    else:
        area = signed_area
    common_factor = (x[0:-1] * y[1:] - x[1:] * y[0:-1])
    Ixx = (1 / 12) * np.sum(common_factor * (y[0:-1] ** 2 + y[0:-1] * y[1:] + y[1:] ** 2))
    Iyy = (1 / 12) * np.sum(common_factor * (x[0:-1] ** 2 + x[0:-1] * x[1:] + x[1:] ** 2))
    Ixy = (1 / 24) * np.sum(common_factor *
                            (x[0:-1] * y[1:] + 2 * x[0:-1] * y[0:-1] + 2 * x[1:] * y[1:] + x[1:] * y[0:-1]))

    return area, centroid, Ixx, Iyy, Ixy

def reconstruct_polygon(component, edges):
    """
    Reconstruct a closed polygon from a connected component of nodes.
    Ensures correct ordering of vertices based on edge connectivity.
    """
    processed = []
    visited = set()  # Track vertices that have already appeared in the "start" position

    for edge in edges:
        if edge[0] not in visited:
            # If the first vertex of the edge hasn't been a start vertex yet
            processed.append(edge)
            visited.add(edge[0])
        else:
            # Otherwise, reverse the edge to make the second vertex the start
            processed.append((edge[1], edge[0]))
            visited.add(edge[1])

    edges = processed

    edge_map = {edge[0]: edge[1] for edge in edges}

    polygon = [component[0]]  # Start with any point in the component

    while polygon[-1] in edge_map:
        next_point = edge_map[polygon[-1]]
        if next_point == polygon[0]:  # Close the polygon
            break
        polygon.append(next_point)
    return polygon

def get_anchor_spacing_matrix(xy_anchors: np.ndarray) -> np.ndarray:
    diffs = xy_anchors[:, None, :] - xy_anchors[None, :, :]  # (n, n, 2)
    spacing_matrix = np.linalg.norm(diffs, axis=-1)  # (n, n)
    return spacing_matrix

def get_governing_result(results_list: Optional[List[RESULTS_LIKE]]) -> Optional[Tuple[RESULTS_LIKE, int]]:
    """
    Return the governing result (object with the highest unity value)
    from a list of result-like objects. Returns None if the list is empty.
    """
    if not results_list:
        return None
    idx, governing = max(enumerate(results_list), key=lambda kv: kv[1].unity)
    return governing, idx

def compute_backing_xy_points(s_or_num_horiz, s_or_num_vert, L_horiz, L_vert, x_offset, y_offset,
                              place_by_horiz='Spacing', place_by_vert='Spacing', manual_x=None, manual_y=None):

    if manual_x is None:
        manual_x = []
    if manual_y is None:
        manual_y = []


    def get_step_and_number(place_by, spacing_or_number, L):
        if place_by == "Spacing":
            step = spacing_or_number
            if step == 0:
                num = 0
            else:
                num = np.floor(L / step + 1)
        else:
            num = spacing_or_number
            if num > 1:
                step = L / (num - 1)
            else:
                step = 0
        return step, int(num)

    def get_anchor_coordinates(step, num, offset):
        pts = []
        pt = 0
        if num == 0:
            return pts

        if num % 2 != 0:
            # For odd number of anchors, place one in the center
            pts.append(0 + offset)
            pt += step
        else:
            pt += step / 2

        while len(pts) < num:
            pts.append(pt + offset)
            pts.append(-pt + offset)
            pt += step
        return pts

    num_h = 0
    num_v = 0

    if place_by_horiz == 'Manual':
        x_pts = manual_x
        num_h = len(manual_x)
    else:
        step_h, num_h = get_step_and_number(place_by_horiz, s_or_num_horiz, L_horiz)
        x_pts = get_anchor_coordinates(step_h, num_h, x_offset)

    if place_by_vert == 'Manual':
        y_pts = manual_y
        num_v = len(manual_y)
    else:
        step_v, num_v = get_step_and_number(place_by_vert, s_or_num_vert, L_vert)
        y_pts = get_anchor_coordinates(step_v, num_v, y_offset)

    if place_by_horiz=='Manual' and place_by_vert=='Manual':
        return np.column_stack((manual_x,manual_y))
    elif num_h > 0 and num_v > 0:
        num_points = num_h * num_v
        x_array = np.zeros(num_points)
        y_array = np.zeros(num_points)

        i = 0
        for ix in range(num_h):
            for iy in range(num_v):
                x_array[i] = x_pts[ix]
                y_array[i] = y_pts[iy]
                i += 1
        return np.column_stack((x_array, y_array))
    else:
        return np.empty((0, 2))


class Utilities:
    def __init__(self):
        pass

    @staticmethod
    def rotation_matrix(angle):
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])

    @staticmethod
    def reflection_matrix(angle):
        if angle == "null":
            return np.array([[1, 0], [0, 1]])
        else:
            theta = np.radians(float(angle))
            return np.array([[np.cos(2 * theta), np.sin(2 * theta)], [np.sin(2 * theta), -np.cos(2 * theta)]])

    @staticmethod
    def transform_points(points, translation, rotation_angle, reflection_angle):
        rotation_matrix = Utilities.rotation_matrix(rotation_angle)
        reflection_matrix = Utilities.reflection_matrix(reflection_angle)
        transformed_points = []
        for point in points:
            point = np.dot(rotation_matrix, point)
            point = np.dot(reflection_matrix, point)
            transformed_point = point + translation
            transformed_points.append(transformed_point)
        return np.array(transformed_points)

    @staticmethod
    def transform_vectors(xy_components, rotation_angle, reflection_angle):
        """ OBSOLETE: Tranlates the directionality of a vector based on 'reflection' and 'rotation' axes.
        Was used previously for slotted-holes defined by a unit vector direction. This functionality has been replaced
        with directional releases."""

        rotation_matrix = Utilities.rotation_matrix(rotation_angle)
        reflection_matrix = Utilities.reflection_matrix(reflection_angle)
        transformed_vecs = []
        for vec in xy_components:
            vec = np.dot(rotation_matrix, vec)
            vec = np.dot(reflection_matrix, vec)
            transformed_vecs.append(vec)
        return np.array(transformed_vecs)

    @staticmethod
    def polygon_properties(vertices, datum=np.array([0, 0])):
        vertices = vertices - datum
        poly = Polygon(vertices)
        # area = poly.area
        centroid = (poly.centroid.x, poly.centroid.y)

        # Compute Moments of Intertia, relative to Origin
        '''https://en.wikipedia.org/wiki/Second_moment_of_area'''
        x, y = vertices[:, 0], vertices[:, 1]

        x = np.concatenate((x, [x[0]]))
        y = np.concatenate((y, [y[0]]))
        # x = np.array([x for x, y in vertices] + [vertices[0][0]])
        # y = np.array([y for x, y in vertices] + [vertices[0][1]])

        signed_area = (np.sum(x[0:-1] * y[1:]) - np.sum(x[1:] * y[0:-1])) / 2
        if signed_area < 0:
            area = -signed_area
            x = x[::-1]
            y = y[::-1]
        else:
            area = signed_area
        common_factor = (x[0:-1] * y[1:] - x[1:] * y[0:-1])
        Ixx = (1 / 12) * np.sum(common_factor * (y[0:-1] ** 2 + y[0:-1] * y[1:] + y[1:] ** 2))
        Iyy = (1 / 12) * np.sum(common_factor * (x[0:-1] ** 2 + x[0:-1] * x[1:] + x[1:] ** 2))
        Ixy = (1 / 24) * np.sum(common_factor *
                                (x[0:-1] * y[1:] + 2 * x[0:-1] * y[0:-1] + 2 * x[1:] * y[1:] + x[1:] * y[0:-1]))

        return area, centroid, Ixx, Iyy, Ixy

    @staticmethod
    def polygon_properties_NEW(vertices, datum=np.array([0, 0])):
        """ Compute area, centroid, and second moments of inertia of a polygon. """

        # Shift vertices relative to datum
        vertices = vertices - datum

        # Shapely computations
        poly = Polygon(vertices)
        area = poly.area
        centroid = (poly.centroid.x, poly.centroid.y)

        # Extract x and y coordinates efficiently
        x, y = vertices[:, 0], vertices[:, 1]

        # Append first point to close the polygon (faster than np.roll)
        x = np.concatenate((x, [x[0]]))
        y = np.concatenate((y, [y[0]]))

        # Compute signed area (shoelace formula)
        cross_product = x[:-1] * y[1:] - x[1:] * y[:-1]
        signed_area = 0.5 * np.sum(cross_product)

        # Reverse order only if needed
        if signed_area < 0:
            x, y = x[::-1], y[::-1]
            cross_product = -cross_product

        # Compute moments of inertia
        Ixx = (1 / 12) * np.sum(cross_product * (y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2))
        Iyy = (1 / 12) * np.sum(cross_product * (x[:-1] ** 2 + x[:-1] * x[1:] + x[1:] ** 2))
        Ixy = (1 / 24) * np.sum(
            cross_product * (x[:-1] * y[1:] + 2 * x[:-1] * y[:-1] + 2 * x[1:] * y[1:] + x[1:] * y[:-1]))

        return area, centroid, Ixx, Iyy, Ixy

    @staticmethod
    def effective_indenter_stiffness(area, modulus, poisson_ratio):
        E_eff = modulus / (1 - poisson_ratio ** 2)
        beta = 2 * E_eff / (np.pi * area) ** 0.5
        # beta = 2 * E_eff / 1  # TESTING: trying a "constant" beta to validate impact on speed

        return beta

    @staticmethod
    def vertical_point_displacements(points, u, x0=0.0, y0=0.0):
        """ points: array of x,y,z distances relative to the origin (0,0)
            u = (dx, dy, dz, rx, ry, rz): element kinematic variables for rigid body rotation
            returns: array of vertical displacement values"""
        return u[2] + u[3] * (points[:, 1] - y0) - u[4] * (points[:, 0] - x0)

    @staticmethod
    def compute_point_displacements_OLD(points, u, x0=0.0, y0=0.0, z0=0.0):
        """ points: array of x,y,z distances relative to the global origin (0,0)
            u = (dx, dy, dz, rx, ry, rz): local kinematic variables for rigid body rotation
            returns: array of vertical displacement values"""
        # Extract points
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Extract displacement components
        ux, uy, uz, rx, ry, rz = u

        # Compute displacements using the relationship d = Cx
        dx = ux + (z - z0) * ry - (y - y0) * rz
        dy = uy - (z - z0) * rx + (x - x0) * rz
        dz = uz - (x - x0) * ry + (y - y0) * rx

        # Stack the displacements into a single (n, 3) array
        displacements = np.stack((dx, dy, dz), axis=-1)

        return displacements

    @staticmethod
    def compute_point_displacements(points, u, x0=0.0, y0=0.0, z0=0.0):
        """Compute displacement values at given points for a rigid body transformation.

        Args:
            points: (n,3) array of x,y,z distances relative to the global origin.
            u: (dx, dy, dz, rx, ry, rz) - local kinematic variables for rigid body motion.
            x0, y0, z0: Reference origin for rotation (default is global origin).

        Returns:
            (n,3) NumPy array of computed displacements.
        """
        # Convert points to NumPy array if not already
        points = np.asarray(points)

        # Extract displacement and rotation components
        ux, uy, uz, rx, ry, rz = u

        # Compute relative coordinates
        x, y, z = (points - np.array([x0, y0, z0])).T

        # Compute displacements using NumPy broadcasting
        displacements = np.empty_like(points)
        displacements[:, 0] = ux + z * ry - y * rz
        displacements[:, 1] = uy - z * rx + x * rz
        displacements[:, 2] = uz - x * ry + y * rx

        return displacements

    @staticmethod
    def obsolete_bearing_area_in_compression(xy_points, u, x0=0.0, y0=0.0):
        """Accepts vertices representing the bearing element boundary, along with the displacement values at those points
        Returns the xy_points of the compression region of the bearing element"""
        # TODO: [QA] there seems to be a problem when dividing into multiple bearing. Consider "el" supports example

        z_values = Utilities.vertical_point_displacements(xy_points, u)
        if all(z_values >= 0):
            return []

        compression_boundary = []
        zero_points = []
        n = len(xy_points)

        for i in range(n):  # Iterate through points, considering consecutive pairs
            p1, p2 = xy_points[i], xy_points[(i + 1) % n]
            z1, z2 = z_values[i], z_values[(i + 1) % n]
            # Check if the line segment crosses z=0
            if z1 * z2 < 0:
                # Calculate the intersection point
                t = -z1 / (z2 - z1)
                x_new = p1[0] + t * (p2[0] - p1[0])
                y_new = p1[1] + t * (p2[1] - p1[1])
                intersection_point = [x_new, y_new]

                # Add the intersection point to the list if the first point is below z=0
                if z1 < 0:
                    compression_boundary.append(p1)
                    zero_points.append(False)
                    compression_boundary.append(intersection_point)
                    zero_points.append(True)
                else:
                    # If the first point is above z=0, start with the intersection point
                    compression_boundary.append(intersection_point)
                    zero_points.append(True)

            elif z1 <= 0:
                # If the segment doesn't cross z=0 and p1 is on or below the plane, add p1
                compression_boundary.append(p1)
                zero_points.append(False)

        # Identify whether the bearing region is divided into multiple compression zone polyogons
        boundaries = []
        compression_boundary = np.array(compression_boundary)
        # zero_points = Utilities.vertical_point_displacements(compression_boundary, d0, theta_x, theta_y) == 0
        if sum(zero_points) > 1:
            zero_points = np.append(zero_points, zero_points[0])
            poly_end_idx = np.where(np.logical_and(zero_points[:-1], zero_points[1:]))[0]
            n = len(compression_boundary)

            for i in range(len(poly_end_idx)):
                '''For DEBUGGING'''  # todo: remove
                import matplotlib.pyplot as plt
                plt.plot(xy_points[:, 0], xy_points[:, 1], '-k')
                for ii in range(len(xy_points)):
                    color = 'b' if z_values[ii] < 0 else 'r'
                    plt.plot(*xy_points[ii, :], '.', color=color)
                for (x, y), iszero in zip(compression_boundary, zero_points[:-1]):
                    if iszero:
                        plt.plot(x, y, 'ok')
                    else:
                        plt.plot(x, y, 'o', markeredgecolor='black', markerfacecolor='none', markersize=10)
                ''''''
                poly_start_idx = (poly_end_idx[i - 1] + 1) % n
                if poly_start_idx <= poly_end_idx[i]:
                    poly_idx = list(range(poly_start_idx, poly_end_idx[i] + 1))
                else:
                    poly_idx = list(range(poly_start_idx, n)) + list(range(0, poly_end_idx[i] + 1))
                boundaries.append(compression_boundary[poly_idx])
        elif compression_boundary.size > 0:  # Entire boundary is in compression (no points on NA line)
            boundaries.append(compression_boundary)

        return boundaries  # Note, Returned polygons are not closed (i.e. does not repeat start point)

    @staticmethod
    def bearing_area_in_compression(xy_points, u, x0=0.0, y0=0.0):
        """
        Accepts vertices representing the bearing element boundary, along with the displacement values at those points.
        Returns the xy_points of the compression regions of the bearing element as separate polygons.
        """
        if len(xy_points) == 0:
            return []

        if not np.array_equal(xy_points[0], xy_points[-1]):
            xy_points = np.vstack([xy_points, xy_points[0]])

        z_values = Utilities.vertical_point_displacements(xy_points, u, x0=x0, y0=y0)
        if all(z_values >= 0):  # No compression region
            return []

        n = len(xy_points)
        compression_edges = []  # Edges connecting points below the xy-plane
        intersection_points = []  # Store intersection points with the xy-plane
        graph = defaultdict(list)  # Graph of compression points and connections

        # Step 1: Traverse the polygon and add intersections
        for i in range(n-1):
            p1, p2 = xy_points[i], xy_points[(i + 1) % n]
            z1, z2 = z_values[i], z_values[(i + 1) % n]

            if z1 <= 0 and z2 <= 0:  # Both points are below the xy-plane
                compression_edges.append((tuple(p1), tuple(p2)))
                graph[tuple(p1)].append(tuple(p2))
                graph[tuple(p2)].append(tuple(p1))
            elif z1 * z2 <= 0:  # Edge crosses the xy-plane (or one vertex is on the xy_plane
                # Calculate intersection point
                t = -z1 / (z2 - z1)
                x_new = p1[0] + t * (p2[0] - p1[0])
                y_new = p1[1] + t * (p2[1] - p1[1])
                intersection_point = (x_new, y_new)
                intersection_points.append(intersection_point)

                # Add edges from below-plane points to the intersection
                if z1 < 0:
                    compression_edges.append((tuple(p1), intersection_point))
                    graph[tuple(p1)].append(intersection_point)
                    graph[intersection_point].append(tuple(p1))
                else:
                    compression_edges.append((intersection_point, tuple(p2)))
                    graph[intersection_point].append(tuple(p2))
                    graph[tuple(p2)].append(intersection_point)

        # Step 2: Identify the extreme intersection point
        intersection_points = sorted(intersection_points, key=lambda p: (p[0], p[1]))

        # Step 3: Add edges between intersection points
        for i in range(0, len(intersection_points) - 1, 2):
            p1, p2 = intersection_points[i], intersection_points[i + 1]

            # Add to the graph
            graph[p1].append(p2)
            graph[p2].append(p1)

            # Add to the compression_edges list
            compression_edges.append((p1, p2))

        # Step 4: Use graph traversal to find connected components
        def bfs(node, visited):
            """Perform BFS to find all connected nodes."""
            component = []
            queue = deque([node])
            visited.add(node)
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            return component

        visited = set()
        compression_regions = []
        for node in graph:
            if node not in visited:
                # Get all connected nodes in this component
                component = bfs(node, visited)
                # Reconstruct a polygon from the component
                polygon = Utilities.reconstruct_polygon(component, compression_edges)
                if len(polygon) > 2:
                    compression_regions.append(np.array(polygon))

        return compression_regions

    @staticmethod
    def reconstruct_polygon(component, edges):
        """
        Reconstruct a closed polygon from a connected component of nodes.
        Ensures correct ordering of vertices based on edge connectivity.
        """
        processed = []
        visited = set()  # Track vertices that have already appeared in the "start" position

        for edge in edges:
            if edge[0] not in visited:
                # If the first vertex of the edge hasn't been a start vertex yet
                processed.append(edge)
                visited.add(edge[0])
            else:
                # Otherwise, reverse the edge to make the second vertex the start
                processed.append((edge[1], edge[0]))
                visited.add(edge[1])

        edges = processed

        edge_map = {edge[0]: edge[1] for edge in edges}

        polygon = [component[0]]  # Start with any point in the component

        while polygon[-1] in edge_map:
            next_point = edge_map[polygon[-1]]
            if next_point == polygon[0]:  # Close the polygon
                break
            polygon.append(next_point)
        return polygon

    @staticmethod
    def reconstruct_polygon_SCRATCH(edges):
        """
        Reconstruct a closed polygon from a connected component of nodes.
        Ensures correct ordering of vertices based on edge connectivity,
        even when edge directions are inconsistent.
        """
        # Initialize the polygon with the first edge
        polygon = list(edges[0])  # Start with the first edge as a list of two points
        edges_to_process = edges[1:]  # Remaining edges to process

        while edges_to_process:
            for i, edge in enumerate(edges_to_process):
                if edge[0] == polygon[-1]:  # Edge starts where the polygon ends
                    polygon.append(edge[1])  # Add the other endpoint
                    edges_to_process.pop(i)
                    break
                elif edge[1] == polygon[-1]:  # Edge ends where the polygon ends
                    polygon.append(edge[0])  # Add the other endpoint (reverse direction)
                    edges_to_process.pop(i)
                    break
                elif edge[0] == polygon[0]:  # Edge starts where the polygon starts
                    polygon.insert(0, edge[1])  # Add the other endpoint at the start
                    edges_to_process.pop(i)
                    break
                elif edge[1] == polygon[0]:  # Edge ends where the polygon starts
                    polygon.insert(0, edge[0])  # Add the other endpoint at the start (reverse direction)
                    edges_to_process.pop(i)
                    break
            else:
                raise ValueError("Edges do not form a single closed polygon")

        # Ensure the polygon is closed (first and last vertices should match)
        if polygon[0] != polygon[-1]:
            polygon.append(polygon[0])

        return polygon

    @staticmethod
    def compute_backing_xy_points(s_or_num_horiz, s_or_num_vert, L_horiz, L_vert, x_offset, y_offset,
                                  place_by_horiz='Spacing', place_by_vert='Spacing', manual_x=None, manual_y=None):

        if manual_x is None:
            manual_x = []
        if manual_y is None:
            manual_y = []


        def get_step_and_number(place_by, spacing_or_number, L):
            if place_by == "Spacing":
                step = spacing_or_number
                if step == 0:
                    num = 0
                else:
                    num = np.floor(L / step + 1)
            else:
                num = spacing_or_number
                if num > 1:
                    step = L / (num - 1)
                else:
                    step = 0
            return step, int(num)

        def get_anchor_coordinates(step, num, offset):
            pts = []
            pt = 0
            if num == 0:
                return pts

            if num % 2 != 0:
                # For odd number of anchors, place one in the center
                pts.append(0 + offset)
                pt += step
            else:
                pt += step / 2

            while len(pts) < num:
                pts.append(pt + offset)
                pts.append(-pt + offset)
                pt += step
            return pts

        num_h = 0
        num_v = 0

        if place_by_horiz == 'Manual':
            x_pts = manual_x
            num_h = len(manual_x)
        else:
            step_h, num_h = get_step_and_number(place_by_horiz, s_or_num_horiz, L_horiz)
            x_pts = get_anchor_coordinates(step_h, num_h, x_offset)

        if place_by_vert == 'Manual':
            y_pts = manual_y
            num_v = len(manual_y)
        else:
            step_v, num_v = get_step_and_number(place_by_vert, s_or_num_vert, L_vert)
            y_pts = get_anchor_coordinates(step_v, num_v, y_offset)

        if place_by_horiz=='Manual' and place_by_vert=='Manual':
            return np.column_stack((manual_x,manual_y))
        elif num_h > 0 and num_v > 0:
            num_points = num_h * num_v
            x_array = np.zeros(num_points)
            y_array = np.zeros(num_points)

            i = 0
            for ix in range(num_h):
                for iy in range(num_v):
                    x_array[i] = x_pts[ix]
                    y_array[i] = y_pts[iy]
                    i += 1
            return np.column_stack((x_array, y_array))
        else:
            return np.empty((0, 2))

    @staticmethod
    def transform_forces(forces, normal_vector):
        """ Transforms a force vector (vx, vy, vz, mx, my, mz) in global coordinates
        defined by a normal vector (and assumed vertical direction aligned with global z"""
        z_vec = np.array([0, 0, 1])
        p_vec = np.cross(z_vec, normal_vector)
        fp = np.dot(forces[0:3], p_vec)
        fz = np.dot(forces[0:3], z_vec)
        fn = np.dot(forces[0:3], normal_vector)
        mp = np.dot(forces[3:6], p_vec)
        mz = np.dot(forces[3:6], z_vec)
        mn = np.dot(forces[3:6], normal_vector)

        return [fp, fz, fn, mp, mz, mn]

    @staticmethod
    def get_center_of_rigidity():
        pass