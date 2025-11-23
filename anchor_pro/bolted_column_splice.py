from anchor_pro.utilities import Utilities
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import root
import os
import pandas as pd

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

matplotlib.use('TkAgg')


class BoltedSplice:
    """ Class for checking the capacity of a bolted splice including the effects of bearing on column below"""

    def __init__(self, bf, tf, tw, d, x_flange, y_flange, x_web, y_web, ru, fy=36):
        self.bf = bf
        self.tf = tf
        self.tw = tw
        self.d = d
        self.ru = ru
        self.fy=fy
        self.delta_max = 0.34

        self.x_flange = x_flange
        self.y_flange = y_flange
        self.x_web = x_web
        self.y_web = y_web

        # self.fasteners = []
        self.xyz_bolt = None
        self.shear_components = None
        self.gauge_length = None
        self.define_fasteners()

        self.boundary = [self.generate_wide_flange_boundary()]
        self.splice_force = np.array([0, 0, 0, 0, 0, 0])
        self.unit_force = None

        self.cz_result = None
        self.kb = None

    def set_splice_force(self, fx, fy, fz, mx, my, mz):
        self.splice_force = np.array([fx, fy, fz, mx, my, mz])
        self.unit_force = self.splice_force / np.linalg.norm(self.splice_force)

    def define_fasteners(self):
        bolt_xyz_list = [(x, (self.d - self.tf) / 2, y) for x, y in zip(self.x_flange, self.y_flange)]
        bolt_xyz_list += [(x, -(self.d - self.tf) / 2, y) for x, y in zip(self.x_flange, self.y_flange)]
        bolt_xyz_list += [(0, x, y) for x, y in zip(self.x_web, self.y_web)]

        self.shear_components = [[1, 0, 1]] * 2 * len(self.x_flange) + [[0, 1, 1]] * len(self.x_web)

        self.xyz_bolt = np.array(bolt_xyz_list)
        self.gauge_length = (2*min(self.xyz_bolt[:, 2]))

    @staticmethod
    def bolt_reaction_to_global_forces(coordinates, reactions):
        x, y, z = coordinates
        rx, ry, rz = reactions
        fx = rx
        fy = ry
        fz = rz
        mx = -z * ry + y * rz
        my = z * rx - x * rz
        mz = -y * rx + x * ry
        return np.array([fx, fy, fz, mx, my, mz])

    def get_bolt_shear_displacement(self, u):
        xyz_deltas = Utilities.compute_point_displacements(self.xyz_bolt, u)
        xyz_deltas_shear = xyz_deltas * self.shear_components
        deltas_shear = np.linalg.norm(xyz_deltas_shear, axis=1)
        return deltas_shear, xyz_deltas_shear

    def get_bolt_resultant(self, u):
        deltas, xyz_deltas = self.get_bolt_shear_displacement(u)
        r_bolts = self.ru * (1 - np.exp(-10 * deltas)) ** 0.55  # AISC Equation
        # Create a mask for non-zero deltas
        nonzero_mask = ~np.isclose(deltas, 0)

        # Initialize the result array with zeros
        r_xyz = np.zeros_like(self.xyz_bolt)

        # Multiply reaction by unit vector in the shear-only direction
        r_xyz[nonzero_mask] = r_bolts[nonzero_mask, np.newaxis] * (
                -xyz_deltas[nonzero_mask] / deltas[nonzero_mask, np.newaxis])

        resultant = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        for i in range(len(self.xyz_bolt)):
            resultant += self.bolt_reaction_to_global_forces(self.xyz_bolt[i], r_xyz[i])
        return resultant, r_xyz

    def get_bolt_dcr(self, u):
        p, r_xyz = self.get_bolt_resultant(u)
        r = np.linalg.norm(r_xyz, axis=1)
        dcr = r / self.ru
        return dcr

    def get_compression_zone_properties(self, u):
        """Returns the geometric properties of the compression zones for given dof values, u"""
        # Identify area of bearing elements that are in compression
        # Loop trough each bearing boundary, then loop through each compression area and assemble list.
        compression_boundaries = [
            compression_boundary
            for boundary in self.boundary
            for compression_boundary in
            Utilities.bearing_area_in_compression(boundary, u, x0=0, y0=0)]

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
            # beta = 29000 / self.gauge_length

            beta = self.fy/(self.delta_max+0.25)

            areas[i] = area
            centroids[i] = centroid
            Ixx_list[i] = Ixx
            Iyy_list[i] = Iyy
            Ixy_list[i] = Ixy
            beta_list[i] = beta

        self.cz_result = {'compression_boundaries': compression_boundaries,
                          'areas': areas,
                          'centroids': centroids,
                          'Ixx': Ixx_list,
                          'Iyy': Iyy_list,
                          'Ixy': Ixy_list,
                          'beta': beta_list}

    def update_bearing_stiffness_matrix(self, u):
        """Returns the bearing stiffness matrix kb based on dof displacemetns, u"""
        self.kb = np.zeros((6, 6))
        self.get_compression_zone_properties(u)

        # get compression zones
        for A, (x_bar, y_bar), Ixx, Iyy, Ixy, beta in zip(
                self.cz_result['areas'],
                self.cz_result['centroids'],
                self.cz_result['Ixx'],
                self.cz_result['Iyy'],
                self.cz_result['Ixy'],
                self.cz_result['beta']):
            kb_cz = self.compression_zone_matrix(A, x_bar, y_bar, Ixx, Iyy, Ixy, beta)
            self.kb += kb_cz

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

    def get_compression_resultants(self, u):
        """Computes the compression zone resultant forces and centroids"""
        self.update_bearing_stiffness_matrix(u)
        resultant = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if len(self.cz_result['compression_boundaries']) != 0:

            x0 = 0
            y0 = 0

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

                resultant += -kb_cz @ u

                fz[i] = resultant[2]
                mx[i] = resultant[3]
                my[i] = resultant[4]

            self.cz_result['fz'] = fz
            self.cz_result['resultant_centroids'] = np.column_stack((-my / fz + x0, mx / fz + y0))
        else:

            self.cz_result['fz'] = []
            self.cz_result['resultant_centroids'] = []
        return resultant

    def get_max_compression_stress(self,u):
        stresses = []
        for i, bound in enumerate(self.cz_result['compression_boundaries']):
            # Extract displacements at vertices of compression boundary
            dz = (Utilities.vertical_point_displacements(bound, u))

            # Multiply displacement by beta to get stress at vertex
            stresses.extend([d*self.cz_result['beta'][i] for d in dz])

        return min(stresses) if stresses else 0

    def get_rn_for_dof_direction(self, u):
        u_scaled = self.get_scaled_dof_displacements(u)

        # Compute reaction force vector
        r = -(self.get_bolt_resultant(u_scaled)[0] + self.get_compression_resultants(u_scaled))
        return r, u_scaled

    def get_scaled_dof_displacements(self, u0):
        """ Given an input DOF solution, the function returns the DOF values scaled so that the extreme fastener is at a displacement of delta max"""

        # Compute displacements at all bolts
        deltas, _ = self.get_bolt_shear_displacement(u0)

        # Scale displacement vector to satisfy max bolt displacement constraint
        sf = self.delta_max / max(deltas)
        u_scaled = sf * u0
        return u_scaled

    def resultant_direction_residual(self, u):
        r, u_scaled = self.get_rn_for_dof_direction(u)

        # Normalize reaction force vector
        r_unit = r / np.linalg.norm(r)

        # Penalty if solution results in excessive deformations
        comp_stress = self.get_max_compression_stress(u_scaled)
        penalty = 1e6 if comp_stress < -self.fy else 0

        # Residual: Difference between reaction direction and applied load direction
        return 100*r_unit - 100*self.unit_force + penalty

    def analyze_splice2(self, u0, method='lm', verbose=False):
        # Iterate until Resultant direction is in Direction of self.splice_force
        # if method=='lm':
        #     options = {'ftol': 1e-8, 'xtol': 1e-8, 'gtol': 1e-8}
        # else:
        #     options = {}
        result = root(self.resultant_direction_residual, u0, method=method)

        # Extract the solution for the displacements
        sol = self.get_scaled_dof_displacements(result.x)
        if verbose:
            return sol, result.success, result
        else:
            return sol, result.success

    def get_splice_dcr(self, u_final):
        # Compute final scaled displacement
        u_final_unit = u_final / np.linalg.norm(u_final)

        # Compute final reaction forces
        r_n, u_scaled = self.get_rn_for_dof_direction(u_final)

        # Compute Demand-to-Capacity Ratio (DCR)
        DCR = np.linalg.norm(self.splice_force) / np.linalg.norm(r_n)
        return DCR

    def generate_wide_flange_boundary(self):
        """
        Generate the outline of a wide flange beam as a list of (x, y) tuples.

        Parameters:
            bf (float): Flange width.
            tf (float): Flange thickness.
            tw (float): Web thickness.
            d (float): Overall depth of the beam.

        Returns:
            list of tuple: List of (x, y) points representing the beam outline.
        """
        # Half dimensions for symmetry
        half_bf = self.bf / 2
        half_tw = self.tw / 2
        half_d = self.d / 2
        tf = self.tf

        # Points defining the I-beam outline
        boundary = [
            # Top flange (clockwise from top-left)
            (-half_bf, half_d),
            (half_bf, half_d),
            (half_bf, half_d - tf),
            (half_tw, half_d - tf),
            (half_tw, -half_d + tf),
            (half_bf, -half_d + tf),
            (half_bf, -half_d),
            (-half_bf, -half_d),
            (-half_bf, -half_d + tf),
            (-half_tw, -half_d + tf),
            (-half_tw, half_d - tf),
            (-half_bf, half_d - tf),
            # Closing the shape
            # (-half_bf, half_d)
        ]

        return np.array(boundary)

    def plot_section(self):
        # Plotting
        plt.figure()

        # Plot top flange
        def closed_shape(arr):
            boundary_x = list(arr[:, 0])
            boundary_x.append(boundary_x[0])
            boundary_y = list(arr[:, 1])
            boundary_y.append(boundary_y[0])
            return boundary_x, boundary_y

        plt.plot(*closed_shape(self.boundary[0]), '-k', linewidth=1)
        for cz in self.cz_result['compression_boundaries']:
            plt.plot(cz[:, 0], cz[:, 1], linewidth=2)

        # Labels and legend
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.title('Splice Section')
        plt.show()

    def plot_extruded_wireframe(self):
        """
        Plot an extruded wireframe figure based on the outline with a given height.

        Parameters:
            outline (list of tuple): List of (x, y) points defining the base outline.
            height (float): Height of the extrusion.
        """
        # Separate x and y coordinates from the outline
        x, y = zip(*self.boundary[0])
        x = x + (x[-1],)
        y = y + (y[-1],)

        height = max(self.y_flange + self.y_web) + 3

        # Create vertices for the top and bottom faces
        z_bottom = [0] * len(x)
        z_top = [height] * len(x)

        # Create 3D points for the top and bottom faces
        bottom_face = list(zip(x, y, z_bottom))
        top_face = list(zip(x, y, z_top))

        # Prepare 3D edges
        edges = []
        for i in range(len(x) - 1):  # -1 to avoid duplicating the closing edge
            edges.append([bottom_face[i], bottom_face[i + 1]])  # Bottom edges
            edges.append([top_face[i], top_face[i + 1]])  # Top edges
            edges.append([bottom_face[i], top_face[i]])  # Vertical edges

        # Plot using matplotlib
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Add edges to the plot
        for edge in edges:
            x_coords, y_coords, z_coords = zip(*edge)
            ax.plot(x_coords, y_coords, z_coords, color='blue')

        # Add the top and bottom faces as filled polygons
        ax.add_collection3d(Poly3DCollection([bottom_face], color='lightblue', alpha=0.5))
        ax.add_collection3d(Poly3DCollection([top_face], color='lightblue', alpha=0.5))

        # Set plot limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    def plot_3d_wireframe(self, u=None, p=None):
        """
        Plot a 3D wireframe with three rectangles representing the flanges and the web.

        Parameters:
            bf (float): Flange width.
            tf (float): Flange thickness.
            tw (float): Web thickness.
            d (float): Overall depth of the beam.
            height (float): Extrusion height.
        """
        # Half dimensions for symmetry
        half_bf = self.bf / 2
        half_tw = self.tw / 2
        half_tf = self.tf / 2
        half_d = self.d / 2
        height = max(self.y_flange + self.y_web) + 3

        # Define rectangles in 3D
        top_flange = [(-half_bf, half_d - half_tf, 0), (half_bf, half_d - half_tf, 0),
                      (half_bf, half_d - half_tf, height),
                      (-half_bf, half_d - half_tf, height), (-half_bf, half_d - half_tf, 0)]
        bottom_flange = [(-half_bf, -(half_d - half_tf), 0), (half_bf, -(half_d - half_tf), 0),
                         (half_bf, -(half_d - half_tf), height),
                         (-half_bf, -(half_d - half_tf), height), (-half_bf, -(half_d - half_tf), 0)]
        web = [(0, half_d - half_tf, 0), (0, -(half_d - half_tf), 0), (0, -(half_d - half_tf), height),
               (0, (half_d - half_tf), height), (0, (half_d - half_tf), 0)]

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot top flange
        for i in range(len(top_flange) - 1):
            x, y, z = zip(top_flange[i], top_flange[i + 1])
            ax.plot(x, y, z, '-.k')

        # Plot bottom flange
        for i in range(len(bottom_flange) - 1):
            x, y, z = zip(bottom_flange[i], bottom_flange[i + 1])
            ax.plot(x, y, z, '-.k')

        # Plot web
        for i in range(len(web) - 1):
            x, y, z = zip(web[i], web[i + 1])
            ax.plot(x, y, z, '-.k')

        # Plot Fasteners
        for xyz, shear_dir in zip(self.xyz_bolt, self.shear_components):
            normal = np.array([1, 1, 1]) - np.array(shear_dir)
            length = 0.25
            ax.plot([xyz[0] - length * normal[0], xyz[0] + length * normal[0]],
                    [xyz[1] - length * normal[1], xyz[1] + length * normal[1]],
                    [xyz[2] - length * normal[2], xyz[2] + length * normal[2]], '.-k')

        # Plot applied Forces
        force_scale = None
        if p is not None:
            # Plot applied Forces
            if p is not None:
                # Unpack the applied forces
                fx, fy, fz, mx, my, mz = p

                # Normalize force components (fx, fy, fz) to a maximum length of 3
                force_magnitude = np.linalg.norm([fx, fy, fz])
                if force_magnitude != 0:
                    force_scale = 0.5 * self.d / force_magnitude
                    fx, fy, fz = fx * force_scale, fy * force_scale, fz * force_scale

                # Plot fx and fy quivers
                ax.quiver(0, 0, -1, fx, 0, 0, color='g', arrow_length_ratio=0.2, capstyle='round')  # fx
                ax.quiver(0, 0, -1, 0, fy, 0, color='g', arrow_length_ratio=0.2, capstyle='round')  # fy

                # Plot fz quiver
                if fz < 0:
                    ax.quiver(0, 0, -1, 0, 0, fz, color='g', arrow_length_ratio=0.2, capstyle='round')  # Downward
                else:
                    ax.quiver(0, 0, -1 - fz, 0, 0, fz, color='g', arrow_length_ratio=0.2, capstyle='round')  # Upward

                # Normalize moment components (mx, my, mz) to a maximum length of 3
                moment_magnitude = np.linalg.norm([mx, my, mz])
                if moment_magnitude != 0:
                    moment_scale = 0.5 * self.d / moment_magnitude
                    mx, my, mz = mx * moment_scale, my * moment_scale, mz * moment_scale

                # Plot mx quiver
                if np.sign(mx) == np.sign(fx):
                    ax.quiver(fx + np.sign(fx) * 0.5, 0, -1, mx, 0, 0, color='g', linestyle='dashed',
                              arrow_length_ratio=0.2,
                              capstyle='round')
                else:
                    ax.quiver(0, 0, -1, mx, 0, 0, color='g', linestyle='dashed', arrow_length_ratio=0.2,
                              capstyle='round')

                # Plot my quiver
                if np.sign(my) == np.sign(fy):
                    ax.quiver(0, fy + np.sign(fy) * 0.5, -1, 0, my, 0, color='g', linestyle='dashed',
                              arrow_length_ratio=0.2,
                              capstyle='round')
                else:
                    ax.quiver(0, 0, -1, 0, my, 0, color='g', linestyle='dashed', arrow_length_ratio=0.2,
                              capstyle='round')

                # Plot mz quiver
                if mz < 0:
                    ax.quiver(0, 0, -abs(fz) - 1.5, 0, 0, mz, color='g', arrow_length_ratio=0.2,
                              capstyle='round')  # Downward
                else:
                    ax.quiver(0, 0, -1.5 - abs(fz) - mz, 0, 0, mz, color='g', arrow_length_ratio=0.2,
                              capstyle='round')  # Upward


        if u is not None:
            # Plot Bolt Reactions
            bolt_forces = self.get_bolt_resultant(u)[1]
            arrow_scale_factor = force_scale if force_scale else 2 / max(np.linalg.norm(bolt_forces, axis=1))
            # arrow_scale_factor = 2/max(np.linalg.norm(bolt_forces, axis=1))
            for xyz, force in zip(self.xyz_bolt, bolt_forces):
                f = force * arrow_scale_factor
                ax.quiver(*xyz, *(f), color='r', arrow_length_ratio=0.2, capstyle='round')

            # Compression Zone Reactions
            def closed_shape(arr):
                boundary_x = list(arr[:, 0])
                boundary_x.append(boundary_x[0])
                boundary_y = list(arr[:, 1])
                boundary_y.append(boundary_y[0])
                return boundary_x, boundary_y

            # Plot the main boundary
            boundary_x, boundary_y = closed_shape(self.boundary[0])
            ax.plot(boundary_x, boundary_y, np.zeros_like(boundary_x), '-k', linewidth=1)

            # Fill the compression boundaries with a shaded region
            for bound, cent, fz in zip(self.cz_result['compression_boundaries'],
                                       self.cz_result['resultant_centroids'], self.cz_result['fz']):

                cz_x, cz_y = closed_shape(bound)
                vertices = [list(zip(cz_x, cz_y, np.zeros_like(cz_x)))]  # Create 3D vertices
                poly = Poly3DCollection(vertices, color='blue', alpha=0.5)
                ax.add_collection3d(poly)

                # fz = arrow_scale_factor*fz
                # ax.quiver(cent[0], cent[1], -1-fz, 0, 0, fz, color='b', arrow_length_ratio=0.2, capstyle='round')

        # Turn off the grid
        ax.grid(False)

        # Turn off the grey background
        ax.set_facecolor('white')

        # Optionally, hide the axes pane (e.g., walls of the 3D plot)
        ax.w_xaxis.pane.fill = False
        ax.w_yaxis.pane.fill = False
        ax.w_zaxis.pane.fill = False

        # Labels and legend
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Splice Configuration and Forces')
        plt.show()


class SpliceAnalyzer:
    def __init__(self, splice_wb_path, forces_paths_dictionary, output_folder, rn):
        self.wb_path = splice_wb_path
        self.forces_paths = forces_paths_dictionary
        self.output_folder = output_folder
        self.rn = rn

        self.df_splice_props = None
        self.df_elems = None
        self.splice_forces = None  # dictionary of dataframes

        self.df_dcr_results = None
        self.df_time_steps = None
        self.df_max_stress = None
        self.df_non_convergence_rate = None

    def analyze_single_splice(self, elem_id, gm_key=None, time_step=None):
        self.read_input_data(gm_key=gm_key)
        df_elem = self.df_elems[self.df_elems['Elem_No'] == elem_id]
        df_splice_props = self.df_splice_props[self.df_splice_props['SpliceID'] == df_elem['SpliceID'].iloc[0]]
        splice, sol, p = self.main_loop(df_splice_props, df_elem, self.splice_forces, self.rn, step_number=time_step)

        return splice, sol, p

    def analyze_all_splices(self):
        self.read_input_data()
        self.main_loop(self.df_splice_props, self.df_elems, self.splice_forces, self.rn)

    def read_input_data(self, gm_key=None):
        # Import Tables
        print('Loading Splice Properties')
        self.df_splice_props = pd.read_excel(self.wb_path, sheet_name='SpliceProps', header=1)
        print('Loading Element List')
        self.df_elems = pd.read_excel(self.wb_path, sheet_name='ElemProps', header=1)

        print('Loading Splice Forces')
        if gm_key:  # Load a single gm record only
            self.splice_forces = {gm_key: pd.read_csv(self.forces_paths[gm_key])}
        else:  # Load all gm records
            self.splice_forces = {k: pd.read_csv(path) for k, path in self.forces_paths.items()}

        # Initialize Results Dataframes
        self.df_dcr_results = pd.DataFrame(columns=[key for key in self.forces_paths],
                                           index=self.df_elems['Elem_No'])

        self.df_max_stress = pd.DataFrame(columns=[key for key in self.forces_paths],
                                           index=self.df_elems['Elem_No'])

        self.df_time_steps = pd.DataFrame(columns=[key for key in self.forces_paths],
                                           index=self.df_elems['Elem_No'])

        self.df_non_convergence_rate = pd.DataFrame(columns=[key for key in self.forces_paths],
                                                    index=self.df_elems['Elem_No'])

    def main_loop(self, df_splice_props, df_elems, splice_forces, rn, step_number=None):
        """
        df_splice_props: a dataframe containing data related to the number and layout of fasteners,
        df_elems: a dataframe containing a list of elements and which splice id to use
        """

        # Define Splice Objects
        splices = {}
        for i, row in df_splice_props.iterrows():
            name = row['SpliceID']
            print(f'Defining {name} Object')
            bf = row['bf_above']
            tf = row['tf_above']
            tw = row['tw_above']
            d = row['d_above']

            x_flange = [float(value) for value in row['x_flange_rivets'].split(",")]
            y_flange = [float(value) for value in row['y_flange_rivets'].split(",")]
            x_web = [float(value) for value in row['x_web_rivets'].split(",")]
            y_web = [float(value) for value in row['y_web_rivets'].split(",")]

            splices[name] = BoltedSplice(bf, tf, tw, d, x_flange, y_flange, x_web, y_web, rn, fy=39)

        # Iterate through all elements
        # for row in [df_elems.iloc[0]]:
        for i, row in df_elems.iterrows():
            ele_id = row['Elem_No']
            print(f'##### Checking Element {ele_id} #####')
            name = row['SpliceID']
            splice = splices[name]

            # Iterate through GM records
            for gm, df in splice_forces.items():
                print(gm)
                # Filter data for the current element ID
                element_data = df[df['ID'] == ele_id]
                if step_number:
                    steps = [step_number]
                else:
                    steps = element_data['Step'].unique()

                max_dcr = 0
                max_bearing_stress = None
                max_step = None

                # Iterate through load points and retrieve DCRs, store highest DCR
                fails = 0
                for step in steps:

                    # print(f'GM: {gm}, Step: {step}')
                    # Filter data for the current step
                    step_data = element_data[element_data['Step'] == step]

                    # Extract p, mx, and my values based on 'ValType'
                    vx = -step_data.loc[step_data['ValType'] == 'V3J', 'Val'].values[0]
                    vy = step_data.loc[step_data['ValType'] == 'V2J', 'Val'].values[0]
                    pu = step_data.loc[step_data['ValType'] == 'PJ', 'Val'].values[0]
                    mx = -step_data.loc[step_data['ValType'] == 'M3J', 'Val'].values[0]
                    my = step_data.loc[step_data['ValType'] == 'M2J', 'Val'].values[0]
                    tz = 0.0

                    # Define Splice Forces
                    splice.set_splice_force(vx, vy, pu, mx, my, 0.0)
                    force_norm = splice.splice_force / np.linalg.norm(splice.splice_force)

                    # Define initial displacement guesses
                    u_init_options = [-1e-7 * force_norm,
                                      *[-1e-7 * np.array(
                                          [np.sign(splice.splice_force[j]) if j == i else 0 for j in range(6)]) for i in
                                        range(6)]]
                    dcr = 0

                    # Analyze splice
                    sol, ok = splice.analyze_splice2(u_init_options[0], method='hybr')
                    if ok:
                        dcr = splice.get_splice_dcr(sol)
                    else:
                        for i, u_init in enumerate(u_init_options):
                            sol, ok = splice.analyze_splice2(u_init)
                            if ok:
                                # dcr = np.max(splice.get_bolt_dcr(sol))
                                dcr = splice.get_splice_dcr(sol)
                                break
                            elif i == len(u_init_options) - 1:
                                fails += 1
                                print(f'GM: {gm}, step: {step}, failed to converge.')
                    if dcr > max_dcr:
                        max_dcr = dcr
                        max_step = step
                        max_bearing_stress = -splice.get_max_compression_stress(sol)

                # Store the maximum DCR for this element and ground motion
                self.df_dcr_results.loc[ele_id, gm] = max_dcr
                self.df_time_steps.loc[ele_id, gm] = max_step
                self.df_max_stress.loc[ele_id, gm] = max_bearing_stress
                self.df_non_convergence_rate.loc[ele_id, gm] = fails / len(steps)

        max_values = self.df_dcr_results.max(axis=1)
        mean_values = self.df_dcr_results.mean(axis=1)
        self.df_dcr_results['Max'] = max_values
        self.df_dcr_results['Mean'] = mean_values

        max_values = self.df_max_stress.max(axis=1)
        mean_values = self.df_max_stress.mean(axis=1)
        self.df_max_stress['Max'] = max_values
        self.df_max_stress['Mean'] = mean_values

        self.df_dcr_results.to_csv(os.path.join(self.output_folder, 'Splice-Results.csv'))
        self.df_max_stress.to_csv(os.path.join(self.output_folder, 'Max Bearing Stress.csv'))
        self.df_time_steps.to_csv(os.path.join(self.output_folder, 'Governing Time Steps.csv'))
        self.df_non_convergence_rate.to_csv(os.path.join(self.output_folder, 'Non-Convergence-Rate.csv'))

        if len(df_elems) == 1:
            return splice, sol, (vx, vy, pu, mx, my, tz)
