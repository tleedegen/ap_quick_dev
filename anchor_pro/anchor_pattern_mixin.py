# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 10:54:28 2025

@author: djmiller
"""
import numpy as np

class AnchorPatternMixin():
    @staticmethod
    def get_anchor_spacing_matrix(xy_anchors: np.ndarray) -> np.ndarray:
        # Get Inter-Anchor Spacing

        # n = len(xy_anchors)
        # spacing_matrix = np.empty((n, n))
        # for i in range(n):
        #     for j in range(n):
        #         spacing_matrix[i, j] = np.linalg.norm(xy_anchors[j] - xy_anchors[i])
        # return spacing_matrix

        # shape: (n, n, 2) -> distances along last axis
        diffs = xy_anchors[:, None, :] - xy_anchors[None, :, :]  # (n, n, 2)
        spacing_matrix = np.linalg.norm(diffs, axis=-1)  # (n, n)
        return spacing_matrix

    @staticmethod
    def get_anchor_groups(radius, spacing_matrix):
        boolean_matrix = spacing_matrix < radius
        boolean_matrix = boolean_matrix.astype(int)  # Convert the boolean matrix to integer for matrix operations
        groups_matrix = np.copy(boolean_matrix)

        # Keep multiplying until there are no new connections found
        while True:
            new_matrix = groups_matrix @ boolean_matrix

            # Ensure we don't exceed the boolean logic (1 is True, everything above 1 is still True)
            new_matrix[new_matrix > 1] = 1

            # Check if the matrix has changed
            if np.array_equal(new_matrix, groups_matrix):
                break
            else:
                groups_matrix = new_matrix

        # Convert back to boolean
        return groups_matrix.astype(bool)