"""Interactive UI components for facial region editing and preview.

This module provides PyVista-based interactive tools for refining facial
region detection and previewing anonymization results.

Initial Contributors:
    - Face Anonymization Project | 2024
"""

import logging

import numpy as np
import pyvista as pv

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from cedalion import units

from .validator import ValidationMetrics


logger = logging.getLogger("cedalion")


class FacialRegionEditor:
    """Interactive editor for facial region mask.

    Allows user to:
    - See detected facial region highlighted
    - Add/remove vertices from mask by clicking
    - Verify protected points are excluded

    Follows the pattern of OptodeSelector in cedalion.plots.

    Attributes:
        surface: The TrimeshSurface being edited
        mask: Current facial region mask
        protected_points: Points that cannot be part of facial region
        plotter: PyVista plotter instance
    """

    def __init__(
        self,
        surface: cdc.TrimeshSurface,
        initial_mask: np.ndarray,
        protected_points: cdt.LabeledPointCloud,
        protection_radius: float = 15.0,
    ):
        """Initialize the facial region editor.

        Args:
            surface: The TrimeshSurface to edit
            initial_mask: Initial boolean mask for facial region
            protected_points: Points to protect (optodes + landmarks)
            protection_radius: Radius around protected points in mm
        """
        self.surface = surface
        self.mask = initial_mask.copy()
        self.protected_points = protected_points
        self.protection_radius = protection_radius
        self.plotter = pv.Plotter()
        self._mesh_actor = None
        self._protected_actors = []

    def _get_vtk_mesh(self) -> pv.PolyData:
        """Convert surface to PyVista mesh."""
        vtk_surface = cdc.VTKSurface.from_trimeshsurface(self.surface)
        return pv.wrap(vtk_surface.mesh)

    def _update_visualization(self):
        """Update the mesh visualization with current mask."""
        if self._mesh_actor is not None:
            self.plotter.remove_actor(self._mesh_actor)

        mesh = self._get_vtk_mesh()

        # Create scalars for visualization
        # 0 = non-facial, 1 = facial region
        scalars = self.mask.astype(float)

        self._mesh_actor = self.plotter.add_mesh(
            mesh,
            scalars=scalars,
            cmap=["white", "red"],
            clim=[0, 1],
            show_scalar_bar=False,
            opacity=0.9,
            smooth_shading=True,
            pickable=True,
        )

    def _add_protected_points(self):
        """Add spheres for protected points."""
        points = self.protected_points.pint.dequantify().values
        labels = [str(l) for l in self.protected_points.label.values]

        for i, (point, label) in enumerate(zip(points, labels)):
            sphere = pv.Sphere(radius=3, center=point)
            actor = self.plotter.add_mesh(
                sphere, color="green", smooth_shading=True
            )
            self._protected_actors.append(actor)

            # Add label
            self.plotter.add_point_labels(
                [point], [label], font_size=12, point_color="green"
            )

    def _on_pick(self, picked_point):
        """Handle vertex picking to toggle mask."""
        if picked_point is None:
            return

        point = np.array(picked_point)

        # Check if point is protected
        protected = self.protected_points.pint.dequantify().values
        distances = np.linalg.norm(protected - point, axis=1)

        if np.any(distances < self.protection_radius):
            logger.info("Cannot modify protected region")
            return

        # Find nearest vertex
        vertices = self.surface.mesh.vertices
        distances = np.linalg.norm(vertices - point, axis=1)
        nearest_idx = np.argmin(distances)

        # Use brush to toggle multiple vertices
        brush_radius = 10.0  # mm
        affected_mask = distances < brush_radius

        # Toggle mask for affected vertices
        if self.mask[nearest_idx]:
            # Remove from facial region
            self.mask[affected_mask] = False
        else:
            # Add to facial region (excluding protected)
            protected_positions = self.protected_points.pint.dequantify().values
            for idx in np.where(affected_mask)[0]:
                v = vertices[idx]
                dist_to_protected = np.min(
                    np.linalg.norm(protected_positions - v, axis=1)
                )
                if dist_to_protected >= self.protection_radius:
                    self.mask[idx] = True

        self._update_visualization()

    def show(self) -> np.ndarray:
        """Display editor and return refined mask.

        Returns:
            Refined boolean mask after user editing
        """
        # Set up visualization
        self._update_visualization()
        self._add_protected_points()

        # Enable picking
        self.plotter.enable_surface_point_picking(
            callback=self._on_pick,
            show_message="Click to add/remove vertices from facial region",
            show_point=False,
            tolerance=0.005,
        )

        # Add instructions
        self.plotter.add_text(
            "Left-click: Toggle facial region\n"
            "Green spheres: Protected points (cannot modify)\n"
            "Close window when done",
            position="upper_left",
            font_size=10,
        )

        # Show and wait
        self.plotter.show()

        return self.mask


class AnonymizationPreview:
    """Side-by-side comparison of original and anonymized meshes.

    Provides interactive visualization to compare the original scan
    with the anonymized version, including validation metrics display.

    Attributes:
        original: Original TrimeshSurface
        anonymized: Anonymized TrimeshSurface
        metrics: Validation metrics to display
    """

    def __init__(
        self,
        original: cdc.TrimeshSurface,
        anonymized: cdc.TrimeshSurface,
        metrics: ValidationMetrics = None,
    ):
        """Initialize the preview.

        Args:
            original: Original TrimeshSurface
            anonymized: Anonymized TrimeshSurface
            metrics: Optional validation metrics to display
        """
        self.original = original
        self.anonymized = anonymized
        self.metrics = metrics

    def _get_vtk_mesh(self, surface: cdc.TrimeshSurface) -> pv.PolyData:
        """Convert surface to PyVista mesh."""
        vtk_surface = cdc.VTKSurface.from_trimeshsurface(surface)
        return pv.wrap(vtk_surface.mesh)

    def show(self):
        """Display side-by-side comparison."""
        # Create plotter with two viewports
        plotter = pv.Plotter(shape=(1, 2))

        # Left viewport: Original
        plotter.subplot(0, 0)
        original_mesh = self._get_vtk_mesh(self.original)

        # Check if mesh has colors
        if hasattr(self.original.mesh.visual, 'vertex_colors'):
            try:
                colors = self.original.mesh.visual.to_color().vertex_colors
                original_mesh["colors"] = colors[:, :3]  # RGB only
                plotter.add_mesh(
                    original_mesh,
                    scalars="colors",
                    rgb=True,
                    smooth_shading=True,
                )
            except Exception:
                plotter.add_mesh(original_mesh, color="white", smooth_shading=True)
        else:
            plotter.add_mesh(original_mesh, color="white", smooth_shading=True)

        plotter.add_text("Original", position="upper_left", font_size=14)

        # Right viewport: Anonymized
        plotter.subplot(0, 1)
        anonymized_mesh = self._get_vtk_mesh(self.anonymized)

        if hasattr(self.anonymized.mesh.visual, 'vertex_colors'):
            try:
                colors = self.anonymized.mesh.visual.to_color().vertex_colors
                anonymized_mesh["colors"] = colors[:, :3]
                plotter.add_mesh(
                    anonymized_mesh,
                    scalars="colors",
                    rgb=True,
                    smooth_shading=True,
                )
            except Exception:
                plotter.add_mesh(anonymized_mesh, color="white", smooth_shading=True)
        else:
            plotter.add_mesh(anonymized_mesh, color="white", smooth_shading=True)

        plotter.add_text("Anonymized", position="upper_left", font_size=14)

        # Add metrics if available
        if self.metrics is not None:
            status = "PASS" if self.metrics.protected_points_preserved else "FAIL"
            metrics_text = (
                f"Validation: {status}\n"
                f"Max Protected Deviation: {self.metrics.max_protected_deviation:.3f}mm\n"
                f"Facial Displacement: {self.metrics.facial_displacement_mean:.1f}mm (mean)"
            )
            plotter.add_text(
                metrics_text,
                position="lower_left",
                font_size=10,
            )

        # Link cameras so both views rotate together
        plotter.link_views()

        plotter.show()


class DisplacementViewer:
    """Visualize vertex displacement as a heatmap on the mesh.

    Shows how much each vertex moved during anonymization using
    a color-coded visualization.
    """

    def __init__(
        self,
        surface: cdc.TrimeshSurface,
        displacements: np.ndarray,
        facial_mask: np.ndarray = None,
    ):
        """Initialize the displacement viewer.

        Args:
            surface: The anonymized TrimeshSurface
            displacements: Per-vertex displacement values in mm
            facial_mask: Optional mask to highlight facial region
        """
        self.surface = surface
        self.displacements = displacements
        self.facial_mask = facial_mask

    def _get_vtk_mesh(self) -> pv.PolyData:
        """Convert surface to PyVista mesh."""
        vtk_surface = cdc.VTKSurface.from_trimeshsurface(self.surface)
        return pv.wrap(vtk_surface.mesh)

    def show(self, cmap: str = "hot"):
        """Display displacement heatmap.

        Args:
            cmap: Colormap for displacement visualization
        """
        plotter = pv.Plotter()

        mesh = self._get_vtk_mesh()
        mesh["displacement"] = self.displacements

        plotter.add_mesh(
            mesh,
            scalars="displacement",
            cmap=cmap,
            smooth_shading=True,
            scalar_bar_args={
                "title": "Displacement (mm)",
                "vertical": True,
            },
        )

        # Add statistics
        stats_text = (
            f"Max: {self.displacements.max():.2f}mm\n"
            f"Mean: {self.displacements.mean():.2f}mm\n"
            f"Median: {np.median(self.displacements):.2f}mm"
        )

        if self.facial_mask is not None:
            facial_disp = self.displacements[self.facial_mask]
            stats_text += (
                f"\n\nFacial Region:\n"
                f"Max: {facial_disp.max():.2f}mm\n"
                f"Mean: {facial_disp.mean():.2f}mm"
            )

        plotter.add_text(stats_text, position="upper_right", font_size=10)

        plotter.show()


def quick_preview(
    original: cdc.TrimeshSurface,
    anonymized: cdc.TrimeshSurface,
    metrics: ValidationMetrics = None,
):
    """Quick function to preview anonymization results.

    Convenience function that creates an AnonymizationPreview and shows it.

    Args:
        original: Original TrimeshSurface
        anonymized: Anonymized TrimeshSurface
        metrics: Optional validation metrics
    """
    preview = AnonymizationPreview(original, anonymized, metrics)
    preview.show()


def quick_displacement_view(
    surface: cdc.TrimeshSurface,
    displacements: np.ndarray,
    facial_mask: np.ndarray = None,
):
    """Quick function to view displacement heatmap.

    Args:
        surface: The anonymized TrimeshSurface
        displacements: Per-vertex displacement values in mm
        facial_mask: Optional mask to highlight facial region
    """
    viewer = DisplacementViewer(surface, displacements, facial_mask)
    viewer.show()
