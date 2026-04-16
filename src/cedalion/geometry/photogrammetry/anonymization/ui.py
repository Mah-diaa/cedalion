"""Interactive UI components for nasion selection.

Provides a PyVista-based interactive picker for manual nasion (Nz)
selection as a fallback when automatic detection fails.

Initial Contributors:
    - Face Anonymization Project | 2024
"""

import logging

import numpy as np
import pyvista as pv

import cedalion.dataclasses as cdc

logger = logging.getLogger("cedalion")


def pick_nasion(surface: cdc.TrimeshSurface) -> np.ndarray | None:
    """Open an interactive window for the user to click the nasion point.

    Displays the textured mesh and lets the user click on the nasion (Nz).
    The clicked point is snapped to the nearest mesh vertex.

    Args:
        surface: TrimeshSurface from photogrammetry scan.

    Returns:
        Nasion position as numpy array of shape (3,) in mm, or None if
        the user closes the window without clicking.
    """
    from scipy.spatial import KDTree

    picked_point = [None]  # mutable container for closure

    def _on_pick(point):
        if point is not None:
            picked_point[0] = np.array(point)

    plotter = pv.Plotter(notebook=False)

    # Add textured mesh using cedalion's plot_surface for consistency
    import cedalion.plots
    cedalion.plots.plot_surface(plotter, surface, opacity=1.0)

    plotter.enable_surface_point_picking(
        callback=_on_pick,
        left_clicking=True,
        show_point=True,
        point_size=20,
        color="red",
        show_message="Left-click Nz (nasion), then close the window",
    )

    plotter.add_text(
        "Left-click NASION (bridge of the nose), then close window",
        position="upper_left",
        font_size=14,
    )

    plotter.show()

    if picked_point[0] is None:
        return None

    # Snap to nearest mesh vertex
    vertices = surface.mesh.vertices
    tree = KDTree(vertices)
    _, idx = tree.query(picked_point[0])
    nasion = vertices[idx].copy()
    logger.debug(f"Manual nasion selected: {nasion}")
    return nasion


def refine_mask_interactive(
    surface: cdc.TrimeshSurface,
    mask: np.ndarray,
    protected_positions: np.ndarray,
    protected_labels: list[str] | None = None,
    brush_radius: float = 25.0,
    protection_radius: float = 15.0,
) -> np.ndarray:
    """Interactively refine a deletion mask with a click-to-paint brush.

    Opens a PyVista window showing the surface colored by ``mask`` (red = to
    delete, white = keep). Left-click paints vertices within ``brush_radius``
    under the cursor. Keys:

    - ``t`` -- toggle between ADD and REMOVE modes
    - ``+`` -- increase brush radius by 5mm (capped at 100mm)
    - ``-`` -- decrease brush radius by 5mm (floor 5mm)

    In ADD mode, clicks inside ``protection_radius`` of any protected position
    are ignored (used for landmarks/optodes that must remain).

    Args:
        surface: TrimeshSurface to refine.
        mask: Initial deletion mask, shape (n_vertices,).
        protected_positions: Protected landmark positions, shape (K, 3).
        protected_labels: Labels for each protected position (for display).
            Defaults to numeric labels.
        brush_radius: Starting brush radius (mm).
        protection_radius: Radius around protected points where ADD is blocked.

    Returns:
        Refined boolean mask of shape (n_vertices,). Returns the input mask
        unchanged if the user closes the window without picking.
    """
    import pyvista as pv

    from cedalion.dataclasses import VTKSurface

    verts = np.asarray(surface.mesh.vertices)
    refined_mask = mask.copy()
    if protected_labels is None:
        protected_labels = [f"P{i}" for i in range(len(protected_positions))]

    click_count = [0]
    brush_radius_ref = [float(brush_radius)]
    mode = ["add"]

    pvplt = pv.Plotter(notebook=False)

    vtk_surface = VTKSurface.from_trimeshsurface(surface)
    pv_mesh = pv.wrap(vtk_surface.mesh)
    pv_mesh["mask"] = refined_mask.astype(float)

    def update_title():
        pvplt.add_text(
            f"Mode: {mode[0].upper()}  |  Brush: {brush_radius_ref[0]:.0f}mm\n"
            "T = toggle add/remove  |  +/- = brush size",
            position="lower_right",
            font_size=14,
            name="status",
            color="black",
        )

    def on_pick(picked_point):
        if picked_point is None:
            return
        point = np.array(picked_point)
        dists = np.linalg.norm(verts - point, axis=1)
        circle = dists < brush_radius_ref[0]

        if mode[0] == "add":
            dist_to_protected = np.linalg.norm(protected_positions - point, axis=1)
            if np.any(dist_to_protected < protection_radius):
                print("  Skipped -- too close to a protected landmark")
                return
            new_verts = circle & ~refined_mask
            n_changed = int(new_verts.sum())
            if n_changed == 0:
                return
            refined_mask[new_verts] = True
            pvplt.add_mesh(
                pv.Sphere(radius=brush_radius_ref[0], center=point),
                color="yellow",
                opacity=0.2,
            )
        else:
            removed = circle & refined_mask
            n_changed = int(removed.sum())
            if n_changed == 0:
                return
            refined_mask[removed] = False
            pvplt.add_mesh(
                pv.Sphere(radius=brush_radius_ref[0], center=point),
                color="cyan",
                opacity=0.2,
            )

        click_count[0] += 1
        pv_mesh["mask"] = refined_mask.astype(float)
        sign = "+" if mode[0] == "add" else "-"
        print(
            f"  Click {click_count[0]} ({mode[0]}): {sign}{n_changed:,} vertices "
            f"(total: {int(refined_mask.sum()):,})"
        )

    def toggle_mode():
        mode[0] = "remove" if mode[0] == "add" else "add"
        update_title()
        print(f"  Mode: {mode[0].upper()}")

    def increase_radius():
        brush_radius_ref[0] = min(brush_radius_ref[0] + 5, 100)
        update_title()
        print(f"  Brush radius: {brush_radius_ref[0]:.0f}mm")

    def decrease_radius():
        brush_radius_ref[0] = max(brush_radius_ref[0] - 5, 5)
        update_title()
        print(f"  Brush radius: {brush_radius_ref[0]:.0f}mm")

    pvplt.add_mesh(
        pv_mesh,
        scalars="mask",
        cmap=["white", "red"],
        clim=[0, 1],
        show_scalar_bar=False,
        opacity=0.9,
        smooth_shading=True,
        pickable=True,
    )
    for label, pos in zip(protected_labels, protected_positions):
        pvplt.add_mesh(pv.Sphere(radius=3, center=pos), color="lime")
        pvplt.add_point_labels(
            [pos], [label], font_size=12, point_size=0,
            text_color="lime", shape=None, always_visible=True,
        )
    pvplt.enable_surface_point_picking(
        callback=on_pick, left_clicking=True, show_point=False, tolerance=0.005,
    )
    pvplt.add_key_event("t", toggle_mode)
    pvplt.add_key_event("plus", increase_radius)
    pvplt.add_key_event("minus", decrease_radius)
    update_title()
    pvplt.show()

    logger.debug(
        f"Refinement: {click_count[0]} clicks, final mask: {int(refined_mask.sum())} vertices"
    )
    return refined_mask
