""" ...
"""

# Import packages
import numpy as np
import pandas as pd
import os
import xarray as xr
import pyvista as pv
from pathlib import Path
import matplotlib.pyplot as plt
import trimesh
import nibabel as nib
import yaml
import sys

import mne

import cedalion
from cedalion.sim import synthetic_hrf
from cedalion import units
import cedalion.sim.forward_model_local as fw  # local version of forward_model, to allow modifications
import cedalion.dataclasses as cdc
from cedalion.dataclasses import PointType
import cedalion.geometry.landmarks
import cedalion.sim.synthetic_hrf as synhrf
import cedalion.models.glm as glm
import cedalion.dot as dot
import pint
import abc

import cedalion.vis.blocks as vbx
import cedalion.vis.anatomy
import cedalion.vis.anatomy.sensitivity_matrix

from cedalion.sim.utils_plots import plot_activation, plot_parcellated_surface
from cedalion.sim.source_simulator import SourcesConfig

class SimulatefNIRSAndEEGData():
    """ Simulate fNIRS and EEG data from specified sources and head model on top of real background data.    
    """

    def __init__(self, 
                 subject,
                 subjects_dir, 
                 verbose=True,
                 seed=None):
        
        self.subject = subject
        self.subjects_dir = subjects_dir
        self.subject_dir = Path(subjects_dir) / subject
        self.verbose = verbose

        # Define plot formats for stimuli and sources
        self.cmap = plt.get_cmap('tab10')
        self.fmt = {'rest': {'ec': 'black', 'fc': 'gray', 'alpha': 0.1},}

        # Check subject directory exists and is not empty
        if not os.path.isdir(self.subject_dir):
            raise FileNotFoundError(f"Subject directory {self.subject_dir} not found.")
        if len(os.listdir(self.subject_dir)) == 0:
            raise ValueError(f"Subject directory {self.subject_dir} is empty.")

        self.config_file = None
        self.seed = seed if seed is not None else np.random.randint(0, 1e6)

        # Backup current stdout (the Jupyter OutStream) Because some libraries mess with it (Report to Eike)
        self._original_stdout = sys.stdout

        if self.verbose:
            print(f"Random seed set to {self.seed}.")
            print(f"Initialized SimulatefNIRSAndEEGData for subject {self.subject} in {self.subject_dir}.")

    def load_background_data(self, 
                             fnirs_file, 
                             eeg_file,
                             rest_window=None,):
        """Load fNIRS and EEG background data from snirf and fif files.
        
        It defines resting state period from rest_window (list or tuple of two elements: [T_rest_start, T_rest_end]).
        If rest_window is not provided, set to full duration of recordings.
        """

        # --- fNIRS and EEG background data ---
        if self.verbose:
            print("\n--- Loading FNIRS and EEG background data ---")
        # Read fNIRS and EEG background data
        self.fnirs_bg_rec = _read_fnirs(snirf_file=fnirs_file)
        self.fnirs_bg_rec['od'] = cedalion.nirs.cw.int2od(self.fnirs_bg_rec['amp'])
        self.eeg_bg_raw = _read_eeg(eeg_file=eeg_file)
        
        # Check modalities are time-aligned.
        fnirs_t_lims = (self.fnirs_bg_rec['amp'].time.values[0], self.fnirs_bg_rec['amp'].time.values[-1])
        eeg_t_lims = (self.eeg_bg_raw.times[0], self.eeg_bg_raw.times[-1])
        if self.verbose:
            print(f"fNIRS background data time range: {fnirs_t_lims[0]:.2f} s to {fnirs_t_lims[1]:.2f} s")
            print(f"EEG background data time range: {eeg_t_lims[0]:.2f} s to {eeg_t_lims[1]:.2f} s")
        if fnirs_t_lims[0] > eeg_t_lims[1] or eeg_t_lims[0] > fnirs_t_lims[1]:
            raise ValueError("fNIRS and EEG background data time ranges do not overlap.")


        if self.verbose:
            print("--- fNIRS and EEG background data loaded Successfully ---")

        # --- Resting state period ---
        if self.verbose:
            print("\n--- Defining resting state period ---")

        # Define resting state period from rest_stim
        if rest_window is None:
            if self.verbose:
                print("No rest window provided, setting to full duration of recordings...")
            T_rest_start = 0
            T_rest_end = np.min([fnirs_t_lims[1], eeg_t_lims[1]])
            
        else:
            if not isinstance(rest_window, (list, tuple)) or len(rest_window) != 2:
                raise ValueError("rest_window must be a list or tuple of two elements: [T_rest_start, T_rest_end]")
            T_rest_start, T_rest_end = rest_window
            if T_rest_start < 0 or T_rest_end <= T_rest_start:
                raise ValueError("Invalid rest_window values. Ensure that 0 <= T_rest_start < T_rest_end.")
            if T_rest_end > np.min([fnirs_t_lims[1], eeg_t_lims[1]]):
                T_rest_end = np.min([fnirs_t_lims[1], eeg_t_lims[1]])
                print(f"Warning: T_rest_end exceeds recording duration, setting to {T_rest_end:.2f} s.")
            if T_rest_start >= np.min([fnirs_t_lims[1], eeg_t_lims[1]]):
                raise ValueError("T_rest_start exceeds recording duration.")
            if T_rest_start < np.max([fnirs_t_lims[0], eeg_t_lims[0]]):
                T_rest_start = 0
                print("Warning: T_rest_start is before recording start, setting to 0 s.")
            
        
        rest_stim = pd.DataFrame({'onset': [T_rest_start], 
                                  'duration': [T_rest_end - T_rest_start],
                                  'value': [1],
                                  'trial_type': ['rest']})

        self.fnirs_bg_rec.stim = rest_stim
        self.eeg_bg_raw.set_annotations(mne.Annotations(onset=rest_stim.onset.values,
                                                    duration=rest_stim.duration.values,
                                                    description=rest_stim.trial_type.values))

        self.rest_stim = rest_stim
        self.T_rest_start = T_rest_start * units.seconds
        self.T_rest_end = T_rest_end * units.seconds

        if self.verbose:
            print(f"-- Resting state period defined from {self.T_rest_start} to {self.T_rest_end} --")

        # Stored validated variables
        self.fnirs_file = fnirs_file
        self.eeg_file = eeg_file
        self.meas_list = self.fnirs_bg_rec._measurement_lists['amp']

    def build_headmodel(
            self,
            head_model_name: 'str' = 'colin27',
            segmentation_dir = None,
            mask_files: 'dict[str, str]' = {'csf': 'csf.nii', 
                                            'gm': 'gm.nii', 
                                            'scalp': 'scalp.nii', 
                                            'skull': 'skull.nii', 
                                            'wm': 'wm.nii'},
            brain_surface_file: 'str' = None,
            scalp_surface_file: 'str' = None,
            landmarks_ras_file: 'Path | str | None' = None,
            brain_seg_types: 'list[str]' = ['gm', 'wm'],
            scalp_seg_types: 'list[str]' = ['scalp'],
            smoothing: 'float' = 0.5,
            brain_face_count: 'int | None' = 180000,
            scalp_face_count: 'int | None' = 60000,
            fill_holes: 'bool' = False,
            parcel_file: 'Path | str | None' = None
            ):
        """ Build head model from surface meshes, and fNIRS and EEG montages.

        Two-surface head model is built from surfaces in voxel space, then transformed to RAS space.
        The full 10-10 landmarks are built from the 5 fiducials, that should be included in the landmarks_ras_file,
        and assigned to both head models.
        """

        # --- Build headmodel ---
        if self.verbose:
            print("\n--- Building head model ---")

        # Load head model from cedalion datasets
        if head_model_name.lower() in ['colin27']:
            print(f"Building head model from {head_model_name}...")
            # Load Atlas-based head model and montage from cedalion
            head_ijk = dot.get_standard_headmodel(head_model_name.lower())
            parcel_file = cedalion.data.get_colin27_parcel_file()

            # Set input parameters to None since not used
            segmentation_dir = None
            mask_files = None
            brain_surface_file = None
            scalp_surface_file = None
            landmarks_ras_file = None
            brain_seg_types = None
            scalp_seg_types = None
            smoothing = None
            fill_holes = None

        elif head_model_name.lower() in ['icbm152']:
            print(f"Building head model from {head_model_name}...")
            # Load Atlas-based head model and montage from cedalion
            head_ijk = dot.get_standard_headmodel(head_model_name.lower())
            parcel_file = cedalion.data.get_icbm152_parcel_file()

            # Set input parameters to None since not used
            segmentation_dir = None
            mask_files = None
            brain_surface_file = None
            scalp_surface_file = None
            landmarks_ras_file = None
            brain_seg_types = None
            scalp_seg_types = None
            smoothing = None
            fill_holes = None

        # Build head model from individual MRI
        elif head_model_name == 'individual':
            if self.verbose:
                print("Building head model from individual MRI...")
            # Validate input directories
            if not segmentation_dir or not os.path.isdir(segmentation_dir):
                raise ValueError("segmentation_dir must be provided and must be a valid directory when head_model_name is 'individual'")
            if brain_surface_file and not os.path.isfile(brain_surface_file):
                brain_surface_file = None
                print("Warning: brain_surface_file not found, will be generated from segmentation masks.")
            if scalp_surface_file and not os.path.isfile(scalp_surface_file):
                scalp_surface_file = None
                print("Warning: scalp_surface_file not found, will be generated from segmentation masks.")
            if not segmentation_dir / landmarks_ras_file or not os.path.isfile(segmentation_dir / landmarks_ras_file):
                raise ValueError("landmarks_ras_file must be provided and must be a valid file when head_model_name is 'individual'")
            if parcel_file and not os.path.isfile(parcel_file):
                raise ValueError("parcel_file must be a valid file when provided")
            
            # Build two-surface headmodel from surfaces, in voxel space
            head_ijk = dot.TwoSurfaceHeadModel.from_surfaces(
                    segmentation_dir=segmentation_dir,
                    mask_files=mask_files,
                    brain_surface_file= brain_surface_file,
                    scalp_surface_file= scalp_surface_file,
                    landmarks_ras_file=landmarks_ras_file,
                    brain_seg_types=brain_seg_types,
                    scalp_seg_types=scalp_seg_types,
                    brain_face_count=brain_face_count,
                    scalp_face_count=scalp_face_count,
                    smoothing=smoothing,
                    fill_holes=fill_holes,
                    parcel_file=parcel_file
                )

        else:
            raise ValueError("head_model_name must be either 'colin27' or 'individual'")
        
        # Log number of faces and vertices
        if self.verbose:
            print(f"Number of brain faces and vertices = ({head_ijk.brain.nfaces}, {head_ijk.brain.nvertices})")
            print(f"Number of scalp faces and vertices = ({head_ijk.scalp.nfaces}, {head_ijk.scalp.nvertices})")

        # Change to subject (RAS) space by applying an affine transformation on the head model
        if self.verbose:
            print("Transforming head model to RAS space...")
        head_ras = head_ijk.apply_transform(head_ijk.t_ijk2ras)

        # Build complete 10-10 landmarks from the 5 fiducials
        if self.verbose:
            print("Building full 10-10 landmarks from fiducials...")
        landmark_label_map = {'nz': 'Nz', 'lpa': 'LPA', 'rpa': 'RPA', 'iz': 'Iz', 'cz': 'Cz'}  # Map to label convention used in cedalion
        landmarks_ras = [landmark_label_map.get(l.lower()) for l in head_ras.landmarks.label.values]
        landmarks_ras = head_ras.landmarks.assign_coords({'label': landmarks_ras})
        lmbuilder = cedalion.geometry.landmarks.LandmarksBuilder1010(head_ras.scalp, landmarks_ras)
        all_landmarks_ras = lmbuilder.build()
        all_landmarks_ijk = all_landmarks_ras.points.apply_transform(head_ijk.t_ras2ijk)
        # Update landmarks for both headmodels
        head_ras.landmarks = all_landmarks_ras
        head_ijk.landmarks = all_landmarks_ijk
        
        self.head_ijk = head_ijk
        self.head_ras = head_ras
        self.brain_face_count = head_ras.brain.nfaces
        self.scalp_face_count = head_ras.scalp.nfaces
        self.parcels = cedalion.io.read_parcellations(parcel_file) if parcel_file else None

        if self.verbose:
            print("--- Head model built successfully ---")

        # Store parameters
        self.head_model_name = head_model_name
        self.smoothing = smoothing
        self.fill_holes = fill_holes
        self.segmentation_dir = segmentation_dir
        self.mask_files = mask_files
        self.brain_surface_file = brain_surface_file
        self.scalp_surface_file = scalp_surface_file
        self.landmarks_ras_file = landmarks_ras_file
        self.parcel_file = parcel_file

    def build_montage(self, 
                      montage_file=None,
                      eeg_montage_name='standard_1020'):
        """ Build fNIRS-EEG montage.

        If montage_file is provided, load montage from montage_file (tsv format). If not provided, try to get fNIRS montage from SNIRF file.
        If no EEG electrodes found in montage, build EEG montage from MNE built-in montage specified by eeg_montage_name.
        Then, the two montages are combined, avoiding duplicates. The resulting combined montage should have all fNIRS optodes 
        (as PointType.SOURCE, and PointType.DETECTOR types), all EEG electrodes (as PointType.ELECTRODE), 
        and at least 3 fiducials (as PointType.LANDMARK), e.g. ['Nz', 'Iz', 'LPA', 'RPA']. Cz is included as a fiducial,
        only if it was not already present in the EEG montage as an electrode. 
        The montage is transformed and snapped to scalp surface to obtain self.montage_ras (RAS) and self.montage_ijk (Voxel).
        """

        # --- Montage ---
        if self.verbose:
            print("\n--- Building fNIRS and EEG montage ---")

        if hasattr(self, 'head_ras') is False:
            raise ValueError("Head model must be built before building montage. Please run build_headmodel() first.")

        # Try to read montage from montage_file
        if montage_file:
            if self.verbose:
                print(f"Loading montage file {montage_file}...")
            if not os.path.isfile(montage_file):
                raise FileNotFoundError(f"Montage file {montage_file} not found.")            
            montage = cedalion.io.probe_geometry.load_tsv(montage_file)
        # Try to get montage from SNIRF file
        else:  
            if self.verbose:
                print("Montage file not provided, using fNIRS montage from SNIRF file.")
            montage = self.fnirs_bg_rec.geo3d
            if montage is None:
                raise ValueError("fNIRS montage not found in SNIRF file. Please provide a valid fNIRS montage file.")

        # Validate montage: Landmarks
        landmarks = montage.label[montage.type == PointType.LANDMARK]
        if len(landmarks) < 3:
            raise ValueError(f"Montage must contain at least 3 landmarks, found {len(landmarks)}. Please provide a valid montage file.")
        
        # Validate montage: Optodes
        optodes_montage = set(montage.label[montage.type.isin([PointType.SOURCE, PointType.DETECTOR])].values)
        optodes_ts = set(np.append(self.fnirs_bg_rec['amp'].source.values, self.fnirs_bg_rec['amp'].detector.values))
        if optodes_montage != optodes_ts:
            raise ValueError("fNIRS montage optodes do not match fNIRS time series optodes. Please provide a valid Montage file.")
        
        # Set EEG montage profile from MNE built-in montages
        if eeg_montage_name not in mne.channels.get_builtin_montages():
            raise ValueError(f"EEG montage {eeg_montage_name} not found in MNE built-in montages. Available montages are: {mne.channels.get_builtin_montages()}")
        self.eeg_bg_raw.set_montage(eeg_montage_name)

        # Identify coordinate system
        crs = montage.transpose('label', ...).dims[-1]
        if self.verbose:
            print(f"Found montage coordiante system to be '{crs}'.")

        # Transform montage to RAS space and snap to head scalp surface
        if self.verbose:
            print("Transforming montage to RAS space and snapping to scalp surface...")
        montage_ras = self.head_ras.align_and_snap_to_scalp(montage)

        electrodes = montage_ras.label[montage.type == PointType.ELECTRODE]
        # If no EEG electrodes in montage, build from raw data
        if len(electrodes) == 0:
            if self.verbose:
                print(f"No EEG electrodes found in montage file, building EEG montage from MNE built-in montage: {eeg_montage_name}...")
            eeg_montage = _build_eeg_montage(self.eeg_bg_raw) # Build (xr) montage from raw data
            eeg_montage_ras = self.head_ras.align_and_snap_to_scalp(eeg_montage)
            
            # Combine fNIRS and EEG montage, avoiding duplicates
            eeg_montage_ras = eeg_montage_ras[eeg_montage_ras.type != PointType.LANDMARK]  # Remove landmarks to avoid duplicates
            # If Cz is labeled as an EEG electrode, remove it from fnirs montage landmarks
            if ('Cz' in eeg_montage_ras.label) and ('Cz' in landmarks):
                montage_ras = montage_ras.drop_sel({'label': 'Cz'})
            montage_ras = xr.concat([montage_ras, eeg_montage_ras], dim='label')

        # Try to read EEG montage geometry from montage_file
        else:
            # Validate EEG electrodes in montage
            electodes_montage = set(self.eeg_bg_raw.copy().pick('eeg').ch_names)
            electrodes_ts = set(electrodes.values)
            if electodes_montage != electrodes_ts:
                raise ValueError("EEG montage electrodes do not match EEG time series electrodes. " \
                "Please provide a valid montage file including EEG electrodes or remove them altogether to build them from MNE built-in montage.")

        self.montage_ras = montage_ras
        self.fnirs_bg_rec.geo3d = montage_ras  # Update fNIRS background record montage
        self.montage_ijk = self.montage_ras.points.apply_transform(self.head_ijk.t_ras2ijk)  # Build also montage in voxel space
        
        # Store montage-specific parameters
        self.montage_file = montage_file
        self.eeg_montage_name = eeg_montage_name
        self.ch_fnirs = self.fnirs_bg_rec['amp'].channel.values
        self.ch_electrodes = self.eeg_bg_raw.ch_names
        self.ch_eeg = self.eeg_bg_raw.copy().pick('eeg').ch_names
        self.nch_fnirs = len(self.ch_fnirs)
        self.nch_electrodes = len(self.ch_electrodes)
        self.nch_eeg = len(self.ch_eeg)

        if self.verbose:
            print("--- fNIRS and EEG montage loaded/builded successfully ---")

    def build_sensitivity(self, sensitivity_fname=None):
        """ Build or load sensitivity matrix for fNIRS channels and head model.
        
        If sensitivity_filename is provided, load sensitivity matrix from file. 
        Otherwise, build sensitivity matrix from scratch using Cedalion's wrapper around nirfaster, 
        and save fluence and sensitivity matrix to file.
        """

        if sensitivity_fname:
            if not os.path.isfile(sensitivity_fname):
                raise FileNotFoundError(f"Sensitivity file {sensitivity_fname} not found.")
        else:

            # Fluence and sensitivity filenames
            fluence_fname = f'fluence_HM{self.head_model_name}_fbrain{self.brain_face_count}_fscalp{self.scalp_face_count}_fNIRSch{self.nch_fnirs}_EEGch{self.nch_eeg}.hdf5'
            sensitivity_fname = fluence_fname.replace('fluence', 'sensitivity')
            fluence_fname =  self.subject_dir / fluence_fname
            sensitivity_fname = self.subject_dir / sensitivity_fname

            if self.verbose:
               print("\n--- Building sensitivity matrix from scratch ---")
            
            # Compute fluence and sensitivity matrix and save to file
            fwm = cedalion.dot.ForwardModel(self.head_ijk, self.montage_ijk, self.meas_list)
            fwm.compute_fluence_nirfaster(fluence_fname=fluence_fname)
            fwm.compute_sensitivity(fluence_fname=fluence_fname, 
                                    sensitivity_fname=sensitivity_fname)
            if self.verbose:
                print(f"--- Sensitivity matrix built and saved in {sensitivity_fname} ---")

        if self.verbose:
            print(f"\n--- Loading sensitivity matrix from {sensitivity_fname}. ---")
        self.sensitivity = cedalion.io.forward_model.load_Adot(sensitivity_fname)

        # Add units if not present
        if self.sensitivity.pint.units is None:
            if self.sensitivity.units is not None:
                self.sensitivity = self.sensitivity * pint.Unit(self.sensitivity.units)
            else:
                print('Warning, no units found for sensitivity, setting to mm')
                self.sensitivity = self.sensitivity * units.mm

        if self.verbose:
            print("--- Sensitivity matrix loaded ---")

    def build_leadfield(self, 
                        eeg_forward_sol_fname=None,
                        conductivity=(0.3, 0.006, 0.3), 
                        overwrite=False):
        """ Build or load EEG leadfield.
        
        If eeg_forward_sol_fname is provided, load EEG forward solution from file and 
        BEM surfaces from subject's bem folder. If not provided, build EEG forward solution 
        and BEM surfaces from scratch using MNE-Python, and save EEG forward solution and BEM surfaces to file.

        The BEM surfaces are built from the segmentation masks in the head model. 
        These are the same ones used for generating the volume mesh during the sensitivity calculation. 
        For the forward solution, the source surface is defined as the head model’s brain surface, 
        i.e. dipole locations correspond to vertices in the brain mesh.

        In either case, the EEG forward solution is converted to fixed orientation (normal to cortex),
        and the leadfield is extracted and stored as self.leadfield. The BEM surfaces are stored as
        self.bem_trimesh (Cedalion TrimeshSurface objects).

        """

        self.conductivity = conductivity

        # Define EEG source as dipoles on brain surface (~source surface)
        dipole_locations = self.head_ijk.brain.vertices
        dipole_orientations = self.head_ijk.brain.get_vertex_normals(dipole_locations, 
                                                                     normalized=True)

        # Build or load EEG forward solution
        if eeg_forward_sol_fname is None:
            
            if self.verbose:
                print("\n--- Building EEG leadfield from scratch ---")

            # Create EEG forward model
            fwm_eeg = fw.ForwardModelEEG(self.head_ijk, 
                                        self.montage_ijk, 
                                        dipole_locations=dipole_locations.pint.dequantify().values,
                                        dipole_orientations=dipole_orientations.pint.dequantify().values)

            # Generate BEM mesh
            fwm_eeg.generate_BEM_mesh()

            # Compute leadfields
            fwm_eeg.compute_leadfields_BEM(conductivity=conductivity)
            eeg_fwd = fwm_eeg.fwd
            bem_mesh = fwm_eeg.bem_mesh

            if self.verbose:
                print("--- EEG leadfield built successfully ---")

            # Save BEM meshes in mne/freesurfer format
            if self.verbose:
                print(f"Saving BEM meshes to {self.subject_dir / 'bem'}...")

            if not os.path.isdir(self.subject_dir / 'bem'):
                os.makedirs(self.subject_dir / 'bem')
            names = {'inner_skull': 'inner_skull.surf',
                    'outer_skull': 'outer_skull.surf',
                    'outer_skin': 'outer_skin.surf'}
            
            for k, surf in bem_mesh.items():
                name = names[k]
                pos, tri = surf  # pos are in mm
                # Use nibabel to write the surface
                nib.freesurfer.io.write_geometry(Path(self.subject_dir, 'bem', name), pos, tri)

            # Save EEG forward solution to file
            eeg_forward_sol_fname = f'EEG_fwd_HM{self.head_model_name}_fbrain{self.brain_face_count}_fscalp{self.scalp_face_count}_fNIRSch{self.nch_fnirs}_EEGch{self.nch_eeg}_fwd.fif'
            eeg_forward_sol_fname = self.subject_dir / eeg_forward_sol_fname

            if self.verbose:
                print(f"Saving EEG forward solution to {eeg_forward_sol_fname}...")

            # Save the canonical/free version (i.e. prior to conversion to fixed orientation)!
            mne.write_forward_solution(eeg_forward_sol_fname, eeg_fwd, overwrite=overwrite)

        # Load EEG forward solution and BEM meshes from file
        else:
            if self.verbose:
                print(f"\n--- Loading EEG forward solution from {eeg_forward_sol_fname} ---")
            
            # Load EEG forward solution from file
            if not os.path.isfile(eeg_forward_sol_fname):
                raise FileNotFoundError(f"EEG forward solution file {eeg_forward_sol_fname} not found.")
            
            eeg_fwd = mne.read_forward_solution(eeg_forward_sol_fname, verbose=self.verbose)

            # Load BEM surfaces from subject's bem folder.
            if self.verbose:
                print(f"\n --- Loading BEM meshes from {self.subject_dir / 'bem'} ---")

            bem_mesh = {}
            names = {'inner_skull': 'inner_skull.surf',
                     'outer_skull': 'outer_skull.surf',
                     'outer_skin': 'outer_skin.surf'}
            for k, name in names.items():
                surf_path = Path(self.subject_dir, 'bem', name)
                if not os.path.isfile(surf_path):
                    raise FileNotFoundError(f"BEM surface file {surf_path} not found." +
                                            " Please run build_leadfield() without eeg_forward_sol_fname to generate BEM surfaces.")
                pos, tri = mne.surface.read_surface(surf_path)
                bem_mesh[k] = (pos, tri)

            # Validate loaded forward solution is compatible with head model and montage
            if self.verbose:
                print("Validating loaded EEG forward solution...")
            verts_from_fwd = eeg_fwd['src'][0]['vertno']
            verts_from_surf = dipole_locations.label.values[eeg_fwd['src'][0]['inuse'].astype(bool)]
            if set(verts_from_fwd) != set(verts_from_surf):
                raise ValueError("Dipole locations from loaded EEG forward solution do not match head model brain surface vertices.")
            ch_names_from_fwd = eeg_fwd['sol']['row_names']
            if set(ch_names_from_fwd) != set(self.ch_eeg):
                raise ValueError("EEG channel names from loaded EEG forward solution do not match EEG montage electrodes.")

            if self.verbose:
                print("--- EEG forward solution and BEM meshes loaded successfully ---")

        # Convert EEG forward solution to fixed orientation (normal to cortex)
        # surf_ori => rotates each dipoles into the surface coordinate frame (z axis is surface normal).
        # force_fixed=True => collapses 3 components → 1 component per source, aligned with the local surface normal
        # use_cps=True => uses closest-point smoothing to make the normal field less noisy across neighboring vertices
        eeg_fwd = mne.convert_forward_solution(eeg_fwd,
                                         surf_ori=True, force_fixed=True, use_cps=True)


        # Exclude dipoles that did not make it for the source model (outside brain surface)
        dipole_excluded_mask = eeg_fwd['src'][0]['inuse'].astype(bool)
        dipole_locations = dipole_locations[dipole_excluded_mask]
        self.dipole_locations = dipole_locations.rename({'label': 'vertex'}) * units.mm
        dipole_orientations = dipole_orientations[dipole_excluded_mask]
        self.dipole_orientations = dipole_orientations.rename({'label': 'vertex'})
        
        # Extract leadfields (with fixed orientation pointing normal to cortex)
        leadfield = eeg_fwd['sol']['data'].astype(np.float32)
        # Get channel names (can be different from input electrode names)
        ch_names = np.array(eeg_fwd['sol']['row_names'])
        # Wrap in xarray (vertex index matches dipole_locations)
        leadfield = xr.DataArray(data=leadfield,
                                  dims=['channel', 'vertex'],
                                  coords={'channel': ch_names,
                                          'vertex': self.dipole_locations.vertex.values}
                                          )
        if 'parcel' in self.dipole_locations.coords:
            leadfield = leadfield.assign_coords({'parcel': ('vertex', self.dipole_locations.parcel.values)})
        
        # Rescale values to bring them to unit range
        leadfield /= np.abs(leadfield.values).max()

        # Add units
        leadfield = leadfield * (units.uV / (units.nA * units.m)) # Leadfield units: uV / (nA m)

        # Build Cedalion TrimeshSurface for each BEM mesh
        process = False
        bem_trimesh = {}
        for k, (verts, faces) in bem_mesh.items():
            surf_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=process)
            # Rearange face edges so all normals pint to the same direction (inwards or outwards)
            trimesh.repair.fix_winding(surf_mesh)
            trimesh.repair.fix_normals(surf_mesh)
            # Convert to Cedalion TrimeshSurface
            surf_mesh = cdc.TrimeshSurface(surf_mesh, 'ijk', cedalion.units.Unit("1"))
            # Flip normals so they all point away from the center of gravity (typically outside)
            surf_mesh.fix_vertex_normals()
            bem_trimesh[k] = surf_mesh
        
        
        self.eeg_fwd = eeg_fwd
        self.leadfield = leadfield
        self.bem_trimesh = bem_trimesh
        
    def init_source_config(self, 
                           names: 'list[str]',
                           spatial_amps: 'list[float] | None' = None,
                           temporal_amps: 'list[float] | None' = None,
                           trial_types: 'list[str] | None' = None):
        
        """ Initialize SourcesConfig object from a list of source names, and optional spatial and temporal amplitudes.
        
        The source configuration object is stored as self.source_cfg, and the shared spatial and temporal amplitudes
        are stored as self.shared_spatial_amps and self.shared_temporal_amps dictionaries, respectively. These will be used
        later when building the individual source spatial and temporal profiles.
        """

        if self.verbose:
            print("\n--- Initializing source configuration ---")
        
         # Initialize source configuration
        source_cfg = SourcesConfig()
        source_cfg.new(names=names)

        self.source_cfg = source_cfg
        self.Ns = len(self.source_cfg)
        
        # Store shared spatial and temporal amplitudes for later use
        if spatial_amps is None:
            spatial_amps = [1.0] * self.Ns
        # single value for all sources
        elif isinstance(spatial_amps, (int, float)):
            spatial_amps = [spatial_amps] * self.Ns
        elif isinstance(spatial_amps, (list, np.ndarray, tuple)) and len(spatial_amps) == self.Ns:
            if any(a < 0 or a > 1 for a in spatial_amps):
                raise ValueError("spatial_amps must be in the range [0, 1].")
        else:
            raise ValueError("spatial_amps must be a list, array, or tuple of length equal to number of sources.")
        
        self.shared_spatial_amps = {src.name: amp for src, amp in zip(self.source_cfg, spatial_amps)}

        if temporal_amps is None:
            temporal_amps = [1.0] * self.Ns
        # single value for all sources
        elif isinstance(temporal_amps, (int, float)):
            temporal_amps = [temporal_amps] * self.Ns
        elif isinstance(temporal_amps, (list, np.ndarray, tuple)) and len(temporal_amps) == self.Ns:
            if any(a < 0 for a in temporal_amps):
                raise ValueError("temporal_amps must be non-negative.")
        else:
            raise ValueError("temporal_amps must be a list, array, or tuple of length equal to number of sources.")

        self.shared_temporal_amps = {src.name: amp for src, amp in zip(self.source_cfg, temporal_amps)}

        # Set trial types if provided
        if trial_types is not None:
            if isinstance(trial_types, (list, np.ndarray, tuple)) and len(trial_types) == self.Ns:
                trial_source_map ={}
                for tt, s in zip(trial_types, names):
                    tt = tt if isinstance(tt, list) else [tt]
                    for t in tt:
                        if isinstance(t, str) is False:
                            raise ValueError("Each trial type must be a string.")
                        elif t not in trial_source_map:
                            trial_source_map[t] = [s]
                        else:
                            trial_source_map[t].append(s)
            # Single trial type for all sources
            elif isinstance(trial_types, str):
                trial_source_map = {trial_types: s.name for s in self.source_cfg}
            else:
                raise ValueError("trial_types must be a list, array, or tuple of length equal to number of sources.")
        else:
            if self.verbose:
                print("No trial_types provided, using all sources in source_cfg as trial types...")
            trial_source_map = {src.name: [src.name] for src in self.source_cfg}

        self.trial_source_map = trial_source_map

        if self.verbose:
            print(f"--- Source configuration initialized with {self.Ns} sources ---")

    def build_shared_source_spatial_profile(self):
        """ Build shared source spatial profiles by mapping source locations to brain vertices.
        
        The profiles defined here are common to both modalities. 
        This is the way in which we incorporate spatial (Where) co-modulation.
        The source locations are defined in the source configuration object (self.source_cfg).
        The source locations can be defined either by parcel name or landmark name.
        The mapped vertices are stored in the source configuration object.
        """

        if self.verbose:
            print("\n--- Building shared source spatial profiles ---")

        # Check source configuration has been initialized
        if hasattr(self, 'source_cfg') is False:
            raise ValueError("Source configuration not initialized. Please run init_source_config() first.")

        # Map source location to brain vertices 
        all_vertices = self.head_ijk.brain.vertices        
        for i, source in enumerate(self.source_cfg):
            
            # Validate spatial profile
            if source.spatial is None:
                raise ValueError(f"Source spatial profile not defined for source '{source.name}'. Define it via self.source_cfg.set_spatial().")
            if source.spatial.vertices is not None:
                print(f"Warning: Source spatial vertices already set for source '{source.name}', overwriting them...")
            
            src_name = source.name
            loc = source.spatial.loc
            selector = source.spatial.selector

            # Add shared amp information to source spatial profile
            if source.spatial.amp in [None, 1.0]:
                if self.verbose:
                    print(f"Setting shared spatial amp for source '{src_name}' to {self.shared_spatial_amps[src_name]}...")
                source.spatial.amp = self.shared_spatial_amps[src_name]
            else:
                raise ValueError(f"Found spatial amp already set for source '{src_name}', please remove it to use shared spatial amps." +
                                 "Shared amplitudes must be set via init_source_config().")

            if  selector == 'parcel':
                
                # Validate parcel information
                if self.parcel_file is None:
                    raise ValueError("No parcellation file provided in head model, cannot select source by parcel.")

                h = source.spatial.hemi
                
                # Normalize parcel name
                loc = loc.strip().lower()                    
                # Look for vertices matching src name (and hemisphere)
                if h is not None:
                    # Assume hemisphere is indicated by suffix _lh or _rh somewhere in the parcel name
                    h = '_' + _norm_hemi(h)  
                    parcel_mask = [(loc in l.strip().lower()) and (h in l.strip().lower()) for l in all_vertices.parcel.values]
                else:
                    parcel_mask = [loc in l.strip().lower() for l in all_vertices.parcel.values]
            
                vertices = all_vertices.label[parcel_mask].values.tolist()
                if len(vertices) == 0:
                    print(f"Warning: No vertices found for parcel '{loc}'" + (f" in hemisphere '{h}'." if h is not None else "."))
            
            elif selector == 'landmark':
                
                if loc not in self.head_ijk.landmarks.label.values:
                    raise ValueError(f"Landmark '{loc}' must be one of the landmarks in the head model: {self.head_ijk.landmarks.label.values}")

                source_pos = self.head_ijk.landmarks.sel(label=loc)
                vertices = self.head_ijk.brain.mesh.kdtree.query(source_pos.pint.dequantify())[1]
                vertices = [int(vertices)]  # Wrap in list

            # Update spatial profile with mapped vertices
            source.spatial.set_vertices(vertices)
            
            # Assign colors for plotting
            self.fmt[src_name] = {'ec': None, 'fc': self.cmap(i), 'alpha': 0.3}

            if self.verbose:
                print(f"Mapped source location '{src_name}' to brain vertices, n_vertices = {len(vertices)}.")
        
        if self.verbose:
            print("--- Shared source spatial profiles built successfully ---")
        
    def build_fnirs_source_spatial_profile(self):
        """ Build fNIRS-specific source spatial profiles from source configuration.

        The fNIRS source spatial profiles are built using Cedalion's synhrf.build_spatial_activation() function,
        which creates a spatial activation pattern on the brain surface based on the source vertices, spatial scale,
        and amplitude defined in the source configuration object (self.source_cfg). Contrary to
        synhrf.build_spatial_activation() original purpose, activation here is assumed to be shared between HbO and HbR,
        and dimensionless, i.e. representing relative amplitude changes only. Therefore, the resulting spatial profiles
        stored in self.s_fnirs_spatial have dimensions (vertex, source).
        """

        # Build spatial profiles
        print("\n---- Building fNIRS spatial profiles ---")

        if hasattr(self, 'source_cfg') is False:
            raise ValueError("Source configuration not initialized. Please run init_source_config() first.")

        # Initialize spatial profiles
        self.s_fnirs_spatial = None

        for i, source in enumerate(self.source_cfg):
            
            # validate spatial profile
            if source.spatial.vertices is None:
                raise ValueError(f"Source spatial vertices not set for source '{source.name}'. Please run build_shared_source_spatial_profile() first.")

            # Use Cedalion functionality to build fNIRS spatial activation representing HbO and HbR concentration changes
            activity = synhrf.build_spatial_activation(
                self.head_ras.brain,
                source.spatial.vertices,
                spatial_scale= source.spatial.scale * units.cm,
                intensity_scale= 1 * units.micromolar,  # Units will be removed later
                hbr_scale= 1  # We don't use HbR changes here
            )

            # Add source dimension
            activity = activity.expand_dims('source', 2).assign_coords({'source': [source.name]})

            # Little hack to bring activity to a dimensionless spatial distribution, common to HbO and HbR
            # Pick HbO and remove dimensions
            activity = activity.sel(chromo='HbO').drop_vars('chromo')
            activity = activity.pint.dequantify()
            activity.attrs = {}
            # Remove current scale, and replace it for relative amplitudes
            activity = activity / activity.max(dim='vertex') * source.spatial.amp

            if i==0:
                self.s_fnirs_spatial = activity
            else:
                self.s_fnirs_spatial = xr.concat([self.s_fnirs_spatial, activity], dim='source')

            if self.verbose:
                print(f"fNIRS spatial profile built for source '{source.name}', with scale {source.spatial.scale} and amplitude {source.spatial.amp}.")

        # Add vertex and parcel coordinates to spatial profiles
        self.s_fnirs_spatial = self.s_fnirs_spatial.assign_coords(vertex=self.head_ijk.brain.vertices.label.values)
        if 'parcel' in self.head_ijk.brain.vertices.coords:
            self.s_fnirs_spatial = self.s_fnirs_spatial.assign_coords({'parcel': ('vertex', self.head_ijk.brain.vertices.sel(label=activity.vertex).parcel.values)})
        
        if self.verbose:
            print("--- fNIRS spatial profiles built successfully ---")

    def build_eeg_source_spatial_profile(self):
        """ Build EEG-specific source spatial profiles from source configuration.

        The EEG source spatial profiles are built as dipoles only at the seed vertices
        defined in the source configuration object (self.source_cfg). Therefore, the resulting spatial profiles
        stored in self.s_eeg_spatial have dimensions (vertex, source), with non-zero values only at the seed vertices.
        These non-zero values are defined by the source spatial amplitude in the source configuration object. Dipole
        orientations are normal to cortex.
        """

        # Build spatial profiles
        print("\n--- Building EEG spatial profiles ---")

        if hasattr(self, 'source_cfg') is False:
            raise ValueError("Source configuration not initialized. Please run init_source_config() first.")

        # Initialize spatial profiles
        self.s_eeg_spatial = None

        # Compute some helper variables
        v_dipoles = self.dipole_locations.vertex.values.tolist()
        v_pos_dipoles = self.dipole_locations.pint.dequantify().values
        # Original vertex location
        v_pos_original = self.head_ijk.brain.vertices.pint.dequantify()

        for source in self.source_cfg:
            
            # validate spatial profile
            if source.spatial.vertices is None:
                raise ValueError(f"Source spatial vertices not set for source '{source.name}'. Please run build_shared_source_spatial_profile() first.")
            
            # Activity as a dipole orientation only at the seed vertex
            activity= np.zeros(self.dipole_locations.shape[0])

            source_vertices = source.spatial.vertices

            v_ndx_list = []
            for v in source_vertices:
                if v in v_dipoles:
                    v_ndx = v_dipoles.index(v)
                else:
                    # Find the closest vertex in the dipole locations (may not be the exact same because
                    # some vertices may have been removed when building the leadfield)
                    v_ndx = np.argmin(np.linalg.norm(v_pos_dipoles - v_pos_original.sel(label=v).values, axis=1))

                v_ndx_list.append(v_ndx)

            # Set activity magnitude at the selected dipole vertices as relative amplitude divided by number of vertices
            # (In this way the total amplitude (sum over vertices) is preserved regardless of number of vertices)
            activity[v_ndx_list] = 1.0 * source.spatial.amp / len(v_ndx_list)

            activity = xr.DataArray(activity, 
                                    dims=['vertex'], 
                                    coords={'vertex': self.dipole_locations.vertex.values})
            
            # Add source dimension
            activity = activity.expand_dims('source', 1).assign_coords({'source': [source.name]})

            if self.s_eeg_spatial is None:
                self.s_eeg_spatial = activity
            else:
                self.s_eeg_spatial = xr.concat([self.s_eeg_spatial, activity], dim='source')
    
            # Add parcel coordinates
            if 'parcel' in self.head_ijk.brain.vertices.coords:
                self.s_eeg_spatial = self.s_eeg_spatial.assign_coords({'parcel': ('vertex', self.head_ijk.brain.vertices.sel(label=activity.vertex).parcel.values)})

            if self.verbose:
                print(f"EEG spatial profile built for source '{source.name}', with amplitude {source.spatial.amp}.")
        
        if self.verbose:
            print("--- EEG spatial profiles built successfully ---")

    def build_fnirs_mixing_matrix(self):
        """ Build fNIRS mixing matrix from spatial profiles, sensitivity matrix, and extinction coefficients."""

        # Build fNIRS mixing matrix
        if self.verbose:
            print("\n--- Building fNIRS mixing matrix ---")
        
        # Select only brain vertices
        sensitivity_brain = self.sensitivity.sel(vertex=self.sensitivity.is_brain)
        # Read extinction coefficients
        Ext_coeff = cedalion.nirs.get_extinction_coefficients("prahl", self.fnirs_bg_rec.wavelengths)
        # Build mixing matrix from sensitivity, spatial profile, and extinction coefficients
        A = sensitivity_brain @ self.s_fnirs_spatial * Ext_coeff
        # Add units
        A = A.pint.to('1/micromolar')
        # # Normalize so each source has max amplitude equal to its spatial amp  TODO: Remove this, explanation give in my notes
        # for source in self.source_cfg:
        #     src_amp = source.spatial.amp
        #     A_source = A.sel(source=source.name) 
        #     A.loc[dict(source=source.name)] = A_source * src_amp / A_source.max() * A.pint.units
        self.A_fnirs = A

        if self.verbose:
            print("--- fNIRS mixing matrix built successfully ---")

    def build_eeg_mixing_matrix(self):
        """ Build EEG mixing matrix from spatial profiles and leadfield."""

        # Build EEG mixing matrix
        if self.verbose:
            print("\n--- Building EEG mixing matrix ---")
        
        # Build mixing matrix from leadfield and spatial profile by contracting over vertex and orientation
        A = self.leadfield @ self.s_eeg_spatial

        # Normalize so each source has max amplitude equal to its spatial amp
        for source in self.source_cfg:
            src_amp = source.spatial.amp
            A_source = A.sel(source=source.name) 
            A.loc[dict(source=source.name)] = A_source * src_amp / abs(A_source).max() * A.pint.units

        self.A_eeg = A

        if self.verbose:
            print("--- EEG mixing matrix built successfully ---")

    def build_shared_stimuli_marker(self,
                             T_stim: 'float' = 10.0 * units.seconds,
                             T_rest_min: 'float' = 8.0 * units.seconds,
                             T_rest_max: 'float' = 12.0 * units.seconds,
                             min_stim_amp: 'float' = 0.5,
                             max_stim_amp: 'float' = 1.0,
                             order: 'str' = 'random',
                             response_t: float = 0.0):
        """ Build stimulus markers for fNIRS and EEG data.
        
        It builds the stimulus markers, defining the schedule of the simulated experiment. 
        This is the way in which we incorporate temporal (When) co-modulation between modalities.
        It uses Cedalion's synthetic_hrf.build_stim_df() function to build a DataFrame
        with stimulus markers, based on the provided stimulus duration, rest interval,
        stimulus amplitude range, and trial source mapping. 
        The simulated markers span the time interval defined by T_rest_start and T_rest_end.
        The resulting stimulus markers are stored in self.stim. 
        Reaction time is simulated by adding a random response time to the stimulus onsets and durations,
        resulting in self.stim_with_jitter. The latter is used to set the schedule in the source configuration object.
        The response times are uniformly sampled between 0 and response_t seconds.
        """

        if self.verbose:
            print("\n--- Building stimulus markers ---")

        if hasattr(self, 'source_cfg') is False:
            raise ValueError("Source configuration not initialized. Please run init_source_config() first.")
        if hasattr(self, 'trial_source_map') is False:
            raise ValueError("Trial source map not initialized. Please run init_source_config() first.")

        # Check values have units
        if not isinstance(T_stim, units.Quantity):
            print("Warning, T_stim has no units, assuming seconds")
            T_stim = T_stim * units.seconds
        if not isinstance(T_rest_min, units.Quantity):
            print("Warning, T_rest_min has no units, assuming seconds")
            T_rest_min = T_rest_min * units.seconds
        if not isinstance(T_rest_max, units.Quantity):
            print("Warning, T_rest_max has no units, assuming seconds")
            T_rest_max = T_rest_max * units.seconds
        if not isinstance(response_t, (float, int)) or response_t < 0:
            raise ValueError("response_t must be a non-negative float representing seconds.")
        if response_t > T_stim.magnitude:
            raise ValueError("response_t cannot be larger than T_stim.")

        
        self.T_stim = T_stim
        self.T_rest_min = T_rest_min
        self.T_rest_max = T_rest_max
        self.stim_amp_min = min_stim_amp
        self.stim_amp_max = max_stim_amp

        # Build stimulus markers using Cedalion synthetic_hrf functionality
        self.stim = synthetic_hrf.build_stim_df(
            max_time= self.T_rest_end - self.T_rest_start,
            trial_types=list(self.trial_source_map.keys()),
            min_interval=self.T_rest_min,
            max_interval=self.T_rest_max,
            min_stim_dur = self.T_stim,
            max_stim_dur = self.T_stim,
            min_stim_value = self.stim_amp_min,
            max_stim_value =self.stim_amp_max,
            order=order,
        )

        # Shift onsets by T_rest_start
        self.stim['onset'] = self.stim['onset'] + self.T_rest_start.magnitude

        # Simulate reaction times by adding jitter to onsets and durations
        onsets = [o + float(np.random.rand(1)*response_t) for o in self.stim['onset'].values]
        durations = [d + float(np.random.rand(1)*response_t) for d in self.stim['duration'].values]
        stim_with_jitter = self.stim.copy()
        stim_with_jitter['onset'] = onsets
        stim_with_jitter['duration'] = durations
        self.stim_with_jitter = stim_with_jitter

        # Add schedulle information in SourcesConfig object

        if self.verbose:
            print("Setting stimulus schedule in source configuration...")
        for source in self.source_cfg:

            # Restrict stim dataframe to trials mapped to this source (handle multiple trial types per source)
            trials_source = [k for k, v in self.trial_source_map.items() if source.name in v]
            stim_source = self.stim_with_jitter.loc[self.stim_with_jitter.trial_type.isin(trials_source)]

            source.set_schedule(onsets=stim_source['onset'].values,
                                durations=stim_source['duration'].values,
                                values=stim_source['value'].values)
            
            if self.verbose:
                print(f"Source '{source.name}' schedule set with {len(stim_source)} trials of types {trials_source}.")
            
        if self.verbose:
            print(f"--- Stimulus markers built successfully, total trials = {len(self.stim)} ---")

    def build_fnirs_source_temporal_profile(self):
        """ Build fNIRS temporal profiles from stimulus markers and canonical HRF.
        
        The fNIRS temporal profiles are built using Cedalion's synthetic_hrf module,
        which creates HRF-based temporal profiles for each source in the source configuration
        based on the stimulus markers defined in self.stim. The resulting temporal profiles
        are stored in self.s_fnirs_temporal with dimensions (time, chromo, source). When multiple
        trial types are mapped to the same source, their each of their HRF responses are scaled
        accordingly with the stimulus values and later summed to create a single temporal profile per source.
        The HbO and HbR scales for each source are infered from the source configuration object,
        and they scale each source temporal profile equally for all trial types mapped to that source.
        The HRF model can be selected (currently only 'Gamma' is supported). 
        """

        if self.verbose:
            print("\n--- Building fNIRS temporal profiles ---")

        if hasattr(self, 'source_cfg') is False:
            raise ValueError("Source configuration not initialized. Please run init_source_config() first.")
        
        # Build base time series for the HRF with shape (channel, chromo, time)
        ts_hrf = self.fnirs_bg_rec['od'] * 0
        ts_hrf = ts_hrf.rename({'wavelength': 'chromo'}).assign_coords({'chromo': ['HbO', 'HbR']})

        self.s_fnirs_temporal = None

        for source in self.source_cfg:

            if source.temporal_fnirs is None:
                raise ValueError(f"Source fNIRS temporal profile not set for source '{source.name}'. Set it via self.source_cfg.set_temporal_fnirs().")

            src_name = source.name
            model = source.temporal_fnirs.model
            model_params = source.temporal_fnirs.model_params
            available_basis = [name for name in dir(glm.basis_functions) 
                           if isinstance(getattr(glm.basis_functions, name), abc.ABCMeta)]
            if model not in available_basis:
                raise ValueError("model must be one of the available basis functions in glm.basis_functions: "
                                    f"{available_basis}, got '{model}' instead.")
            # Build design matrix with HRF basis functions
            basis_fct = getattr(glm.basis_functions, model)(**model_params)  # Instantiate basis function class
            
            # Add shared amp information to source temporal profile
            if source.temporal_fnirs.amp in [None, 1.0]:
                if self.verbose:
                    print(f"Setting shared temporal amp for source '{src_name}' to {self.shared_temporal_amps[src_name]}...")
                source.temporal_fnirs.amp = self.shared_temporal_amps[src_name]
            else:
                raise ValueError(f"Found temporal amp already set for source '{src_name}', please remove it to use shared temporal amps." +
                                 " Shared amplitudes must be set via init_source_config().")

            # Define HbO and HbR scales from source configuration            
            amp = source.temporal_fnirs.amp
            ratio = source.temporal_fnirs.hbr_hbo_ratio
            if amp <= 0 or amp > 1:
                raise ValueError(f"Invalid fNIRS amplitude {amp} for source '{source.name}', must be in (0, 1].")
            if ratio >= 0 or ratio < -1:
                raise ValueError(f"Invalid HbR/HbO ratio {ratio} for source '{source.name}', must be in [-1, 0).")
            
            HbO_s = amp
            HbR_s = HbO_s * ratio
            
            # Restrict stim dataframe to trials mapped to this source (handle multiple trial types per source)
            trials_source = [k for k, v in self.trial_source_map.items() if source.name in v]
            stim_source = self.stim_with_jitter.loc[self.stim_with_jitter.trial_type.isin(trials_source)]

            dms = glm.design_matrix.hrf_regressors(ts_hrf, stim_source, basis_fct)
            hrf_regs = dms.common
            if model == 'GammaDeriv':
                # Keep only derivative HRF
                hrf_regs = hrf_regs.sel(regressor=[r for r in hrf_regs.regressor.values if 'gamma_deriv' in r])
            elif model in ['GaussianKernels', 'GaussianKernelsWithTails']:
                # Sum all Gaussian kernels into single regressor using random weights
                hrf_regs_avg = None
                for t in trials_source:
                    reg_names = [r for r in hrf_regs.regressor.values if t in r]
                    weights =  np.random.rand(len(reg_names))
                    weights /= weights.sum()
                    # Average weighted HRF regressor
                    hrf_regs_t = (hrf_regs.sel(regressor=reg_names) * weights[np.newaxis, :, np.newaxis]).sum(dim='regressor')
                    # Make it into a single HRF dimension
                    hrf_regs_t = hrf_regs_t.expand_dims('regressor').assign_coords(regressor=[f'HRF {t}'])
                    hrf_regs_t = hrf_regs_t.transpose('time', 'regressor', 'chromo')
                    if hrf_regs_avg is None:
                        hrf_regs_avg = hrf_regs_t
                    else:
                        hrf_regs_avg = xr.concat([hrf_regs_avg, hrf_regs_t], dim='regressor')

                hrf_regs = hrf_regs_avg  # Replace with averaged HRF regressors

            hrf_regs = hrf_regs / hrf_regs.max()  # Normalize to max of 1
            
            # Get max values per trial type and scale accordingly
            values_max = [stim_source[stim_source.trial_type == t.split(' ')[1]].value.max() for t in hrf_regs.regressor.values]
            print(values_max)
            hrf_regs = hrf_regs * xr.DataArray(values_max,
                                        dims=['regressor'], 
                                        coords={'regressor': hrf_regs.regressor.values})
            # Collapse to single regressor per source
            hrf_regs = hrf_regs.sum(dim='regressor')
            # Add source dimension
            hrf_regs = hrf_regs.expand_dims('source', 1).assign_coords(source=[source.name])
            # Scale HRF regressors for HbO and HbR
            hrf_regs = hrf_regs * xr.DataArray([HbO_s, HbR_s], 
                                        dims=['chromo'], 
                                        coords={'chromo': ['HbO', 'HbR']})

            # Add units (assumed micromolar)
            hrf_regs = hrf_regs * units.micromolar

            if self.s_fnirs_temporal is None:
                self.s_fnirs_temporal = hrf_regs
            else:
                self.s_fnirs_temporal = xr.concat([self.s_fnirs_temporal, hrf_regs], dim='source')

            if self.verbose:
                print(f"fNIRS temporal profile built for source '{source.name}', HbO scale: {HbO_s}, HbR scale: {HbR_s}.")

        if self.verbose:
            print("--- fNIRS temporal profiles built successfully ---")

    def build_eeg_source_temporal_profile(self):
        """ Build EEG temporal profiles from stimulus markers and ERD model.
        
        The EEG temporal profiles are built using an ERD model, where oscillatory activity is simulated
        in a specified frequency band, modulated by event-related desynchronization/synchronization (ERD/ERS) 
        envelopes based on the stimulus markers. The ERD drop (and ERS rebound) factors are calculated as
        the product of the stimulus values (trial-by-trial variability) and the source temporal amplitude 
        defined in the source configuration object. The latter is a shared amplitude across all trials
        and between modalities. To avoid full suppression of oscillatory activity during ERD, an overall scale
        of 0.9 is applied to the combined ERD drop factor. The resulting temporal profiles are stored in 
        self.s_eeg_temporal with dimensions (time, source).

        """

        if self.verbose:
            print("\n--- Building EEG temporal profiles ---")

        if hasattr(self, 'source_cfg') is False:
            raise ValueError("Source configuration not initialized. Please run init_source_config() first.")
        
        # --- Helper functions ---
        def _oscillatory_eeg_source_sum_sinusoids(t, freqs):
            """ Simulate a random oscillatory source in a frequency band by summing many sinusoids.
            
            The resulting PSD is relatively flat within the band, producing an unrealistic squared-shape PSD.
            Therefore, this function is not used, and kept only for reference.
            """

            fmin, fmax = freqs
            N_osc = int(t.max()*(fmax - fmin)*1.1)  # Number of oscillators to sum (guarrantees dense coverage of band)
            freqs  = np.linspace(fmin, fmax, N_osc)
            phases = np.random.uniform(0, 2*np.pi, size=N_osc)
            x = sum(np.sin(2*np.pi*fi*t + ph) for fi, ph in zip(freqs, phases))
            x /= np.abs(x).max()  # Normalize to 1

            return x
        
        def _oscillatory_eeg_source(t, freqs):
            """
            Simulate a random oscillatory source in a frequency band, with a peaky PSD.

            This version uses an AR(2) damped oscillator with a center frequency
            drawn uniformly from the band [fmin, fmax], instead of summing many
            sinusoids (which yields a squarish PSD).

            Parameters
            ----------
            t : array, shape (n_times,)
                Time vector in seconds (assumed regularly sampled).
            freqs : tuple (fmin, fmax)
                Frequency band of interest (Hz).

            Returns
            -------
            x : array, shape (n_times,)
                Oscillatory source, normalized so that max |x| = 1.
            """
            fmin, fmax = freqs
            t = np.asarray(t)
            n_samples = t.size

            # Infer sampling frequency from time vector
            dt = np.mean(np.diff(t))
            sfreq = 1.0 / dt

            # Choose a random center frequency within the band
            # f0 = np.random.uniform(fmin, fmax)
            # Use center frequency
            f0 = (fmin + fmax) / 2.0  

            # AR(2) parameters: r controls peak width, sigma the broadband noise level
            r = 0.99   # closer to 1.0 -> narrower & taller peak
            sigma = .1

            omega = 2 * np.pi * f0 / sfreq
            a1 = 2 * r * np.cos(omega)
            a2 = -r**2

            rng = np.random.default_rng()
            eps = rng.normal(0.0, sigma, size=n_samples)

            x = np.zeros(n_samples)
            # Initialize with noise
            x[0] = eps[0]
            x[1] = eps[1]
            for n in range(2, n_samples):
                x[n] = a1 * x[n - 1] + a2 * x[n - 2] + eps[n]

            # Normalize to max amplitude 1
            max_abs = np.max(np.abs(x))
            if max_abs > 0:
                x = x / max_abs

            return x

        def _evelope_eeg_source(t, 
                                onset, 
                                erd_dur, 
                                erd_drop, 
                                ers_gain, 
                                ers_dur):
            
            """ Simulate an event-related desynchronization/synchronization (ERD/ERS) envelope.    
            - ERD: reduce amplitude during [onset, onset+erd_dur] by 1 - erd_drop, smooth edges
            - ERS: short rebound after ERD by factor (1 + ers_gain), Hann-shaped
            """

            def smooth_step(t, a, b):
                """ Smooth step helper (0->1) over [a,b] using a half-cosine"""
                y = np.zeros_like(t, float)
                m = (t >= a) & (t <= b)
                if m.any():
                    tau = (t[m] - a) / (b - a)
                    y[m] = 0.5 - 0.5 * np.cos(np.pi * tau)
                y[t > b] = 1.0
                return y

            # ERD: fade down then up (two smooth steps)
            T_smooth = .3  # Time of smoothing edge decays in seconds
            fade_in  = smooth_step(t, onset, onset + T_smooth)          # 0 -> 1
            fade_out = 1.0 - smooth_step(t, onset + erd_dur - T_smooth, onset + erd_dur)  # 1 -> 0
            plateau  = 1.0 - (fade_in * fade_out)  # ~0 in the middle, 1 at edges
            env = erd_drop * plateau + (1 - erd_drop)  # scale by erd_drop

            # ERS: short post-ERD rebound, Hann-shaped bump
            ers_start = onset + erd_dur
            ers_end   = ers_start + ers_dur
            m = (t >= ers_start) & (t <= ers_end)
            if m.any():
                hann = np.hanning(m.sum())
                bump = np.zeros_like(t)
                bump[m] = hann
                env *= 1.0 + ers_gain * (bump / bump.max())

            return env

        # Build base time series
        t = self.eeg_bg_raw.times

        self.s_eeg_temporal = None

        for source in self.source_cfg:

            if source.temporal_eeg is None:
                raise ValueError(f"Source EEG temporal profile not set for source '{source.name}'. Set it via self.source_cfg.set_temporal_eeg().")
            
            src_name = source.name
            model = source.temporal_eeg.model
            freqs = source.temporal_eeg.model_params['freq_band']
            ers_scale = source.temporal_eeg.model_params['ers_rebound_gain']

            if model.lower() != 'erd':
                raise ValueError(f"Unknown source model '{model}' for source '{source.name}'. Currently, only 'ERD' is supported.")

            # Add shared amp information to source temporal profile
            if source.temporal_eeg.amp in [None, 1.0]:
                if self.verbose:
                    print(f"Setting shared temporal amp for source '{src_name}' to {self.shared_temporal_amps[src_name]}...")
                source.temporal_eeg.amp = self.shared_temporal_amps[src_name]
            else:
                raise ValueError(f"Found temporal amp already set for source '{src_name}', please remove it to use shared temporal amps." +
                                 " Shared amplitudes must be set via init_source_config().")

            # Simulate oscillatory component in the specified frequency band for the entire duration
            s_osc = _oscillatory_eeg_source(t, freqs=freqs)

            # Simulate ERD-like envelope modulations based on the stimulus schedule
            s_env = np.ones_like(t)
            onsets = source.schedule.onsets
            durations = source.schedule.durations
            # Use schedule values for trial-specific amplitudes
            amps = source.schedule.values.copy()
            amps *= source.temporal_eeg.amp  # Scale by subject-specific temporal amp (shared across trials and between modalities)
            amps *= 0.9  # Overall scale to avoid full suppression 
            for o, d, a in zip(onsets, durations, amps):
                s_trial_env = _evelope_eeg_source(t, 
                                    onset=o, 
                                    erd_dur=d, 
                                    erd_drop=a, 
                                    ers_gain=a * ers_scale,
                                    ers_dur=.5)  # .5 second duration for ERS rebound

                s_env *= s_trial_env

            s_tmp = s_osc * s_env

            s_tmp = xr.DataArray(s_tmp.reshape(-1, 1),
                                dims=['time', 'source'],
                                coords={'time': t,
                                        'source': [source.name]})
            
            # Add units (assumed microvolts)
            s_tmp = s_tmp * units.nA * units.m
            
            if self.s_eeg_temporal is None:
                self.s_eeg_temporal = s_tmp
            else:
                self.s_eeg_temporal = xr.concat([self.s_eeg_temporal, s_tmp], dim='source')

            if self.verbose:
                print(f"EEG temporal profile built for source '{source.name}', frequency band: {freqs} Hz.")
        
        if self.verbose:
            print("--- EEG temporal profiles built successfully ---")
                                          
    def apply_fnirs_forward_model(self):    
        """Apply fNIRS forward model to get simulated fNIRS data.
        """

        if self.verbose:
            print("\n--- Applying fNIRS forward model ---")

        def _stack_source_dimension(ts):
            """ Stack 'source' and 'chromo' dimensions into a single 'source' dimension."""

            # Make sure dimensions are correct
            if 'source' not in ts.dims or 'chromo' not in ts.dims:
                raise ValueError("Input xarray must have 'source' and 'chromo' dimensions.")

            ts = ts.stack(source_new=('source', 'chromo'))
            ts = ts.assign_coords({'source_new': [f'{s}_{c}' for s, c in zip(ts.source.values, ts.chromo.values)]})
            ts = ts.rename({'source_new': 'source'})

            return ts

        # Combine spatial and temporal profiles per source to get voxel-wise source activity.
        if self.verbose:
            print("Combining fNIRS spatial and temporal profiles to get voxel-wise source activity...")
        s_fnirs_voxel = xr.concat([self.s_fnirs_temporal.sel(source=s.name) * self.s_fnirs_spatial.sel(source=s.name) for s in self.source_cfg], dim='source')
        self.s_fnirs_voxel = s_fnirs_voxel
        self.fnirs_sim_voxel = s_fnirs_voxel.sum(dim='source')

        # Stack source dimension in A and s to apply forward model for each source separately, then sum
        A_fnirs_stacked = _stack_source_dimension(self.A_fnirs)
        s_fnirs_temporal_stacked = _stack_source_dimension(self.s_fnirs_temporal)
        source_spatial_profile_stacked = s_fnirs_temporal_stacked.source.values
        self.fnirs_sim_od_s = xr.concat([A_fnirs_stacked.sel(source=s) * s_fnirs_temporal_stacked.sel(source=s) for s in source_spatial_profile_stacked], dim='source')
        self.fnirs_sim_od = self.fnirs_sim_od_s.sum(dim='source')

        if self.verbose:
            print("--- fNIRS forward model applied successfully ---")

    def apply_eeg_forward_model(self):    
        """
        Apply the EEG forward model to simulate EEG data based on source configurations and background EEG data.
        This method performs the following steps:
        1. Identifies unique frequency bands across sources and reduces them to non-overlapping sets.
        2. Bandpass filters the background EEG data into these non-overlapping frequency bands.
        3. Normalizes the power of the simulated EEG data for each source to match the background power in the corresponding frequency band.
        4. Constructs the simulated EEG data by forward modeling each source separately and summing the contributions.
        The resulting simulated EEG data is stored in the `self.eeg_sim_raw` attribute as an MNE Raw object.
        Attributes:
            self.eeg_sim_s (xarray.DataArray): Simulated EEG data for each source.
            self.eeg_sim (xarray.DataArray): Summed simulated EEG data across all sources.
            self.eeg_sim_raw (mne.io.Raw): Simulated EEG data as an MNE Raw object.
            self.x_bg_dif (numpy.ndarray): Accumulated out-of-band background EEG data.
            self.x_bg_bands (numpy.ndarray): Accumulated in-band background EEG data.
        Notes:
            - The simulated EEG data is normalized to ensure that the power in each frequency band matches the background power,
              scaled by the source's temporal amplitude.
            - The method assumes that the background EEG data (`self.eeg_bg_raw`) is already loaded and available.
            - Note that the variables self.x_bg_dif and self.x_bg_bands are used later in add_background_eeg, but only
                if combine_method='band_supression', otherwise they are not used at all! For this reason, they may be 
                deprecated in future versions.

        Raises:
            ValueError: If the frequency bands of the sources cannot be matched to the combined frequency bands.
        Parameters:
            None
        Returns:
            None
        """
        

        if self.verbose:
            print("\n--- Applying EEG forward model ---")
        
        # Identify unique frequency bands across sources
        freqs = set([s.temporal_eeg.model_params['freq_band'] for s in self.source_cfg])

        # Reduce frequency bands to non-overlapping sets
        freqs_combined = []
        for f in freqs:
            for f2 in freqs:
                # Check for overlap
                if not (f[1] < f2[0] or f2[1] < f[0]):
                    # Merge
                    f = (min(f[0], f2[0]), max(f[1], f2[1]))
            if f in freqs_combined:
                continue
            freqs_combined.append(f)

        # Bandpass filter background data to each non-overlapping band
        x_bg = self.eeg_bg_raw.copy().pick(picks='eeg')
        x_bg_dif = x_bg._data # To accumulate out-of-band background
        x_bg_bands = np.zeros_like(x_bg._data)  # To accumulate in-band background
        x_bg_bands_dict = {}
        norm_factors = {}
        for fmin, fmax in freqs_combined:
            x_bg_band = x_bg.copy()
            x_bg_band.filter(
                l_freq=fmin, 
                h_freq=fmax, 
                method='fir', phase='zero'
                )
            # Update dict
            x_bg_bands_dict[f'{fmin}-{fmax}'] = x_bg_band
            # Subtract from out-of-band background
            x_bg_dif = x_bg_dif - x_bg_band._data
            # Accumulate in-band background
            x_bg_bands = x_bg_bands + x_bg_band._data

            # Calculate normalization factor for this band
            norm_factor = 0
            for source in self.source_cfg:
                fmin_s, fmax_s = source.temporal_eeg.model_params['freq_band']
                if fmin <= fmin_s and fmax >= fmax_s:
                    amp_band = source.temporal_eeg.amp
                    norm_factor += amp_band**2

            norm_factor = np.sqrt(norm_factor)
            norm_factors[f'{fmin}-{fmax}'] = norm_factor
        
        # To restrict data to rest period only
        t0, t1 = self.T_rest_start.magnitude, self.T_rest_end.magnitude
        i0, i1 = x_bg.time_as_index([t0, t1])

        # Build mixture of sources by forward modeling each source separately and summing.
        # We use source-specific scalings to guarantee that each source's power in its frequency band
        # matches the background power in that band, scaled by the source temporal amplitude.
        eeg_sim_s = None
        for source in self.source_cfg:

            # Forward model for individual source
            x_sim = self.A_eeg.sel(source=source.name) * self.s_eeg_temporal.sel(source=source.name)

            # Read source frequency band
            fmin, fmax = source.temporal_eeg.model_params['freq_band']
            # Get corresponding combined band
            fmin, fmax = [f for f in freqs_combined if (f[0] <= fmin and fmax <= f[1]) ][0]
            # Get corresponding background data
            x_bg_band = x_bg_bands_dict[f'{fmin}-{fmax}']
            
            # Select channel with highest activity
            A_eeg_source = self.A_eeg.sel(source=source.name)
            max_ch = A_eeg_source.channel[abs(A_eeg_source).argmax()].values

            # Restrict data to rest period only for noise estimation
            # x_bg_band = x_bg_band.get_data(picks=max_ch)[0, slice(i0, i1)]
            x_bg_band = x_bg_band._data[:, slice(i0, i1)]  # all channels
            x_sim_rest = x_sim.sel(channel=max_ch, time=slice(t0, t1))
            
            # De-mean on rest window
            x_bg_band -= x_bg_band.mean(axis=1, keepdims=True)
            x_sim_rest -= x_sim_rest.mean(dim='time')

            # Power (RMS^2), restricted to rest period
            P_bg = (x_bg_band**2).mean()
            P_source = (x_sim_rest**2).mean()
            scale = np.sqrt(P_bg / P_source)

            # Scale to match background power in band times relative amplitude
            x_sim = x_sim * scale * source.temporal_eeg.amp
            # Normalize so total power matches background power in band, accounting for other sources in same band
            x_sim = x_sim / norm_factors[f'{fmin}-{fmax}']
            # Since background was in V, after scalin x_sim is also in V
            x_sim.data = x_sim.data.magnitude * pint.Unit('V')
            
            if eeg_sim_s is None:
                eeg_sim_s = x_sim.expand_dims('source')
            else:
                eeg_sim_s = xr.concat([eeg_sim_s, x_sim], dim='source')
        
        self.eeg_sim_s = eeg_sim_s
        self.eeg_sim = self.eeg_sim_s.sum(dim='source')

        # Create MNE Raw object (already in Volts as expected by MNE)
        data = self.eeg_sim.transpose("channel", "time")
        data = data.pint.dequantify().values

        # Add info, montage and annotations
        bg_raw = self.eeg_bg_raw.copy().pick(self.eeg_sim.channel.values)
        info = bg_raw.info
        montage = bg_raw.get_montage()
        annotations = mne.Annotations(onset=self.stim.onset.values,
                                    duration=self.stim.duration.values,
                                    description=self.stim.trial_type.values)

        sim_raw = mne.io.RawArray(data, info, verbose=False)
        sim_raw.set_montage(montage, verbose=False)
        sim_raw.set_annotations(annotations)

        self.eeg_sim_raw = sim_raw
        self.x_bg_dif = x_bg_dif
        self.x_bg_bands = x_bg_bands

        if self.verbose:
            print("--- EEG forward model applied successfully ---")


    def add_background_fnirs(self, snr_db=1):
        """Add background fNIRS data to simulated fNIRS data at specified SNR (in dB).

        SNR is defined in the rest period only, as:
            signal_power / noise_power

        computed on the channel and wavelength with maximum synthetic activity,
        then converted from dB to linear internally.

        Parameters
        ----------
        snr_db : float
            Target SNR in dB for the rest period on the channel & wavelength
            with maximum synthetic activity.
        """

        if self.verbose:
            print("\n--- Adding background fNIRS data ---")

        synth = self.fnirs_sim_od.copy()          # (n_ch, 2, n_t)
        bg    = self.fnirs_bg_rec['od'].copy()    # (n_ch, 2, n_t)

        # Restrict to rest period for power estimation
        tsel = slice(self.T_rest_start.magnitude, self.T_rest_end.magnitude)
        synth_sel = synth.sel(time=tsel)
        bg_sel    = bg.sel(time=tsel)

        # Demean along time
        synth_sel = synth_sel - synth_sel.mean(dim='time')
        bg_sel    = bg_sel - bg_sel.mean(dim='time')

        # Channel/wavelength with max synthetic activity
        ch_wl_max = synth_sel.where(synth_sel == synth_sel.max(), drop=True)
        ch_max = ch_wl_max['channel'].values[0]
        wl_max = ch_wl_max['wavelength'].values[0]

        synth_max = synth_sel.sel(channel=ch_max, wavelength=wl_max)
        bg_max    = bg_sel.sel(channel=ch_max, wavelength=wl_max)

        # Powers (RMS^2)
        P_sig = (synth_max ** 2).mean()
        P_noise_orig = (bg_max ** 2).mean()

        # Equalize signal & noise power on that channel (baseline SNR = 1)
        scale = np.sqrt(P_noise_orig / P_sig).pint.dequantify().values

        # Convert SNR from dB to linear
        snr_lin = 10.0 ** (snr_db / 10.0)

        # Linear SNR -> alpha (fraction of power allocated to 'signal')
        # SNR = alpha / (1 - alpha)  ->  alpha = SNR / (1 + SNR)
        alpha = snr_lin / (1.0 + snr_lin)

        # Normalized mixture: keeps power ≈ constant, sets SNR ≈ snr_lin
        fnirs_od = np.sqrt(1.0 - alpha) * bg + np.sqrt(alpha) * synth * scale

        # Store for later inspection
        self.snr_fnirs_db = snr_db
        self.snr_fnirs_lin = snr_lin
        self.alpha_fnirs = alpha
        self.fnirs_od = fnirs_od

        if self.verbose:
            print("--- Background fNIRS data added successfully ---")

    def add_background_eeg(self, snr_db=0.0, combine_method='overall_supression'):
        """
        Add background EEG to simulated EEG at a specified Signal-to-Noise Ratio (SNR) in decibels (dB), 
        using one of the available combination methods.
        This method combines synthetic EEG data with background EEG data to achieve a target SNR. 
        The SNR is calculated based on the power ratio of the synthetic signal to the background noise 
        in the specified synthetic band during the "rest" window. The combination method determines 
        how the synthetic and background EEG data are mixed.
        Parameters:
        -----------
        snr_db : float, optional
            Target Signal-to-Noise Ratio (SNR) in decibels (dB). Default is 0.0.
        combine_method : str, optional
            Method for combining synthetic and background EEG data. Must be one of:
            - 'band_supression': Suppresses background EEG in the source bands, then adds synthetic 
              EEG and out-of-band background EEG.
            - 'overall_supression': Suppresses background EEG overall, then adds synthetic EEG. 
              Does not use band-specific suppression.
            - 'simple_scaling': Scales the synthetic EEG data without suppressing the background EEG.
        Attributes Updated:
        -------------------
        snr_eeg_db : float
            The target SNR in decibels (dB).
        snr_eeg_lin : float
            The target SNR in linear scale.
        alpha_eeg : float
            Scaling factor derived from the SNR for power ratio scaling.
        eeg : mne.io.Raw
            The combined EEG data after applying the specified combination method.
        Raises:
        -------
        ValueError
            If an unknown `combine_method` is provided.
        Notes:
        ------
        - The SNR is computed on the "rest" window using band-passed signals.
        - The method identifies the channel with the maximum synthetic band power on the rest window 
          to calculate the SNR.
        - The combined EEG data retains the annotations of the synthetic EEG data.
        Example:
        --------
        >>> eeg_pipeline.add_background_eeg(snr_db=10.0, combine_method='band_supression')
        """
        

        if self.verbose:
            print("\n--- Adding background EEG data ---")

        # Convert SNR from dB to linear
        snr_lin = 10.0 ** (snr_db / 10.0)

        # Calculate alpha from snr (for power ratio scaling)    
        alpha = snr_lin / (1 + snr_lin)

        # Combined data has same structure as background raw
        x_combined = self.eeg_bg_raw.copy().pick(picks='eeg')
        # Define combined data in three possible ways:
        # Suppress background in source bands, then add synthetic and out-of-band background
        if combine_method == 'band_supression':
            x_combined._data = self.x_bg_dif + self.x_bg_bands * (1 - alpha) + self.eeg_sim_raw._data * alpha
        # Suppress background overall, then add synthetic (x_bg_bands and x_bg_dif not used in this case)
        elif combine_method == 'overall_supression':
            x_bg = self.eeg_bg_raw.copy().pick(picks='eeg')._data
            x_combined._data = np.sqrt(1 - alpha) * x_bg + np.sqrt(alpha) * self.eeg_sim_raw._data
            # Scale final data to match original power levels
            x_combined._data = x_combined._data * np.sqrt((x_bg**2).mean() / (x_combined._data**2).mean())
        elif combine_method == 'simple_scaling':
            # Simple scaling of synthetic data without background suppression
            x_combined._data = self.eeg_bg_raw.copy().pick(picks='eeg')._data + snr_lin * self.eeg_sim_raw._data
        else:
            raise ValueError(f"Unknown combine_method '{combine_method}', must be one of ['band_supression', 'overall_supression', 'simple_scaling'].")

        x_combined.set_annotations(self.eeg_sim_raw.annotations)

        # Store attributes
        self.snr_eeg_db = snr_db
        self.snr_eeg_lin = snr_lin
        self.alpha_eeg = alpha
        self.eeg = x_combined

        if self.verbose:
            print("--- Background EEG data added successfully ---")

    @classmethod
    def run_full_pipeline(cls,
                          config_file: str, 
                          snr: float = 10.0,
                          verbose: bool = True,
                          seed: float = 42,
                          overwrite: bool = False,
                          save: bool = False) -> 'SimulatefNIRSAndEEGData':
        
        """ Run full simulation pipeline from configuration file.

        This class method runs the complete simulation pipeline for fNIRS and EEG data
        based on the provided configuration file. It initializes the simulation class,
        builds source configurations, head models, spatial and temporal profiles,
        mixing matrices, stimulus markers, applies forward models, and adds background data.
        """

        config_file = Path(config_file).expanduser().resolve()
        with config_file.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Split configuration into sections
        cfg_paths = cfg.get('paths', {})
        cfg_bg = cfg.get('background', {})
        cfg_hm = cfg.get('head_montage', {})
        cfg_fm = cfg.get('forward_model', {})
        cfg_sources = cfg.get('source_config', {})
        cfg_markers = cfg.get('stimulus_markers', {})

        # --- INTIALIZE SIMULATION CLASS ---
        subjects_dir = Path(cfg_paths.get('subjects_dir', None))
        subject = Path(cfg_paths.get('subject', None))
        sub_dir = subjects_dir / subject

        sim = cls(subjects_dir=subjects_dir,
                  subject=subject,
                  verbose=verbose,)
        sim.seed = seed
        sim.overwrite = overwrite
        sim.save = save

        
        # --- BACKGROUND DATA ---
        fnirs_background_file = sub_dir / cfg_bg.get('fnirs_background', None)
        eeg_background_file = sub_dir / cfg_bg.get('eeg_background', None)
        rest_window = cfg_bg.get('rest_window', None)  # todo
        sim.load_background_data(fnirs_file=fnirs_background_file, 
                                eeg_file=eeg_background_file,
                                rest_window=rest_window)
        
        # --- HEADMODEL AND MONTAGE ---
        headmodel_name = cfg_hm.get('headmodel_name', None)
        segmentation_masks = cfg_hm.get('segmentation_masks', None)

        headmodel_dir = sub_dir / 'headmodel'
        brain_surface_file = headmodel_dir / cfg_hm.get('brain_surface', None)
        scalp_surface_file = headmodel_dir / cfg_hm.get('scalp_surface', None)
        landmarks_ras_file = headmodel_dir / cfg_hm.get('landmarks_ras', None)
        parcel_file = headmodel_dir / cfg_hm.get('parcels', None)
        montage_file = sub_dir / cfg_hm.get('montage_file', None)
        # Heamodel
        sim.build_headmodel(head_model_name=headmodel_name, 
                            segmentation_dir=headmodel_dir,
                            mask_files=segmentation_masks,
                            brain_surface_file=brain_surface_file,
                            scalp_surface_file=scalp_surface_file,
                            landmarks_ras_file=landmarks_ras_file,
                            brain_face_count=None,
                            scalp_face_count=None,
                            brain_seg_types=['gm', 'wm'],
                            scalp_seg_types=['scalp'],
                            smoothing=0.5,
                            fill_holes=True,
                            parcel_file=parcel_file)
        # Montage
        sim.build_montage(montage_file=montage_file,
                  eeg_montage_name='standard_1020')
        
        # --- BUILD FORWARD MODELS ---
        sensitivity_file = sub_dir / cfg_fm.get('sensitivity', None)
        eeg_forward_sol_file = sub_dir / cfg_fm.get('eeg_forward_sol', None)
        conductivity = cfg_fm.get('conductivity', None)

        # Sensitvity
        sim.build_sensitivity(sensitivity_fname=sensitivity_file)
        # Leadfield
        sim.build_leadfield(eeg_forward_sol_fname=eeg_forward_sol_file,
                    conductivity=conductivity)

        # --- SOURCE CONFIGURATION ---
        shared_spatial_amps = cfg_sources.get('spatial_amps', {})
        shared_temporal_amps = cfg_sources.get('temporal_amps', {})
        trial_types = cfg_sources.get('trial_types', {})
        sources = cfg_sources.get('sources', [])
        # Initialize source configuration from list of dicts
        sources_config = SourcesConfig.from_list(sources)
        source_names = [s.name for s in sources_config]

        # Initialize source configuration in sim class
        sim.init_source_config(names=source_names,
                       spatial_amps=shared_spatial_amps,
                       temporal_amps=shared_temporal_amps,
                       trial_types=trial_types)
        
        # Add spatial and temporal information to initialized sources
        sim.source_cfg = sources_config

        # --- SPATIAL PROFILE ---
        # Build shared source spatial profiles
        sim.build_shared_source_spatial_profile()
        # Build modality-specific spatial profiles
        sim.build_fnirs_source_spatial_profile()
        sim.build_eeg_source_spatial_profile()
        # Calculate mixing matrices
        sim.build_fnirs_mixing_matrix()
        sim.build_eeg_mixing_matrix()

        # --- STIMULUS MARKERS ---
        sim.build_shared_stimuli_marker(**cfg_markers)

        # --- TEMPORAL PROFILE ---
        sim.build_fnirs_source_temporal_profile()
        sim.build_eeg_source_temporal_profile()

        # --- APPLY FORWARD MODEL---
        sim.apply_fnirs_forward_model()
        sim.apply_eeg_forward_model()

        # --- ADD BACKGROUND DATA ---
        sim.add_background_fnirs(snr=snr)
        sim.add_background_eeg(snr=snr)


        return sim
        


    def plot_surfaces(self, 
                      plot_montage=False, 
                      plot_brain=False,
                      plot_scalp=False,
                      plot_parcels=False,
                      plot_bem=False,
                      plot_head_landmarks=False, 
                      show_labels=False, 
                      plot_activity=False, 
                      crs='ras'):
        """ Plot surfaces: brain, scalp, montage, parcels, head landmarks.
        """

        # Validate coordinate system
        if crs not in ['ras', 'ijk']:
            raise ValueError("crs must be either 'ras' or 'ijk'")

        # Check what is available
        has_montage = hasattr(self, 'montage_ras') and hasattr(self, 'montage_ijk')
        has_head = hasattr(self, 'head_ras') and hasattr(self, 'head_ijk')
        has_parcels = hasattr(self, 'parcel_file') and self.parcel_file is not None

        if has_montage:
                montage = self.montage_ras if crs == 'ras' else self.montage_ijk
        elif not has_montage and plot_montage:
            raise ValueError("Montage not found, cannot plot montage.")
        
        if has_head:
            head = self.head_ras if crs == 'ras' else self.head_ijk
        else:
            for surf in [plot_brain, plot_scalp, plot_head_landmarks]:
                if surf:
                    raise ValueError(f"Head model not found, cannot plot surface {surf}. Please run build_headmodel() first.")
                
        if has_parcels is False and plot_parcels:
            raise ValueError("Parcellation file not found in head model, cannot plot parcels.")

        # Initialize plotter
        plt_pv = pv.Plotter()

        # Montage
        if plot_montage:
            vbx.plot_labeled_points(plt_pv, montage, show_labels=show_labels)
        # Brain
        if plot_brain:
            vbx.plot_surface(plt_pv, head.brain, color="w")
        # Scalp
        if plot_scalp:
            vbx.plot_surface(plt_pv, head.scalp, opacity=.3)
        # Parcels
        if plot_parcels:
            plot_parcellated_surface(plt_pv, head.brain, parcel_file=self.parcel_file)
        # Head landmarks
        if plot_head_landmarks:
            vbx.plot_labeled_points(plt_pv, head.landmarks, show_labels=show_labels, color='orange')
        # Activity
        if plot_activity:
            print("Plotting source spatial activity not implemented yet.")
            # TODO: implement activity plotting
        # BEM surfaces
        if plot_bem:
            if hasattr(self, 'bem_trimesh') is False or self.bem_trimesh is None:
                raise ValueError("BEM surfaces not available. Please run build_leadfield() first with leadfield_fname=None to generate BEM surfaces.")
            for k, surf in self.bem_trimesh.items():
                vbx.plot_surface(plt_pv, surf, color="lightgray", opacity=.3)

        plt_pv.show()

        # Restore original stdout (report to Eike)
        sys.stdout = self._original_stdout

    def plot_sensitivity(self, low_th=-3, high_th=0, crs='ras'):
        """ Plot sensitivity matrix on brain surface."""


        # Validate coordinate system
        if crs not in ['ras', 'ijk']:
            raise ValueError("crs must be either 'ras' or 'ijk'")
        
        head = self.head_ras if crs == 'ras' else self.head_ijk
        montage = self.montage_ras if crs == 'ras' else self.montage_ijk

        plotter = cedalion.vis.anatomy.sensitivity_matrix.Main(
        sensitivity=self.sensitivity.pint.dequantify(),
        brain_surface=head.brain,
        head_surface=head.scalp,
        labeled_points=montage)

        plotter.plot(high_th=high_th, low_th=low_th)
        plotter.plt.show()

        # Restore original stdout (report to Eike)
        sys.stdout = self._original_stdout


    def plot_eeg_dipoles(self, plot_step=20):
        """ Plot EEG dipoles on brain surface."""

        # Downsample for visualization
        N_dipoles = len(self.dipole_locations)
        dipole_locations_ds = self.dipole_locations[:N_dipoles:plot_step].rename({'vertex': 'label'})
        dipole_orientations_ds = self.dipole_orientations[:N_dipoles:plot_step].rename({'vertex': 'label'})

        pc_dipoles = pv.PolyData(dipole_locations_ds.pint.dequantify().values)

        plotter = pv.Plotter()
        # vbx.plot_labeled_points(plotter, source_spatial_profile_ds, show_labels=False)
        plotter.add_mesh(pc_dipoles, point_size=8, render_points_as_spheres=True, color='g')
        vbx.plot_surface(plotter, self.head_ijk.brain, color="w", opacity=1)
        vbx.plot_vector_field(plotter, dipole_locations_ds, dipole_orientations_ds)
        plotter.show()

        # Restore original stdout (report to Eike)
        sys.stdout = self._original_stdout


    def plot_source_spatial_profile(self):
        """ Plot source locations on brain surface with colors graduated by amplitude."""

        # Convert brain surface to pyvista mesh
        b = cdc.VTKSurface.from_trimeshsurface(self.head_ras.brain)
        b = pv.wrap(b.mesh)

        # Assign colors to vertices belonging to each source
        v_colors = {}
        for s in self.source_cfg:
            src, vertices, amp = s.name, s.spatial.vertices,  s.spatial.amp
            for v in vertices:
                c = self.fmt[src]['fc']
                v_colors[v] = (c[0], c[1], c[2], amp)

        b['vertices'] = np.asarray([
            v_colors.get(v, (0.8, 0.8, 0.8, 1)) 
            for v in self.head_ras.brain.vertices.label.values
            ])

        plotter = pv.Plotter()

        vbx.plot_surface(
            plotter,
            self.head_ras.brain,
            color="w",
            opacity=1
        )
        plotter.add_mesh(
            b,
            scalars='vertices',
            rgb=True,
            smooth_shading=False
        )

        plotter.show()

    def plot_fnirs_spatial_activation(self, 
                                      plot_montage=False, 
                                      show_labels=False):
        """ Plot spatial activation on brain surface."""

        activity = self.s_fnirs_spatial.sum(dim='source')
        montage = self.montage_ras if plot_montage else None
        plot_activation(activity, 
                        self.head_ras, 
                        title=f"Source Locations: {[s.name for s in self.source_cfg]}", 
                        montage = montage, 
                        show_labels=show_labels)
        
    def plot_eeg_spatial_activation(self, 
                                    N_max=50,  # Max number of points to plot per source
                                    show_labels=False):
        
        plotter = pv.Plotter()
        for i, source in enumerate(self.source_cfg):

            s_name = source.name
            active_verts_mask = (self.s_eeg_spatial.sel(source=s_name) > 0).values
            locs = self.dipole_locations[active_verts_mask]
            normals = self.dipole_orientations[active_verts_mask]
            color = self.fmt[s_name]['fc']

            # Rename and add needed coords for plotting
            locs = locs.rename({'vertex': 'label'})
            if 'parcel' in locs.coords:
                locs = locs.drop_vars('parcel')
            locs = locs.assign_coords({'label': [s_name]*len(locs)})
            locs = locs.assign_coords({'type': ('label', [PointType.SOURCE]*len(locs))})
            normals = normals.rename({'vertex': 'label'})
            normals = normals.assign_coords({'label': [s_name]*len(normals)})
            normals = normals.assign_coords({'type': ('label', [PointType.SOURCE]*len(normals))})
            
            if len(locs) > 1:
                # Downsample if too many points
                if len(locs) > N_max:
                    step = len(locs) // N_max
                    locs = locs[::step]
                    normals = normals[::step]
                vbx.plot_labeled_points(plotter, locs[0:1], show_labels=show_labels, color=color)
                vbx.plot_labeled_points(plotter, locs[1:], show_labels=False, color=color)
            else:
                vbx.plot_labeled_points(plotter, locs, show_labels=show_labels, color=color)
            vbx.plot_vector_field(plotter, locs, normals)
        vbx.plot_surface(plotter, self.head_ijk.brain, color="w", opacity=1)
        plotter.show()

    def plot_fnirs_topomap(self, wl_ndx=1, figsize=(5, 5)):
        """ Plot scalp topographies of fNIRS activation for each source location"""

        if wl_ndx not in [0, 1]:
            raise ValueError("wl_ndx must be either 0 (lower wavelength) or 1 (higher wavelength)")
        
        od = self.fnirs_bg_rec['od']
        montage = self.montage_ras
        wl = self.fnirs_bg_rec.wavelengths[wl_ndx]
        A_wl = self.A_fnirs.sel(wavelength=wl)
        Amin = A_wl.pint.dequantify().min().values
        Amax = A_wl.pint.dequantify().max().values

        fig, ax = plt.subplots(self.Ns, 1, figsize=(figsize[0]*self.Ns, figsize[1]*self.Ns))
        ax = ax if self.Ns > 1 else [ax]

        # fig.set_size_inches(12, 3*self.Ns)
        for i, source in enumerate(self.source_cfg):

            A = A_wl.sel(source=source.name, chromo='HbO')
            title = f"{wl}nm, activation under {source.name}"
            cedalion.vis.anatomy.scalp_plot(
                od,
                montage,
                A,
                ax[i],
                cmap="YlOrRd",
                title=title,
                vmin=Amin,
                vmax=Amax,
                cb_label="max peak amplitude",
            )
        plt.show()

    def plot_eeg_topomap(self, relative_limits=True):

        # Plot a topomap (Info already has the electrode positions)
        fig, axs = plt.subplots(1, self.Ns, figsize=(4*self.Ns, 5))

        # Determine global vmin and vmax for color scale
        if relative_limits:
            vmin = self.A_eeg.min().pint.dequantify().values
            vmax = self.A_eeg.max().pint.dequantify().values

        for i, s in enumerate(self.source_cfg):

            ax = axs[i] if self.Ns > 1 else axs

            source_activity = self.A_eeg.sel(source=s.name).copy()
            source_units = source_activity.pint.units
            source_activity = source_activity.pint.dequantify().values
            info = self.eeg_bg_raw.copy().pick(self.A_eeg.channel.values.tolist()).info

            if not relative_limits:
                vmin = source_activity.min()
                vmax = source_activity.max()

            im, cn = mne.viz.plot_topomap(source_activity.data, 
                                        info, 
                                        axes=ax,
                                        cmap='RdBu_r', 
                                        show=False, 
                                        vlim=(vmin, vmax), 
                                        sensors=True,
                                        contours=6,
                                        names=info['ch_names'],) 
            ax.set_title(f"Source: {s.name}")

            cax = fig.colorbar(im)
            cax.set_label(f'Activity ({source_units})')

        # Adjust layout
        plt.tight_layout()
        plt.show()


# --- Helper functions ---


def _read_fnirs(snirf_file):
    """ Read raw fNIRS data from SNIRF file to be used as background data.
    """
    
    # Validate paths
    if not os.path.isfile(snirf_file):
        raise FileNotFoundError(f"fNIRS background data file {snirf_file} not found.")
    if not snirf_file.suffix == '.snirf':
        raise ValueError(f"fNIRS background data file {snirf_file} is not a SNIRF file.")

    # Read SNIRF file
    rec = cedalion.io.read_snirf(snirf_file)
    
    # Check only one snirf object
    if len(rec) > 1:
        raise ValueError("Multiple SNIRF objects found, only one is supported.")
    rec = rec[0]

    return rec

def _read_eeg(eeg_file):
    """ Read raw EEG data from .fif file to be used as background data.
    """

    # Validate paths
    eeg_file = Path(eeg_file).expanduser().resolve()
    if not eeg_file.exists() or not eeg_file.is_file():
        raise FileNotFoundError(f"EEG background data file not found or not a file: {eeg_file}")
    if not eeg_file.suffix == '.fif':
        raise ValueError(f"EEG background data file {eeg_file} is not a .fif file.")

    # Read fif file
    raw = mne.io.read_raw_fif(eeg_file, preload=True)

    return raw

def _build_eeg_montage(raw):
    """ Build EEG montage from raw data and return as xArray object.
    """

    all_pos = raw.get_montage().get_positions()
    ch_pos = all_pos['ch_pos']
    landmark_pos = {"Nz": all_pos["nasion"], 
                    "LPA": all_pos["lpa"], 
                    "RPA": all_pos["rpa"]}
    all_pos = {**ch_pos, **landmark_pos}
    data = np.array(list(all_pos.values())) * 1000 * cedalion.units.mm  # MNE coordinates are always in meters
    labels = list(all_pos.keys())
    types = [PointType.ELECTRODE]*len(ch_pos) + [PointType.LANDMARK]*len(landmark_pos)

    # Wrap into a xArray object
    eeg_montage = xr.DataArray(data=data,
                            dims=['label', 'pos'],
                            coords={'label': labels,
                                    'type': ('label', types)})
    return eeg_montage

def _norm_hemi(h: str) -> str:
    h = h.strip().lower()
    if h in ("lh", "l", "left"):  return "lh"
    if h in ("rh", "r", "right"): return "rh"
    raise ValueError(f"Unknown hemisphere tag: {h!r}")

# def save_data(self, save_dir):
#     # Save data
#     stamp = f"_{'-'.join(self.source_cfg)}_T{self.T_sim}_SNR{(self.snr)}.nc"
#     self.y_sim.to_netcdf(save_dir + 'y_sim' + stamp)
#     self.y_od.to_netcdf(save_dir + 'y_od' + stamp)
#     self.s_temporal_fnirs.to_netcdf(save_dir + 's_fnirs' + stamp)
#     self.A_fnirs.to_netcdf(save_dir + 'A_fnirs' + stamp)

#     self.x_sim.to_netcdf(save_dir + 'x_sim' + stamp)
#     self.x.to_netcdf(save_dir + 'x' + stamp)
#     self.s_temporal_eeg.to_netcdf(save_dir + 's_eeg' + stamp)
#     self.A_eeg.to_netcdf(save_dir + 'A_fnirs' + stamp)

#     self.stim.to_csv(save_dir + 'stim' + stamp)

# def check_directories(self, directories):
#     """
#     Check if directories exist and assign them to class attributes
#     """

#     global segmentation_dir, brain_surface_dir, scalp_surface_dir, eeg_sourcemodel_dir

#     # KG
#     segmentation_dir = directories.get('segmentation_dir', None)
#     brain_surface_dir = directories.get('brain_surface_dir', None)
#     scalp_surface_dir = directories.get('scalp_surface_dir', None)
#     eeg_sourcemodel_dir = directories.get('eeg_sourcemodel_dir', None)

# def initialize_parameters(self, config_file_dir):
#     """
#     Load parameters from configuration file and set them as attributes
#     """

#     with open(config_file_dir, 'r') as file:
#         config = yaml.safe_load(file)

#     # Read attributes from nested dictionary and set as attributes
#     for dic in config.values():
#         for k, v in dic.items():
#             setattr(self, k, v)

#     # Build derived parameters
#     self.Ns = len(self.source_cfg)
#     if self.mean is None:
#         self.mean = [0]*self.Ns
#     if type(self.mean) in [int, float]:
#         self.mean = [self.mean]*self.Ns
    
#     if self.std is None:
#         self.std = [10]*self.Ns
#     if type(self.std) in [int, float]:
#         self.std = [self.std]*self.Ns
    
#     if self.amplitude is None:
#         self.amplitude = [1]*self.Ns
#     if type(self.amplitude) in [int, float]:
#         self.amplitude = [self.amplitude]*self.Ns

#     if self.radii is None:
#         self.radii = [0]*self.Ns
#     if type(self.radii) in [int, float]:
#         self.radii = [self.radii]*self.Ns
