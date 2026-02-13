""" Container for Source Specifications.

It stores specifications for multiple sources, each defined by:
- WHERE: spatial specification (parcel or landmark)
- WHEN: schedule specification (onsets, durations, values)
- WHAT: temporal specification (fNIRS or EEG models). Each source can have either fNIRS or EEG temporal spec,
    and for each of them, different models and models' parameters can be chosen.

Each source is represented by a SourceSpec dataclass, and multiple sources are managed by the SourcesConfig dataclass.

Example usage:
    sources_config = SourcesConfig()
    sources_config.new(['source1', 'source2'])

    sources_config.set_spatial(
        name='source1',
        selector='parcel',
        loc='Precentral_L',
        hemi='LH',
        scale=1.0,
        amp=0.8
    )

    sources_config.set_schedule(
        name='source1',
        onsets=[0, 30, 60],
        durations=5,
        values=1.0
    )

    sources_config.set_temporal_fnirs(
        name='source1',
        model='Gamma',
        model_params={tau: 6, sigma: 3},
        hbo_scale=12.0,
        hbr_scale=-6.0
    )

    sources_config.set_temporal_eeg(
        name='source2',
        model='ERD',
        model_params={'frequency_band': (8, 12), 'erd_drop': 0.6}
    )

    config_dict = sources_config.to_dict()


"""

from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional, Sequence, Union, Dict, Any, Iterator
import cedalion.models.glm as glm
import abc

# ---------- types ----------
Selector = Literal["parcel", "landmark"]
Hemi     = Literal["LH", "RH"]

# ---------- WHERE ----------
@dataclass
class SpatialSpec:
    selector: Selector                 # "parcel" | "landmark"
    loc: List[str]                     # parcel names OR landmark labels
    hemi: Optional[Hemi] = None        # required for parcels, ignored for landmarks
    vertices: Optional[List[int]] = None  # filled in during processing
    scale: float = 1.0
    amp: float = 1.0

    vertices = None

    def set_vertices(self, vertices: List[int]) -> None:

        if not isinstance(vertices, list):
            raise ValueError("Vertices must be provided as a list of integers.")
        if not all(isinstance(v, int) and v >= 0 for v in vertices):
            raise ValueError("Each vertex must be a non-negative integer.")
                
        self.vertices = vertices
        return self

# ---------- WHEN ----------
@dataclass
class ScheduleSpec:
    onsets: List[float] | float  # seconds
    durations: List[float] | float  # seconds
    values: List[float] | float  # amplitude scaling factors

# ---------- WHAT (fNIRS) ----------
@dataclass
class FNIRSTemporalSpec:
    model: Literal["Gamma", "Gaussian"] = "Gamma"
    model_params: Optional[Dict[str, Any]] = None
    hbr_hbo_ratio: float = -0.4
    amp: float = 1.0

# ---------- WHAT (EEG) ----------
@dataclass
class EEGTemporalSpec:
    model: Literal["ERD"] = "ERD"
    model_params: Optional[Dict[str, Any]] = None
    amp: float = 1.0

# ---------- ONE SOURCE CONTAINER ----------
@dataclass
class SourceSpec:
    """ Configuration container for a single source.
    
    Each source is defined by its spatial, schedule, and temporal specifications.
    The spatial specification defines WHERE the source is located, and can be 
    set using the `set_spatial` method via parcel or landmark selectors.
    The schedule specification defines WHEN the source is active, and can be
    set using the `set_schedule` method with onsets, durations, and values.
    The temporal specification defines WHAT the source's temporal profile is,
    and can be set using either the `set_temporal_fnirs` method for fNIRS sources
    or the `set_temporal_eeg` method for EEG sources. Each temporal method allows
    selection of different models and their parameters, each tailored to the modality.
    """

    name: str
    spatial: Optional[SpatialSpec] = None
    schedule: Optional[ScheduleSpec] = None
    temporal_fnirs: Optional[FNIRSTemporalSpec] = None
    temporal_eeg: Optional[EEGTemporalSpec] = None

    # --- WHERE ---
    def set_spatial(
        self,
        selector: Selector,
        loc: str,
        hemi: Optional[Hemi] = None,
        scale: float = 1.0,
        amp: float = 1.0,
    ) -> "SourceSpec":
        """ Set spatial specification for the source.
        Args:
            selector (Selector): 'parcel' or 'landmark' to specify the type of location.
            loc (str): Parcel name(s) or landmark label(s).
            hemi (Optional[Hemi]): 'LH' or 'RH' for parcels; ignored for landmarks.
            scale (float): Spatial scale factor in cm.
            amp (float): Relative amplitude (0 < amp <= 1.0).
        """

        # Validate location
        if not isinstance(loc, str):
            raise ValueError("'loc' must be a string or list of strings.")
        
        # Validate scale
        if not isinstance(scale, (int, float)) and scale > 0:
            raise ValueError("'scale' must be a positive float or int.")
        
        # Validate amp
        if not isinstance(amp, (int, float)) and amp > 0 and amp <= 1.0:
            raise ValueError("'amp' must be a float or int between 0 and 1 representing the relative amplitude.")
        
        # Validate parcel information
        if selector == 'parcel':

            if isinstance(hemi, type(None)):
                pass
            elif isinstance(hemi, str) and hemi.lower() in ['lh', 'rh']:
                pass
            else:
                raise ValueError("hemi must be either 'LH', 'RH', or None.")

        elif selector == "landmark":
            if hemi is not None:
                print("Warning: 'hemi' is ignored when selector is 'landmark'.")
            hemi = None

        else:
            raise ValueError("Selector must be either 'parcel' or 'landmark'.")
        
        self.spatial = SpatialSpec(
            selector=selector,
            loc=loc,
            hemi=hemi,
            scale=scale,
            amp=amp
        )
        return self

    # --- WHEN ---
    def set_schedule(
        self,
        onsets: list[float] | float,
        durations: list[float] | float,
        values: list[float] | float,
    ) -> "SourceSpec":
        """ Set schedule specification for the source.
        
        Args:
            onsets (list[float] | float): Onset times in seconds.
            durations (list[float] | float): Durations in seconds.
            values (list[float] | float): Amplitude scaling factors.
        """
        
        # Validate onsets
        if isinstance(onsets, (int, float)):
            onsets = [float(onsets)]
        elif isinstance(onsets, list):
            if not all(isinstance(x, (int, float)) for x in onsets):
                raise ValueError("All onset values must be int or float.")
        Nevents = len(onsets)

        # Validate durations
        if isinstance(durations, (int, float)):
            durations = [float(durations)] * Nevents
        elif isinstance(durations, list):
            if not all(isinstance(x, (int, float)) for x in durations):
                raise ValueError("All duration values must be int or float.")
            if len(durations) != Nevents:
                raise ValueError("Durations length must match onsets length.")
            
        # Validate values
        if isinstance(values, (int, float)):
            values = [float(values)] * Nevents
        elif isinstance(values, list):
            if not all(isinstance(x, (int, float)) for x in values):
                raise ValueError("All value entries must be int or float.")
            if len(values) != Nevents:
                raise ValueError("Values length must match onsets length.")
        
        self.schedule = ScheduleSpec(
            onsets=onsets,
            durations=durations,
            values=values
        )
        return self
    
    # --- WHAT (fNIRS) ---
    def set_temporal_fnirs(
            self,
            model: Literal['Gamma', 'Gaussian'] = 'Gamma',
            model_params: Optional[Dict[str, Any]] = None,
            hbr_hbo_ratio: float = -0.4,
            amp: float = 1.0,
        ) -> "SourceSpec":
        """ Set fNIRS temporal specification for the source.
        
        Args:
            model (Literal['Gamma', 'Gaussian']): Hemodynamic response model.
            model_params (Optional[Dict[str, Any]]): Model-specific parameters.
            amp (float): Amplitude scaling factor.
        """

        # Validate inputs
        available_basis = [name for name in dir(glm.basis_functions) 
                           if isinstance(getattr(glm.basis_functions, name), abc.ABCMeta)]
        if model not in available_basis:
            raise ValueError("model must be one of the available basis functions in glm.basis_functions: "
                                f"{available_basis}") 
        if model_params is not None and not isinstance(model_params, dict):
            raise ValueError("model_params must be a dictionary if provided.")
        if hbr_hbo_ratio >= 0 or hbr_hbo_ratio < -1:
                raise ValueError(f"Invalid HbR/HbO ratio {hbr_hbo_ratio}, must be in [-1, 0).")
        
        self.temporal_fnirs = FNIRSTemporalSpec(
            model=model,
            amp=amp,
            model_params=model_params,
            hbr_hbo_ratio=hbr_hbo_ratio
        )
        return self
    
    # --- WHAT (EEG) ---
    def set_temporal_eeg(
        self,
        model: Literal["ERD"] = "ERD",
        model_params: Optional[Dict[str, Any]] = None,
        amp: float = 1.0,
    ) -> "SourceSpec":
        """ Set EEG temporal specification for the source.
        Args:
            model (Literal["ERD"]): EEG temporal model.
            model_params (Optional[Dict[str, Any]]): Model-specific parameters.
            amp (float): Amplitude scaling factor.
        """

        self.temporal_eeg = EEGTemporalSpec(
            model=model,
            model_params=model_params
        )
        return self
       
    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "SourceSpec":
        """ Create SourceSpec from dictionary. """
        return cls(
            name=cfg['name'],
            spatial=SpatialSpec(**cfg['spatial']) if cfg.get('spatial') else None,
            schedule=ScheduleSpec(**cfg['schedule']) if cfg.get('schedule') else None,
            temporal_fnirs=FNIRSTemporalSpec(**cfg['temporal_fnirs']) if cfg.get('temporal_fnirs') else None,
            temporal_eeg=EEGTemporalSpec(**cfg['temporal_eeg']) if cfg.get('temporal_eeg') else None,
        )
    
# ---------- MULTIPLE SOURCES CONTAINER ----------
@dataclass
class SourcesConfig:
    """ Configuration container for multiple sources.
    
    Each source is represented by a SourceSpec dataclass, and multiple sources are managed by the SourcesConfig dataclass.
    Check the SourceSpec documentation for details on specifying each source's properties. This container
    provides methods to create, modify, and remove sources, as well as to export the entire configuration as a dictionary.
    For modifying individual source specifications, SourcesConfig provides convenient methods that internally delegate 
    to the corresponding SourceSpec methods, such as `set_spatial`, `set_schedule`, `set_temporal_fnirs`, and `set_temporal_eeg`.
    """
    
    sources: List[SourceSpec] = field(default_factory=list)

     # --- iteration & mapping-like sugar ---
    def __iter__(self) -> Iterator[str]:
        """Iterate over source names (like a dict)."""
        return iter(self.sources)
    
    def __len__(self) -> int:
        return len(self.sources)
                   
    def __getitem__(self, name: str) -> SourceSpec:
        return self.sources[name]


    # --- internals ---
    def _idx(self, name: str) -> int:
        for i, s in enumerate(self.sources):
            if s.name == name:
                return i
        raise KeyError(f"Source '{name}' not found. Call new('{name}') first.")

    # --- lifecycle ---
    def new(self, names: list[str] | str) -> "SourcesConfig":
        """Create empty sources if they don't exist (no-op if they do)."""
        
        if not isinstance(names, (list, str)):
            raise TypeError("names must be str or list of str.")
        
        names = [names] if isinstance(names, str) else names
        for name in names: 
            if not isinstance(name, str) or name == "":
                raise TypeError("Each name must be a non-empty string.")
            elif name in [s.name for s in self.sources]:
                print(f"Source '{name}' already exists. Skipping.")
            else:
                self.sources.append(SourceSpec(name=name))

        return self

    def remove(self, names: list[str] | str) -> "SourcesConfig":
        """ Remove sources by name."""
        names = [names] if isinstance(names, str) else names
        for name in names:
            i = self._idx(name)
            del self.sources[i]
        return self

    # --- WHERE ---
    def set_spatial(
        self,
        name: str,
        selector: Selector,
        loc: Union[str, Sequence[str]],
        hemi: Optional[Hemi] = None,
        scale: float = 1.0,
        amp: float = 1.0,
    ) -> "SourcesConfig":
        
        i = self._idx(name)

        self.sources[i].set_spatial(
            selector=selector,
            loc=loc,
            hemi=hemi,
            scale=scale,
            amp=amp
        )
        return self

    # --- WHEN ---
    def set_schedule(
            self,
            name: str,
            onsets: list[float] | float,
            durations: list[float] | float,
            values: list[float] | float,
        ) -> "SourcesConfig":
        
        i = self._idx(name)
        self.sources[i].set_schedule(
            onsets=onsets,
            durations=durations,
            values=values
        )
        return self

    # --- WHAT (fNIRS) ---
    def set_temporal_fnirs(
        self,
        name: str,
        model: Literal['Gamma', 'Gaussian'] = 'Gamma',
        model_params: Optional[Dict[str, Any]] = None,
        hbr_hbo_ratio: float = -0.4,
        amp: float = 1.0,
    ) -> "SourcesConfig":
        
        i = self._idx(name)
        self.sources[i].set_temporal_fnirs(
            model=model,
            model_params=model_params,
            hbr_hbo_ratio=hbr_hbo_ratio,
            amp=amp
        )
        return self

    # --- WHAT (EEG) ---
    def set_temporal_eeg(
        self,
        name: str,
        model: Literal["ERD"] = "ERD",
        model_params: Optional[Dict[str, Any]] = None,
        amp: float = 1.0,
    ) -> "SourcesConfig":
        i = self._idx(name)
        self.sources[i].set_temporal_eeg(
            model=model,
            model_params=model_params,
            amp=amp
        )
        return self

    @classmethod
    def from_list(cls, cfg_list: List[Dict[str, Any]]) -> "SourcesConfig":
        """ Create SourcesConfig from list of dictionaries. """
        sources = [SourceSpec.from_dict(cfg) for cfg in cfg_list]
        return cls(sources=sources)

    # --- export ---
    def to_dict(self) -> Dict[str, Any]:
        return {"sources": [asdict(s) for s in self.sources]}
