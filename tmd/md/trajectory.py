from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from openmm import app

from tmd.fe.cif_writer import build_openmm_topology, convert_single_topology_mols
from tmd.fe.single_topology import SingleTopology
from tmd.fe.topology import BaseTopology, MultiTopology
from tmd.md.builders import HostConfig

TMDTopologyType = BaseTopology | SingleTopology


def get_openmm_topology(tmd_topology: TMDTopologyType, host_config: HostConfig | None) -> app.Topology:
    """Convert a TMD topology object and host config object into an OpenMM topology.

    This function combines the molecules in the TMD topology
    and optional host configuration into a single OpenMM Topology object.

    Parameters
    ----------
    tmd_topology : TMDTopologyType
        The topology object containing the molecular information. Can be a BaseTopology,
        MultiTopology, or SingleTopology. All molecules will be included in the topology.
    host_config : HostConfig or None
        The host configuration object, which may contain additional topology information.
        If provided, the host topology is added to the output topology first.

    Returns
    -------
    openmm.app.Topology
        An OpenMM Topology object representing the combined system.

    Raises
    ------
    ValueError
        If the provided tmd_topology is of an unknown type.
    """
    topo_objs = []
    if host_config is not None:
        topo_objs.append(host_config.omm_topology)
    if isinstance(tmd_topology, BaseTopology):
        if isinstance(tmd_topology, MultiTopology):
            topo_objs.extend(tmd_topology.mols)
        else:
            topo_objs.append(tmd_topology.mol)
    elif isinstance(tmd_topology, SingleTopology):
        topo_objs.append(tmd_topology.mol_a)
        topo_objs.append(tmd_topology.mol_b)
    else:
        raise ValueError(f"Unknown topology type: {type(tmd_topology)}")
    return build_openmm_topology(topo_objs)


def frames_to_mdtraj_trajectory(
    coords: Iterable[NDArray],
    boxes: Iterable[NDArray],
    tmd_topology: TMDTopologyType,
    host_config: HostConfig | None,
    time: Sequence[float] | None,
):
    """Convert trajectory frames into an MDTraj Trajectory object.

    Parameters
    ----------
    coords : iterable of NDArray
        An iterable of coordinate arrays, where each array contains the coordinates
        of the system for a given frame.
    boxes : iterable of NDArray
        An iterable of box arrays, where each array represents the periodic boundary
        conditions for a given frame.
    tmd_topology : TMDTopologyType
        The topology object containing the molecular information.
    host_config : HostConfig or None
        The host configuration object, which may contain additional topology information.
    time : sequence of float or None
        The time associated with each frame in picoseconds. If provided, it must match the length of
        the coordinates and boxes.

    Returns
    -------
    mdtraj.Trajectory
        An MDTraj Trajectory object containing the converted frames and topology. Each ligand is
        stored separately in a residue named LIG.

    Raises
    ------
    RuntimeError
        If MDTraj fails to import, if the number of frames does not match the number of boxes,
        or if the number of frames does not match the provided time sequence.
    ValueError
        If any value in the provided time sequence is not greater than 0.

    Notes
    -----
    If the topology object is a SingleTopology object, be aware of the lambda value that was
    used to generate the frames. Endstates will contain alchemical coordinates for one of the
    molecules.
    """
    try:
        import mdtraj as md
    except ImportError as e:
        raise RuntimeError("MDTraj failed to import, check that it is installed") from e

    openmm_top = get_openmm_topology(tmd_topology, host_config=host_config)
    md_top = md.Topology.from_openmm(openmm_top)

    host_atoms = host_config.conf.shape[0] if host_config is not None else 0
    single_top_conversion = isinstance(tmd_topology, SingleTopology)

    def convert_frame(x):
        if not single_top_conversion:
            return x
        ligand_frame = convert_single_topology_mols(x[host_atoms:], tmd_topology)
        if host_atoms > 0:
            frame = np.concatenate([x[:host_atoms], ligand_frame])
        else:
            frame = ligand_frame
        return frame

    frames = [convert_frame(frame) for frame in coords]

    unitcell_lengths = [np.diag(box) for box in boxes]
    if len(unitcell_lengths) != len(frames):
        raise RuntimeError(f"Number of frames and boxes don't match: {len(frames)} != {len(unitcell_lengths)}")
    if time is not None:
        if len(time) != len(frames):
            raise RuntimeError(f"Number of frames and time don't match: {len(frames)} != {len(time)}")
        if not np.all(np.asarray(time) > 0):
            raise ValueError("All trajectory time must be greater than 0")

    # Currently boxes are always cubic
    unitcell_angles = [[90.0] * 3] * len(unitcell_lengths)

    return md.Trajectory(frames, md_top, time=time, unitcell_lengths=unitcell_lengths, unitcell_angles=unitcell_angles)
