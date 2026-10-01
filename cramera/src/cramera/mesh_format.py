"""
The mesh file formats the viewer can load.
"""

from __future__ import annotations

from enum import StrEnum

from typing_extensions import Optional, Tuple


class MeshFormat(StrEnum):
    """
    A mesh format, identified by the file suffix its files carry.
    """

    STL = ".stl"
    """
    Stereolithography, the format loose demo objects are recorded as.
    """

    OBJ = ".obj"
    """
    Wavefront OBJ, which references its materials through a side ``.mtl`` file.
    """

    DAE = ".dae"
    """
    COLLADA, which references its textures from inside the document.
    """

    GLB = ".glb"
    """
    Binary glTF, which carries its geometry, materials and textures in one file.
    """

    @property
    def names_side_assets(self) -> bool:
        """
        Whether a file of this format names its materials or textures in files beside
        it, which a copy of the mesh has to bring along.
        """
        return self in (MeshFormat.OBJ, MeshFormat.DAE)

    @classmethod
    def suffixes(cls) -> Tuple[str, ...]:
        """
        Every known mesh suffix, in the order the members are declared.
        """
        return tuple(member.value for member in cls)

    @classmethod
    def of_path(cls, path: str) -> Optional[MeshFormat]:
        """
        The format a path names, or None when it names no mesh at all.

        :param path: A file name, path or URI whose suffix is inspected.
        """
        lowered = path.lower()
        for member in cls:
            if lowered.endswith(member.value):
                return member
        return None
