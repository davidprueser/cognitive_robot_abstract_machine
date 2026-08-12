from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, ClassVar, Self, assert_never

import numpy as np

from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageWithTypeDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    RoomWithWallsAndDoors,
    Floor,
    Wall,
    Door,
    DoorWithType,
    Handle,
    Hinge,
    ShelfLayer,
    Cabinet,
    Table,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    Connection6DoF,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
    WorldEntity,
    Body,
)


@dataclass
class EGBase(SubclassJSONSerializer):
    pass


@dataclass
class EGWithID(EGBase):
    id: str

    def create_in_world(
        self,
        world: World,
        object_id_to_mesh_path: dict[str, Path] | None,
        parent: KinematicStructureEntity,
        **kwargs,
    ) -> WorldEntity:
        """
        Create the object in the world by getting its geometry from the
        provided information.

        :param world: The world where the object is created.
        :param object_id_to_mesh_path: A mapping from an object's id to its
            mesh directory path.
        :param parent: The parent of the object in the world.
        :param kwargs: Additional keyword arguments.
        :return: The relevant created body
        """


@dataclass
class EGScale(EGBase):
    """
    The scale of an object.
    """

    height: float
    """
    Scale in z (vertical axis).
    """

    length: float
    """
    Depth of the object, i.e. its shallow front-to-back extent for a shelf.
    """

    width: float
    """
    Face extent of the object, i.e. the wide side a shelf's contents line up
    along.
    """

    def to_json(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "length": self.length,
            "width": self.width,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            height=data["height"],
            length=data["length"],
            width=data["width"],
        )


@dataclass
class EGPoint2D(EGBase):
    x: float
    y: float

    def rotated_into_frame(self, frame_yaw_degrees: float) -> EGPoint2D:
        """
        Express this point, currently an offset along the world axes, in the
        axes of a frame rotated by *frame_yaw_degrees*.

        Needed wherever an object's offset from a rotated parent is stored for
        later re-use *inside* that parent: keeping the offset on the world axes
        makes it mean something different once the parent's own rotation is
        applied again.

        :param frame_yaw_degrees: Yaw of the target frame, in degrees.
        :return: The same offset expressed in the target frame's axes.
        """
        frame_yaw_radians = math.radians(frame_yaw_degrees)
        cosine = math.cos(frame_yaw_radians)
        sine = math.sin(frame_yaw_radians)
        return EGPoint2D(
            x=self.x * cosine + self.y * sine,
            y=-self.x * sine + self.y * cosine,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            x=data["x"],
            y=data["y"],
        )


@dataclass
class EGPosition(EGPoint2D):
    z: float

    def to_json(self) -> dict[str, Any]:
        json = super().to_json()
        json.update({"z": self.z})
        return json

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:

        return cls(
            x=data["x"],
            y=data["y"],
            z=data["z"],
        )


@dataclass
class EGRotation(EGPoint2D):
    """
    Rotation of an object, expressed as roll, pitch, and yaw in degrees.

    Inherits ``x`` (roll) and ``y`` (pitch) from :class:`EGPoint2D`;
    only ``z`` (yaw) varies for objects that sit upright without
    tilting.
    """

    z: float
    """
    Yaw, in degrees, about the local z-axis (vertical axis).
    """

    def as_roll_pitch_yaw_in_radians(self) -> tuple[float, float, float]:
        """
        Convert this rotation into a roll, pitch, yaw tuple in radians.

        :return:``(roll, pitch, yaw)`` in radians.
        """
        conversion_factor = np.pi / 180
        return (
            self.x * conversion_factor,
            self.y * conversion_factor,
            self.z * conversion_factor,
        )

    def to_json(self) -> dict[str, Any]:
        json = super().to_json()
        json.update({"z": self.z})
        return json

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            x=data["x"],
            y=data["y"],
            z=data["z"],
        )


class ObjectType(StrEnum):
    """
    Generalized object categories that unify the tens of thousands of distinct,
    near-instance-specific ``object_type`` strings found in the raw sage10k
    dataset (for example ``"book1"``, ``"book_table2"`` and
    ``"bookchair8eba7fdc"`` all belong to the same real-world category of
    object).

    .. note::
        Mapping the raw sage10k strings onto these generalized members is
        handled separately; this enum only defines the target categories.
    """

    APPAREL = "apparel"
    ART = "art"
    BAG = "bag"
    BASKET = "basket"
    BATHTUB = "bathtub"
    BED = "bed"
    BENCH = "bench"
    BIN = "bin"
    BOOK = "book"
    BOTTLE = "bottle"
    BOWL = "bowl"
    BOX = "box"
    BUCKET = "bucket"
    CABINET = "cabinet"
    CAMERA = "camera"
    CANDLE = "candle"
    CART = "cart"
    CHAIR = "chair"
    CHANDELIER = "chandelier"
    CLOCK = "clock"
    COMPUTER = "computer"
    CONTAINER = "container"
    COUNTER = "counter"
    CRATE = "crate"
    CUP = "cup"
    CUTLERY = "cutlery"
    CUTTING_BOARD = "cutting_board"
    DESK = "desk"
    DISHWASHER = "dishwasher"
    DISPENSER = "dispenser"
    DISPLAYCASE = "displaycase"
    DOOR = "door"
    DRESSER = "dresser"
    DRYER = "dryer"
    FIREPLACE = "fireplace"
    FOOD = "food"
    FOUNTAIN = "fountain"
    FRAME = "frame"
    GLASS = "glass"
    HARDWARE = "hardware"
    JAR = "jar"
    KEYBOARD = "keyboard"
    KNIFE = "knife"
    LADDER = "ladder"
    LAMP = "lamp"
    LIGHT_FIXTURE = "light_fixture"
    LOCKER = "locker"
    MAGAZINE = "magazine"
    MICROWAVE = "microwave"
    MIRROR = "mirror"
    MONITOR = "monitor"
    MOUSE = "mouse"
    NEON_SIGN = "neon_sign"
    NIGHTSTAND = "nightstand"
    OFFICE_SUPPLY = "office_supply"
    OTHER = "other"
    OVEN = "oven"
    PANEL = "panel"
    PANTRY = "pantry"
    PEDESTAL = "pedestal"
    PEGBOARD = "pegboard"
    PEN = "pen"
    PERSONAL_CARE_PRODUCT = "personal_care_product"
    PHONE = "phone"
    PILLOW = "pillow"
    PLANT = "plant"
    PLATE = "plate"
    POT = "pot"
    PRINTER = "printer"
    PROJECTOR = "projector"
    REFRIGERATOR = "refrigerator"
    REMOTE_CONTROL = "remote_control"
    RETAIL_FIXTURE = "retail_fixture"
    SAFETY_EQUIPMENT = "safety_equipment"
    SCULPTURE = "sculpture"
    SHELF = "shelf"
    SIDEBOARD = "sideboard"
    SIGN = "sign"
    SINK = "sink"
    SMALL_APPLIANCE = "small_appliance"
    SOFA = "sofa"
    SPEAKER = "speaker"
    SPORTS_EQUIPMENT = "sports_equipment"
    STAND = "stand"
    TABLE = "table"
    TAPESTRY = "tapestry"
    TELEVISION = "television"
    TEXTILE = "textile"
    TOILET = "toilet"
    TOOL = "tool"
    TOOLBOX = "toolbox"
    TOWEL = "towel"
    TOY = "toy"
    TRASH = "trash"
    TRAY = "tray"
    UTENSIL = "utensil"
    VANITY = "vanity"
    VASE = "vase"
    VEHICLE = "vehicle"
    VENT = "vent"
    WARDROBE = "wardrobe"
    WASHING_MACHINE = "washing_machine"
    WINDOW = "window"
    WORKBENCH = "workbench"


# %%
@dataclass
class EGObject(EGWithID):
    room_id: str
    """
    The id of the room where the object is located.
    """

    place_id: str
    """
    The id of the object where the object is located/placed on/at, e.g. wall,
    floor, anchor. The room-structure values are enumerated by :class:`PlaceId`.
    """

    object_type: ObjectType
    """
    The type of the object.
    """

    scale: EGScale
    """
    The scale of the object.
    """

    position: EGPosition
    """
    The position of the object.
    """

    orientation: EGRotation
    """
    The orientation of the object.
    """

    source_id: str
    """
    Id of the object.

    This is used to identify the object in the dataset.
    """

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "id": self.id,
            "room_id": self.room_id,
            "type": self.object_type,
            "place_id": self.place_id,
            "position": to_json(self.position),
            "rotation": to_json(self.orientation),
            "dimensions": to_json(self.scale),
            "source_id": self.source_id,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs):
        return cls(
            id=data["id"],
            room_id=data["room_id"],
            object_type=ObjectType._value2member_map_.get(
                data["type"], ObjectType.OTHER
            ),
            place_id=data["place_id"],
            position=EGPosition._from_json(data["position"], **kwargs),
            orientation=EGRotation._from_json(data["rotation"], **kwargs),
            scale=EGScale._from_json(data["dimensions"], **kwargs),
            source_id=data["source_id"],
        )

    def create_in_world(
        self,
        world: World,
        mesh_path: Path | None,
        parent: KinematicStructureEntity,
        world_pose: HomogeneousTransformationMatrix | None = None,
        **kwargs,
    ) -> Body:
        """
        Instantiate this object in *world* by loading its PLY mesh from
        *mesh_path*.

        The mesh keeps its own native real-world size, since sage10k PLY assets
        already carry their real dimensions; collisions are checked against that
        real mesh, so stretching it to an independently sampled scale would both
        distort it and disagree with the geometry the layout is resolved
        against.

        Walls are attached with a fixed connection; every other object is
        attached with a movable 6-DoF connection whose pose lives in its degrees
        of freedom, so a resolver can reposition it in place via the ``origin``
        setter.

        :param world: The world where the object is created.
        :param mesh_path: Directory containing the ``objects/`` sub-
            folder with PLY and texture files for this object.
        :param parent: The parent kinematic structure entity.
        :param world_pose: When given, the body is placed at this pose instead
            of the one built from :attr:`position`/ :attr:`orientation`, so a
            caller that already computed the pose can reuse it.
        :raises ValueError: If *mesh_path* is ``None`` or does not exist.
        :return: The created :class:`Body`.
        """
        if mesh_path is None:
            raise ValueError(
                f"No mesh path resolved for object {self.id!r} "
                f"(source_id={self.source_id!r})."
            )
        if not mesh_path.exists():
            raise ValueError(f"Directory {mesh_path} does not exist.")
        ply_file = mesh_path / "objects" / f"{self.source_id}.ply"
        texture_file = mesh_path / "objects" / f"{self.source_id}_texture.png"

        body = Body()
        body.name = PrefixedName(name=str(body.id), prefix=self.id)

        if world_pose is not None:
            world_pose.child_frame = body
            root_T_body = world_pose
        else:
            root_T_body = HomogeneousTransformationMatrix.from_xyz_rpy(
                self.position.x,
                self.position.y,
                self.position.z,
                *self.orientation.as_roll_pitch_yaw_in_radians(),
                reference_frame=parent,
                child_frame=body,
            )

        # sage10k meshes already carry their real-world size, so spawn at
        # identity scale to keep the mesh's true proportions instead of
        # stretching it to an independently sampled scale.
        mesh = Mesh.from_ply_file(
            ply_file_path=str(ply_file),
            texture_file_path=str(texture_file),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=body),
            scale=Scale(1.0, 1.0, 1.0),
        )

        geometry = ShapeCollection([mesh], reference_frame=body)
        body.visual = geometry
        body.collision = geometry

        if self.place_id == "wall":
            with world.modify_world():
                root_C_body = FixedConnection.create_with_dofs(
                    world=world,
                    parent=parent,
                    child=body,
                    parent_T_connection_expression=root_T_body,
                )
                world.add_body(body)
                world.add_connection(root_C_body)
        else:
            with world.modify_world():
                root_C_body = Connection6DoF.create_with_dofs(
                    world=world,
                    parent=parent,
                    child=body,
                )
                world.add_body(body)
                world.add_connection(root_C_body)
            # Placing the pose in the connection's degrees of freedom rather than
            # in a fixed parent expression keeps the object movable: the
            # ``.origin`` setter can later reposition it in place.
            body.parent_connection.origin = root_T_body

        annotation = NaturalLanguageWithTypeDescription(
            root=body, description=None, type_description=self.object_type
        )

        with world.modify_world():
            world.add_semantic_annotation(annotation)

        return body


@dataclass
class EGObject2D(EGWithID):
    """
    An object on a shelf layer — position is 2-D since z is determined by the
    layer.
    """

    room_id: str
    """
    The id of the room where the object is located.
    """

    place_id: str
    """
    The id of the surface the object rests on, e.g. a shelf, floor, or wall.
    """

    object_type: ObjectType
    """
    The category of the object.
    """

    scale: EGScale
    """
    Physical dimensions of the object.
    """

    position: EGPoint2D
    """
    2-D position relative to the centre of the containing shelf layer.
    """

    orientation: EGRotation
    """
    Orientation of the object in Euler angles (degrees).
    """

    source_id: str
    """
    Identifier used to look up the PLY mesh file for this object in the
    dataset.
    """

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "id": self.id,
            "room_id": self.room_id,
            "type": self.object_type,
            "place_id": self.place_id,
            "position": to_json(self.position),
            "rotation": to_json(self.orientation),
            "dimensions": to_json(self.scale),
            "source_id": self.source_id,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            id=data["id"],
            room_id=data["room_id"],
            object_type=ObjectType._value2member_map_.get(
                data["type"], ObjectType.OTHER
            ),
            place_id=data["place_id"],
            position=EGPoint2D._from_json(data["position"], **kwargs),
            orientation=EGRotation._from_json(data["rotation"], **kwargs),
            scale=EGScale._from_json(data["dimensions"], **kwargs),
            source_id=data["source_id"],
        )

    def create_in_world(
        self,
        world: World,
        mesh_path: Path | None,
        parent: KinematicStructureEntity,
        x: float | None = None,
        y: float | None = None,
        z: float = 0.0,
        world_pose: HomogeneousTransformationMatrix | None = None,
        **kwargs,
    ) -> Body:
        """
        Instantiate this object in *world* at the given absolute pose.

        The mesh keeps its own native real-world size, since sage10k PLY assets
        already carry their real dimensions; stretching them to an independently
        sampled scale would distort them.

        :param world: The world where the object is created.
        :param mesh_path: Directory containing the ``objects/`` sub-
            folder with PLY and texture files for this object.
        :param parent: The parent kinematic structure entity.
        :param x: Absolute x in world coordinates (defaults to
            ``self.position.x``).
        :param y: Absolute y in world coordinates (defaults to
            ``self.position.y``).
        :param z: Absolute z in world coordinates.
        :param world_pose: When given, the body is placed at this pose and
            *x*, *y*, *z* are ignored, so a caller that already computed the
            pose can reuse it for both spawning and later repositioning.
        :raises ValueError: If *mesh_path* is ``None`` or does not exist.
        :return: The created :class:`Body`.
        """
        if mesh_path is None:
            raise ValueError(
                f"No mesh path resolved for object {self.id!r} "
                f"(source_id={self.source_id!r})."
            )
        if not mesh_path.exists():
            raise ValueError(f"Directory {mesh_path} does not exist.")
        ply_file = mesh_path / "objects" / f"{self.source_id}.ply"
        texture_file = mesh_path / "objects" / f"{self.source_id}_texture.png"

        body = Body()
        body.name = PrefixedName(name=str(body.id), prefix=self.id)

        if world_pose is not None:
            world_pose.child_frame = body
            root_T_body = world_pose
        else:
            root_T_body = HomogeneousTransformationMatrix.from_xyz_rpy(
                self.position.x if x is None else x,
                self.position.y if y is None else y,
                z,
                *self.orientation.as_roll_pitch_yaw_in_radians(),
                reference_frame=parent,
                child_frame=body,
            )

        # sage10k meshes already carry their real-world size, so spawn at
        # identity scale to keep the mesh's true proportions instead of
        # stretching it to an independently sampled scale.
        mesh = Mesh.from_ply_file(
            ply_file_path=str(ply_file),
            texture_file_path=str(texture_file),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=body),
            scale=Scale(1.0, 1.0, 1.0),
        )

        geometry = ShapeCollection([mesh], reference_frame=body)
        body.visual = geometry
        body.collision = geometry

        with world.modify_world():
            root_C_body = Connection6DoF.create_with_dofs(
                world=world,
                parent=parent,
                child=body,
            )
            world.add_body(body)
            world.add_connection(root_C_body)

        # Placing the pose in the connection's degrees of freedom rather than in
        # a fixed parent expression keeps the object movable: the ``.origin``
        # setter can later reposition it in place.
        body.parent_connection.origin = root_T_body

        annotation = NaturalLanguageWithTypeDescription(
            root=body, description=None, type_description=self.object_type
        )

        with world.modify_world():
            world.add_semantic_annotation(annotation)

        return body


@dataclass
class EGShelfLayer(EGBase):
    """
    A shelf layer for environment generation.

    Carries its own physical dimensions so the RSPN can learn width and
    length alongside object placement, rather than inheriting a fixed
    size from the parent shelf.
    """

    scale: EGScale
    """
    Physical dimensions of the layer slab (width × length × height).
    """

    objects: list[EGObject2D]
    """
    Objects placed on this layer, with positions relative to the layer centre.
    """

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "scale": to_json(self.scale),
            "objects": to_json(self.objects),
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            scale=EGScale._from_json(data["scale"], **kwargs),
            objects=[EGObject2D._from_json(o, **kwargs) for o in data["objects"]],
        )


@dataclass(frozen=True)
class MeshCandidate:
    """
    A mesh asset available for rendering a sampled object, together with the
    generalized object type it was captured from.
    """

    scene_dir: Path
    """
    Directory containing the ``objects/`` sub-folder with this mesh's PLY and
    texture files.
    """

    source_id: str
    """
    Identifier used to look up this mesh's PLY and texture files within
    :attr:`scene_dir`.
    """

    object_type: ObjectType
    """
    The generalized category of the object this mesh was captured from.
    """

    native_extents: tuple[float, float, float] | None = None
    """
    The mesh's own real-world size as ``(width, length, height)``, used to decide
    whether it fits a target space. ``None`` when the size is unknown, in which
    case the candidate is treated as always fitting. A tuple (not an
    :class:`EGScale`) keeps :class:`MeshCandidate` hashable.
    """


@dataclass
class _MeshTypeMatcher:
    """
    Selects, from a pool of candidate meshes, a random one captured from an
    object of the same :class:`ObjectType`.

    Object-type labels in the source dataset are effectively per-
    instance identifiers rather than real categories, so grouping meshes by
    their already-generalized :class:`ObjectType` -- rather than matching
    declared size -- is what keeps a randomly-drawn mesh semantically
    plausible for the category an object was sampled as.

    .. note::
        If the pool holds no mesh of the requested type, ``None`` is returned
        rather than a mesh of some other type. Substituting was what strewed
        generated rooms with arbitrary objects: the cache holds only a few
        hundred floor-capable meshes across dozens of types, so a sampled bed or
        sofa routinely became whichever mesh happened to be drawn.
    """

    MAXIMUM_SIZE_RATIO: ClassVar[float] = 2.0
    """
    How far a candidate's real size may differ from a requested target size, as
    a factor on each axis, before it is rejected.

    A mesh of the right category but the wrong size still looks wrong -- a
    sampled 0.45 m stool spawning as a 1.2 m armchair -- so category alone is
    not enough once the circuit has sampled a size to aim for.
    """

    candidates: list[MeshCandidate]
    """
    Pool of meshes to choose from.
    """

    def random_match(
        self,
        object_type: ObjectType,
        max_extents: EGScale | None = None,
        target_extents: EGScale | None = None,
    ) -> MeshCandidate | None:
        """
        Return a candidate whose :attr:`MeshCandidate.object_type` equals
        *object_type*, or ``None`` when the pool holds none.

        *max_extents* is an upper bound: candidates larger than it on any axis
        are ineligible, which is how shelf contents are kept from piercing the
        layer above. *target_extents* is a size to aim for: candidates further
        than :attr:`MAXIMUM_SIZE_RATIO` from it on any axis are ineligible, and
        the closest remaining one is returned rather than a random one.

        :param object_type: The category of the object a mesh is selected for.
        :param max_extents: Upper bound on the mesh's width/length/height.
        :param target_extents: Size the mesh should match as closely as possible.
        :return: The selected candidate, or ``None`` when nothing is eligible.
        """
        pool = [
            candidate
            for candidate in self.candidates
            if candidate.object_type == object_type
        ]
        if max_extents is not None:
            pool = [
                candidate for candidate in pool if self._fits(candidate, max_extents)
            ]
        if target_extents is None:
            return random.choice(pool) if pool else None

        scored = [
            (self._size_mismatch(candidate, target_extents), candidate)
            for candidate in pool
        ]
        eligible = [
            (mismatch, candidate)
            for mismatch, candidate in scored
            if mismatch <= math.log(self.MAXIMUM_SIZE_RATIO)
        ]
        if not eligible:
            return None
        return min(eligible, key=lambda scored_candidate: scored_candidate[0])[1]

    @staticmethod
    def _size_mismatch(candidate: MeshCandidate, target_extents: EGScale) -> float:
        """
        How far *candidate*'s real size is from *target_extents*, as the largest
        absolute log-ratio across the three axes.

        A log-ratio is used so that being twice too large and half too large
        count equally. Candidates of unknown size score as a perfect match,
        since there is nothing to judge them on and dropping them would thin an
        already sparse pool.

        :param candidate: The mesh candidate to score.
        :param target_extents: The size the mesh should match.
        :return: The mismatch, zero being an exact match.
        """
        native = candidate.native_extents
        if native is None:
            return 0.0
        targets = (target_extents.width, target_extents.length, target_extents.height)
        return max(
            abs(math.log(measured / target))
            for measured, target in zip(native, targets)
            if measured > 0 and target > 0
        )

    @staticmethod
    def _fits(candidate: MeshCandidate, max_extents: EGScale) -> bool:
        """
        Whether *candidate*'s real-world size stays within *max_extents* on
        every axis. Candidates of unknown size are treated as fitting.

        :param candidate: The mesh candidate to test.
        :param max_extents: Per-axis upper bound.
        :return: ``True`` if the candidate fits or its size is unknown.
        """
        native = candidate.native_extents
        if native is None:
            return True
        native_width, native_length, native_height = native
        return (
            native_width <= max_extents.width
            and native_length <= max_extents.length
            and native_height <= max_extents.height
        )


@dataclass
class SpawnedLayout:
    """
    A generated layout instantiated in a :class:`World`.

    Base for the per-generator spawn results, so the in-world resolver can spawn,
    repair, and return any layout through one type.
    """

    world: World
    """
    The world the layout was spawned into.
    """


@dataclass
class SpawnedShelfLayer:
    """
    Runtime handles to one shelf layer instantiated in a :class:`World`.

    Positionally aligned with :attr:`SpawnedShelf.layers`, so the ``i``-th entry
    corresponds to the ``i``-th layer of the spawned shelf.
    """

    surface: ShelfLayer
    """
    The layer's supporting-surface annotation in the world.
    """

    object_bodies: dict[int, Body]
    """
    Bodies spawned for the layer's objects, keyed by their index in
    :attr:`EGShelfLayer.objects`; objects skipped at spawn have no entry.
    """


@dataclass
class SpawnedShelf(SpawnedLayout):
    """
    A shelf instantiated in a :class:`World`, with handles for in-world
    validation and repositioning of its objects.
    """

    parent: KinematicStructureEntity
    """
    The frame the shelf's objects' poses are expressed relative to.
    """

    layers: list[SpawnedShelfLayer]
    """
    Per-layer spawn handles, in shelf-layer order.
    """

    corpus: Body
    """
    The shelf corpus's body, so a caller can check objects for collision
    against its walls in addition to each other.
    """


@dataclass
class EGShelf(EGBase):
    """
    A shelf with four explicit horizontal layers.
    """

    _CORPUS_WALL_THICKNESS: ClassVar[float] = 0.03
    """
    Thickness of the spawned :class:`Cabinet` corpus's walls. The corpus is
    sized larger than the layers' own footprint by this amount (see
    :meth:`spawn_in_world`), so a wall carved out of that footprint never
    intrudes into the region objects were trained to occupy.
    """

    CONTENT_FRAME_YAW_OFFSET_DEGREES: ClassVar[float] = 90.0
    """
    Yaw offset, in degrees, between a shelf's stored orientation and the frame
    its contents are expressed in.

    In the dataset a shelf's contents spread along its wide face, which lies
    along the shelf's local x-axis -- but the spawned :class:`Cabinet` corpus
    keeps its depth on x (its opening is fixed to -x) and its face on y. This
    offset rotates the content frame so the face spread lands on the corpus's
    wide (width) axis instead of overflowing its shallow depth. Extraction and
    :meth:`spawn_in_world` must apply the *same* offset so the two stay inverses.

    ..note:: The sign decides whether the shelf's open face points toward or
        away from the viewer; it is chosen by inspecting the render, not derived.
    """

    _LAYER_SLAB_THICKNESS: ClassVar[float] = 0.02
    """
    Thickness, in metres, of each spawned layer slab.
    """

    _OBJECT_VERTICAL_MARGIN: ClassVar[float] = 0.01
    """
    Slack, in metres, kept between the tallest object a layer accepts and the
    surface above it, so a fitting object never grazes the next slab or ceiling.
    """

    position: EGPoint2D
    """
    Position of the Shelf, relative to its parent frame.
    """

    scale: EGScale
    """
    Scale of the Shelf.
    """

    orientation: EGRotation
    """
    Orientation of the Shelf, relative to its parent frame.
    """

    layers: list[EGShelfLayer]
    """
    The layers of the Shelf.
    """

    source_ids: list[MeshCandidate] | None = field(default=None)
    """
    Pool of candidate meshes used when placing objects on shelf layers.
    """

    @property
    def corpus_footprint(self) -> EGScale:
        """
        The footprint the spawned corpus occupies, in the shelf's own frame.

        Padded by twice the corpus wall thickness so the carved-out interior is
        at least as large as the layers' own footprint -- otherwise a wall
        intrudes into the region objects were trained to occupy, and an object
        placed near the training data's edge margin collides with it (most
        visible on small shelves, where that margin is thinner than the wall).
        A caller placing a shelf against a room wall has to reserve this, not
        the layers' bare footprint, or the corpus reaches through by the pad.

        .. note::
            :attr:`CONTENT_FRAME_YAW_OFFSET_DEGREES` and the corpus's own
            depth-on-x convention cancel, so the span this footprint covers when
            turned by :attr:`orientation` is the span the corpus really
            occupies.
        """
        wall_margin = 2 * self._CORPUS_WALL_THICKNESS
        return EGScale(
            width=max(layer.scale.width for layer in self.layers) + wall_margin,
            length=max(layer.scale.length for layer in self.layers) + wall_margin,
            height=self.scale.height,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "position": to_json(self.position),
            "scale": to_json(self.scale),
            "orientation": to_json(self.orientation),
            "layers": to_json(self.layers),
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            position=EGPoint2D._from_json(data["position"], **kwargs),
            scale=EGScale._from_json(data["scale"], **kwargs),
            orientation=EGRotation._from_json(data["orientation"], **kwargs),
            layers=[EGShelfLayer._from_json(l, **kwargs) for l in data["layers"]],
        )

    def object_local_pose(
        self,
        obj: EGObject2D,
        origin_z: float,
        corpus: KinematicStructureEntity,
    ) -> HomogeneousTransformationMatrix:
        """
        Compute an object's pose in the shelf corpus's own frame.

        Objects are children of the corpus, so their pose is expressed relative
        to it: the corpus already carries the shelf's world position and yaw, and
        moving the corpus moves every object with it. The pose is therefore just
        the object's on-shelf offset (``position.x``/``y`` map straight onto the
        corpus x/y axes, which span the layer's length/width) at height
        *origin_z*, with its own orientation. Used both when first seating an
        object and when moving it to a resampled pose, so the two placements can
        never drift apart -- and it stays correct after the whole shelf is
        repositioned.

        :param obj: The object whose pose is computed.
        :param origin_z: Height of the object body's origin in the corpus frame.
        :param corpus: The shelf corpus body the pose is expressed relative to.
        :return: The object body's pose in the corpus frame.
        """
        roll, pitch, yaw = obj.orientation.as_roll_pitch_yaw_in_radians()
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            obj.position.x,
            obj.position.y,
            origin_z,
            roll,
            pitch,
            yaw,
            reference_frame=corpus,
        )

    def _seat_object_on_layer(
        self,
        obj: EGObject2D,
        body: Body,
        slab_top_z: float,
        corpus: KinematicStructureEntity,
    ) -> None:
        """
        Lower *body* so its mesh rests on the layer slab with a small contact
        overlap.

        Object meshes carry their own, mesh-specific origin offset, so seating
        them by their measured collision bottom -- rather than by a fixed
        origin height -- both makes them actually rest on the slab and gives the
        slight overlap that :func:`is_supported_by` needs to register support.
        Assumes the object is upright, so its body-frame vertical extent equals
        its corpus-frame one.

        :param obj: The object being seated, used to recompute the pose.
        :param body: The already-spawned body to lower.
        :param slab_top_z: Height of the layer slab's top face in the corpus
            frame.
        :param corpus: The shelf corpus body the pose is expressed relative to.
        """
        mesh_bottom = body.collision.combined_mesh.bounds[0][2]
        origin_z = slab_top_z - mesh_bottom - 0.005
        body.parent_connection.origin = self.object_local_pose(obj, origin_z, corpus)

    def spawn_in_world(
        self,
        world: World | None = None,
        parent: KinematicStructureEntity | None = None,
    ) -> SpawnedShelf:
        """
        Instantiate the shelf and its objects inside a :class:`World`, returning
        handles to the created layer annotations and object bodies.

        The handles let a caller validate and reposition individual objects in
        the spawned world without rebuilding it.

        :param world: Existing world to extend. A fresh world with a ``map``
            root body is created when omitted.
        :param parent: The parent entity the shelf's own
            :attr:`position`/ :attr:`orientation` are expressed relative to.
            Defaults to the world's root when omitted.
        :return: The spawned shelf, its world, and per-layer handles.
        """
        _world: World = world if world is not None else World()
        if world is None:
            root = Body(name=PrefixedName(name="map"))
            with _world.modify_world():
                _world.add_body(root)

        _parent = parent if parent is not None else _world.root

        footprint = self.corpus_footprint
        corpus_face = footprint.width
        corpus_depth = footprint.length
        corpus_height = footprint.height
        # Contents are stored in the shelf's content frame (see
        # CONTENT_FRAME_YAW_OFFSET_DEGREES), so the corpus and its slabs are
        # built in that same frame -- the offset must match extraction's.
        yaw_radians = math.radians(
            self.orientation.z + self.CONTENT_FRAME_YAW_OFFSET_DEGREES
        )

        corpus_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=self.position.x,
            y=self.position.y,
            z=corpus_height / 2,
            yaw=yaw_radians,
            reference_frame=_parent,
        )
        with _world.modify_world():
            corpus_annotation = Cabinet.create_with_new_body_in_world(
                name=PrefixedName(name="shelf_corpus"),
                world=_world,
                world_root_T_self=corpus_pose,
                scale=Scale(x=corpus_depth, y=corpus_face, z=corpus_height),
                wall_thickness=self._CORPUS_WALL_THICKNESS,
            )
        corpus_body = corpus_annotation.root
        # Make the whole shelf a movable unit: a room-level resolver repositions
        # it by setting the corpus origin, and its slabs and objects follow.
        _world.make_branch_movable(corpus_body)

        step = corpus_height / (len(self.layers) + 1)
        layer_z_heights = [step * (i + 1) for i in range(len(self.layers))]

        mesh_matcher = _MeshTypeMatcher(candidates=self.source_ids or [])

        spawned_layers: list[SpawnedShelfLayer] = []
        for i, (layer, z_height) in enumerate(zip(self.layers, layer_z_heights)):
            layer_scale = Scale(
                x=layer.scale.length, y=layer.scale.width, z=self._LAYER_SLAB_THICKNESS
            )
            layer_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=self.position.x,
                y=self.position.y,
                z=z_height,
                yaw=yaw_radians,
                reference_frame=_parent,
            )
            with _world.modify_world():
                layer_annotation = ShelfLayer.create_with_new_body_in_world(
                    name=PrefixedName(name=f"layer_{i}"),
                    world=_world,
                    world_root_T_self=layer_pose,
                    scale=layer_scale,
                )
            # Reparent the slab under the corpus so the whole shelf moves as one
            # unit when it is repositioned at the room level; the world pose is
            # preserved by the move.
            _world.move_branch(layer_annotation.root, corpus_body)

            # Slab top expressed in the corpus frame (corpus centre is at
            # z = corpus_height / 2 in the parent frame).
            slab_top_z = (z_height - corpus_height / 2) + self._LAYER_SLAB_THICKNESS / 2
            # Vertical room above this slab: up to the next slab's underside, or
            # the corpus interior ceiling for the topmost layer. Objects taller
            # than this would pierce the shelf above, which the resolver (it only
            # moves objects in the plane) can never repair -- so they are dropped
            # rather than placed.
            if i + 1 < len(self.layers):
                surface_above_z = (
                    layer_z_heights[i + 1] - corpus_height / 2
                ) - self._LAYER_SLAB_THICKNESS / 2
            else:
                surface_above_z = corpus_height / 2 - self._CORPUS_WALL_THICKNESS
            max_object_extents = EGScale(
                width=layer.scale.width,
                length=layer.scale.length,
                height=surface_above_z - slab_top_z - self._OBJECT_VERTICAL_MARGIN,
            )
            object_bodies: dict[int, Body] = {}
            for object_index, obj in enumerate(layer.objects):
                if not isinstance(obj.position.x, (int, float)):
                    continue
                if not self.source_ids:
                    continue
                candidate = mesh_matcher.random_match(
                    obj.object_type, max_extents=max_object_extents
                )
                # No mesh of this type is small enough for the layer; the object
                # is simply too big for the shelf, so leave it out.
                if candidate is None:
                    continue
                obj.source_id = candidate.source_id
                body = obj.create_in_world(
                    _world,
                    candidate.scene_dir,
                    parent=corpus_body,
                    world_pose=self.object_local_pose(obj, slab_top_z, corpus_body),
                )
                object_bodies[object_index] = body
                self._seat_object_on_layer(obj, body, slab_top_z, corpus_body)

            spawned_layers.append(
                SpawnedShelfLayer(
                    surface=layer_annotation,
                    object_bodies=object_bodies,
                )
            )

        return SpawnedShelf(
            world=_world,
            parent=_parent,
            layers=spawned_layers,
            corpus=corpus_body,
        )

    def create_in_world(
        self,
        world: World | None = None,
        parent: KinematicStructureEntity | None = None,
    ) -> World:
        """
        Instantiate the shelf and its objects inside a :class:`World`.

        Thin wrapper over :meth:`spawn_in_world` for callers that only need the
        resulting world and not the per-object body handles.

        :param world: Existing world to extend. A fresh world with a ``map``
            root body is created when omitted.
        :param parent: The parent entity the shelf's own
            :attr:`position`/ :attr:`orientation` are expressed relative to.
            Defaults to the world's root when omitted, so standalone callers
            are unaffected.
        :return: The world containing the shelf.
        """
        return self.spawn_in_world(world, parent).world


def wrap_angle_degrees(angle: float) -> float:
    """
    Wrap *angle* into the half-open interval (-180, 180] degrees.

    :param angle: Angle in degrees.
    :return: The equivalent angle in (-180, 180].
    """
    return ((angle + 180) % 360) - 180
