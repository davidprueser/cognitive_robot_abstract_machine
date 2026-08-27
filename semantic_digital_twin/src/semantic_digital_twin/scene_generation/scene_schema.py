from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, ClassVar, Optional, Self, assert_never

import numpy as np

from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json
from semantic_digital_twin.api import SpawnSpecification
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import MissingLiveSemanticAnnotationError
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
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose2D,
    Vector3,
)
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
        Create the object in the world by getting its geometry from the provided
        information.

        :param world: The world where the object is created.
        :param object_id_to_mesh_path: A mapping from an object's id to its mesh
            directory path.
        :param parent: The parent of the object in the world.
        :param kwargs: Additional keyword arguments.
        :return: The relevant created body
        """


@dataclass
class EGPoint2D(EGBase):
    x: float
    y: float

    def rotated_into_frame(self, frame_yaw_degrees: float) -> EGPoint2D:
        """
        Express this point, currently an offset along the world axes, in the axes of a
        frame rotated by *frame_yaw_degrees*.

        Needed wherever an object's offset from a rotated parent is stored for later re-
        use *inside* that parent: keeping the offset on the world axes makes it mean
        something different once the parent's own rotation is applied again.

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

    Inherits ``x`` (roll) and ``y`` (pitch) from :class:`EGPoint2D`; only ``z`` (yaw)
    varies for objects that sit upright without tilting.
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
    Generalized object categories that unify the tens of thousands of distinct, near-
    instance-specific ``object_type`` strings found in the raw sage10k dataset (for
    example ``"book1"``, ``"book_table2"`` and ``"bookchair8eba7fdc"`` all belong to the
    same real-world category of object).

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


def _scale_to_json(scale: Scale) -> dict[str, Any]:
    """
    Serialize a :class:`~semantic_digital_twin.world_description.geometry.Scale` to
    JSON.

    ``Scale`` is not a :class:`~krrood.adapters.json_serializer.SubclassJSONSerializer`
    -- it is a plain, three-float dataclass shared with the rest of the world
    description -- so it is serialized directly here rather than through :func:`to_json`.

    :param scale: The scale to serialize.
    :return: The JSON representation.
    """
    return {"x": scale.x, "y": scale.y, "z": scale.z}


def _scale_from_json(data: dict[str, Any]) -> Scale:
    """
    Deserialize a :class:`~semantic_digital_twin.world_description.geometry.Scale`
    from JSON written by :func:`_scale_to_json`.

    :param data: The JSON representation.
    :return: The deserialized scale.
    """
    return Scale(x=data["x"], y=data["y"], z=data["z"])


def _mesh_centered_on_footprint(
    ply_file_path: Path,
    texture_file_path: Path,
    body: KinematicStructureEntity,
) -> Mesh:
    """
    Load a sage10k PLY mesh with its origin re-centred onto its own footprint.

    A sage10k PLY's local origin is wherever the scan happened to put it -- a corner, an
    edge, the true centre -- with no guarantee it means anything about the object.
    Placing such a mesh at identity origin would seat the body's TF frame on that
    arbitrary point instead of the object, so the same recorded position visually
    offsets different meshes by different, mesh-specific amounts. Re-centring here makes
    the body's TF frame sit at the mesh's true horizontal centre and lowest point
    instead, matching what :attr:`EGObject.position_is_mesh_corrected` promises
    consumers.

    :param ply_file_path: Path to the PLY geometry file.
    :param texture_file_path: Path to the PLY's texture image.
    :param body: The body the mesh's origin is expressed relative to.
    :return: The mesh, with its origin re-centred onto its own footprint.
    """
    # sage10k meshes already carry their real-world size, so spawn at
    # identity scale to keep the mesh's true proportions instead of
    # stretching it to an independently sampled scale.
    mesh = Mesh.from_ply_file(
        ply_file_path=str(ply_file_path),
        texture_file_path=str(texture_file_path),
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=body),
        scale=Scale(1.0, 1.0, 1.0),
    )
    minimum_bound, maximum_bound = mesh.mesh.bounds
    footprint_center_x = (minimum_bound[0] + maximum_bound[0]) / 2
    footprint_center_y = (minimum_bound[1] + maximum_bound[1]) / 2
    mesh.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        -footprint_center_x,
        -footprint_center_y,
        -minimum_bound[2],
        reference_frame=body,
    )
    return mesh


# %%
@dataclass
class EGObject(EGWithID):
    room_id: str
    """
    The id of the room where the object is located.
    """

    place_id: str
    """
    The id of the object where the object is located/placed on/at, e.g. wall, floor,
    anchor, or the id of a piece of furniture it stands on.
    """

    object_type: ObjectType
    """
    The type of the object.
    """

    scale: Scale
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

    description: Optional[str] = None
    """
    Free-text description of the object as written in the source dataset.
    """

    place_guidance: Optional[str] = None
    """
    Free-text description of where the object is meant to be placed, as written in the
    source dataset.
    """

    position_is_mesh_corrected: bool = True
    """
    Whether :attr:`position` was corrected to the object's true mesh bounding-box
    centre.

    ``False`` means the mesh was unavailable when the object was processed and the
    source dataset's recorded position was kept unchanged, which is not guaranteed to be
    the mesh's centre. Consumers that need centred positions should filter on this.
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
            "dimensions": _scale_to_json(self.scale),
            "source_id": self.source_id,
            "description": self.description,
            "place_guidance": self.place_guidance,
            "position_is_mesh_corrected": self.position_is_mesh_corrected,
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
            scale=_scale_from_json(data["dimensions"]),
            source_id=data["source_id"],
            description=data.get("description"),
            place_guidance=data.get("place_guidance"),
            position_is_mesh_corrected=data.get("position_is_mesh_corrected", True),
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
        Instantiate this object in *world* by loading its PLY mesh from *mesh_path*.

        The mesh keeps its own native real-world size, since sage10k PLY assets already
        carry their real dimensions; collisions are checked against that real mesh, so
        stretching it to an independently sampled scale would both distort it and
        disagree with the geometry the layout is resolved against.

        Walls are attached with a fixed connection; every other object is attached with
        a movable 6-DoF connection whose pose lives in its degrees of freedom, so a
        resolver can reposition it in place via the ``origin`` setter.

        :param world: The world where the object is created.
        :param mesh_path: Directory containing the ``objects/`` sub- folder with PLY and
            texture files for this object.
        :param parent: The parent kinematic structure entity.
        :param world_pose: When given, the body is placed at this pose instead of the
            one built from :attr:`position`/ :attr:`orientation`, so a caller that
            already computed the pose can reuse it.
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
class EGObject2D(EGBase, SpawnSpecification[Body]):
    """
    An object on a shelf layer -- position is 2-D since z is determined by the layer.
    """

    object_type: ObjectType
    """
    The category of the object.
    """

    scale: Scale
    """
    Physical dimensions of the object.
    """

    pose: Pose2D
    """
    Pose of the object relative to the centre of the shelf layer's content frame.

    Always upright (no roll/pitch), so only ``x``, ``y`` and ``yaw`` vary. ``yaw`` is in
    radians, as :class:`~semantic_digital_twin.spatial_types.spatial_types.Pose2D`
    requires -- everywhere else in this schema reasons about yaw in degrees, so it is
    converted at the boundary where a value enters or leaves this field.
    """

    source_id: str
    """
    Identifier used to look up the PLY mesh file for this object in the dataset.
    """

    name: str | None = field(default=None, kw_only=True)
    """
    Optional explicit name for the spawned body.

    Falls back to :attr:`source_id` when unset, replacing the identifier previously
    carried on a now-removed ``id`` field.
    """

    annotation: Optional[Body] = field(default=None, compare=False)
    """
    The body spawned for this object in the world.

    ``None`` until the object is spawned by :meth:`spawn`, which sets it to the body it
    creates.
    """

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "type": self.object_type,
            "pose": to_json(self.pose),
            "dimensions": _scale_to_json(self.scale),
            "source_id": self.source_id,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            object_type=ObjectType._value2member_map_.get(
                data["type"], ObjectType.OTHER
            ),
            pose=Pose2D._from_json(data["pose"], **kwargs),
            scale=_scale_from_json(data["dimensions"]),
            source_id=data["source_id"],
        )

    def spawn(
        self,
        world: World,
        name: str | None = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        *,
        mesh_path: Path | None,
        x: float | None = None,
        y: float | None = None,
        z: float = 0.0,
    ) -> Body:
        """
        Instantiate this object in *world* at the given absolute pose.

        The mesh keeps its own native real-world size, since sage10k PLY assets already
        carry their real dimensions; stretching them to an independently sampled scale
        would distort them.

        :param world: The world the object is created in.
        :param name: Overrides :attr:`name` for the spawned body's naming; falls back
            to :attr:`source_id` when neither is set.
        :param parent: The parent kinematic structure entity. Defaults to the world's
            root when omitted.
        :param parent_T_self: When given, the body is placed at this pose and *x*, *y*,
            *z* are ignored, so a caller that already computed the pose can reuse it for
            both spawning and later repositioning.
        :param mesh_path: Directory containing the ``objects/`` sub-folder with PLY and
            texture files for this object.
        :param x: Absolute x in the parent frame (defaults to :attr:`pose`'s x).
        :param y: Absolute y in the parent frame (defaults to :attr:`pose`'s y).
        :param z: Absolute z in the parent frame.
        :raises ValueError: If *mesh_path* is ``None`` or does not exist.
        :return: The created :class:`Body`.
        """
        if mesh_path is None:
            raise ValueError(
                f"No mesh path resolved for object (source_id={self.source_id!r})."
            )
        if not mesh_path.exists():
            raise ValueError(f"Directory {mesh_path} does not exist.")
        ply_file = mesh_path / "objects" / f"{self.source_id}.ply"
        texture_file = mesh_path / "objects" / f"{self.source_id}_texture.png"

        _parent = parent if parent is not None else world.root

        body = Body()
        body.name = PrefixedName(
            name=str(body.id), prefix=name or self.name or self.source_id
        )

        if parent_T_self is not None:
            root_T_body = parent_T_self.copy_with_new_reference_frames(
                new_reference_frame=_parent, new_child_frame=body
            )
        else:
            root_T_body = HomogeneousTransformationMatrix.from_xyz_rpy(
                self.pose.x if x is None else x,
                self.pose.y if y is None else y,
                z,
                yaw=self.pose.yaw,
                reference_frame=_parent,
                child_frame=body,
            )

        mesh = _mesh_centered_on_footprint(ply_file, texture_file, body)

        geometry = ShapeCollection([mesh], reference_frame=body)
        body.visual = geometry
        body.collision = geometry

        with world.modify_world():
            root_C_body = Connection6DoF.create_with_dofs(
                world=world,
                parent=_parent,
                child=body,
            )
            world.add_body(body)
            world.add_connection(root_C_body)

        # Placing the pose in the connection's degrees of freedom rather than in
        # a fixed parent expression keeps the object movable: the ``.origin``
        # setter can later reposition it in place.
        body.parent_connection.origin = root_T_body

        type_annotation = NaturalLanguageWithTypeDescription(
            root=body, description=None, type_description=self.object_type
        )

        with world.modify_world():
            world.add_semantic_annotation(type_annotation)

        self.annotation = body
        return body


@dataclass
class EGShelfLayer(EGBase, SpawnSpecification[ShelfLayer]):
    """
    A shelf layer for environment generation.

    A layer's footprint is always its shelf's own width and length -- that is how
    layers are extracted and how they are spawned -- so it carries no dimensions of
    its own.

    It also carries where it sits vertically in its shelf. An object's own position is
    two-dimensional, since it simply rests on the slab, so without these the height at
    which a category tends to be kept -- books low, display pieces high -- is nowhere in
    the data.
    """

    _SLAB_THICKNESS: ClassVar[float] = 0.02
    """
    Thickness, in metres, of the spawned layer slab.
    """

    objects: list[EGObject2D]
    """
    Objects placed on this layer, with positions relative to the shelf centre.
    """

    theme_dominant_type: ObjectType
    """
    The object type that occurs most often among this layer's shelf's objects.

    Denormalized from the owning shelf because a fitted circuit passes only
    aggregation statistics from a parent to its parts: a theme known solely to
    the shelf would leave layer contents independent of it, and a book-dominant
    shelf would draw the same objects as a bottle-dominant one.
    """

    height_above_shelf_base: float = 0.0
    """
    Height of the slab above the base of its shelf, in metres.

    This is the reachable height a robot has to plan for. Zero for a layer that was not
    extracted from a real shelf.
    """

    relative_height: float = 0.0
    """
    Where the slab sits between its shelf's base (0) and top (1).

    Carried alongside :attr:`height_above_shelf_base` because it is the form
    that transfers: "books sit low" holds across shelves of different heights,
    while a given height in metres does not.
    """

    vertical_clearance: float = 0.0
    """
    Space above the slab, in metres, up to the next layer or the shelf's interior
    ceiling.

    This is what decides whether an object fits, so keeping it lets that be learned from
    real shelves instead of assumed from evenly spaced layers.
    """

    annotation: Optional[ShelfLayer] = field(default=None, compare=False)
    """
    The layer's supporting-surface annotation in the world.

    ``None`` until the layer is spawned by :meth:`spawn`, which sets it to the
    annotation it creates. Each spawned object's own body lives on its
    :attr:`EGObject2D.annotation`, not here.
    """

    name: str | None = field(default=None, kw_only=True)
    """
    Optional explicit name for the spawned layer annotation and its body.
    """

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "objects": to_json(self.objects),
            "theme_dominant_type": self.theme_dominant_type,
            "height_above_shelf_base": self.height_above_shelf_base,
            "relative_height": self.relative_height,
            "vertical_clearance": self.vertical_clearance,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            objects=[EGObject2D._from_json(o, **kwargs) for o in data["objects"]],
            theme_dominant_type=ObjectType(data["theme_dominant_type"]),
            height_above_shelf_base=data.get("height_above_shelf_base", 0.0),
            relative_height=data.get("relative_height", 0.0),
            vertical_clearance=data.get("vertical_clearance", 0.0),
        )

    def spawn(
        self,
        world: World,
        name: str | None = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        *,
        corpus: KinematicStructureEntity,
        shelf_scale: Scale,
    ) -> ShelfLayer:
        """
        Spawn this layer's slab and reparent it under the shelf ``corpus`` so the whole
        shelf moves as one unit when it is repositioned.

        Calculates the slab's free space eagerly, since it is needed for every
        placement query and survives a later
        :meth:`~semantic_digital_twin.world.World.merge_world` of the shelf into
        another world intact, even though the annotation object it is calculated on
        does not (see :meth:`EGShelf.refresh_layer_annotations`).

        :param world: The world the slab is added to.
        :param name: Overrides :attr:`name` for the spawned annotation and body.
        :param parent: The frame *parent_T_self* is expressed in, before the slab is
            reparented onto *corpus*. Defaults to the world's root when omitted.
        :param parent_T_self: The slab's pose in *parent*'s frame. Mandatory --
            :attr:`height_above_shelf_base` is where this layer's objects were
            *recorded*, not an evenly-spaced slab grid position (see
            :meth:`EGShelf._layer_heights`), so there is no default pose that is not
            wrong by construction; the caller (normally :meth:`EGShelf.spawn`) must
            compute the real slab height itself.
        :param corpus: The shelf corpus this layer belongs to; the slab is reparented
            under it once spawned.
        :param shelf_scale: The owning shelf's scale -- a layer carries no footprint of
            its own.
        :raises ValueError: If *parent_T_self* is omitted.
        :return: The spawned layer's supporting-surface annotation.
        """
        if parent_T_self is None:
            raise ValueError(
                "EGShelfLayer.spawn requires an explicit parent_T_self: "
                "height_above_shelf_base is a recorded position, not a spawnable slab "
                "height."
            )
        _parent = parent if parent is not None else world.root
        slab_scale = Scale(x=shelf_scale.x, y=shelf_scale.y, z=self._SLAB_THICKNESS)
        layer_annotation = ShelfLayer.get_annotation_specification(
            name or self.name or "layer",
            ShelfLayer.get_default_root_kinematic_structure_entity_specification(
                scale=slab_scale
            ),
        ).spawn(world, parent=_parent, parent_T_self=parent_T_self)
        # Reparent the slab under the corpus so the whole shelf moves as one
        # unit when it is repositioned at the room level; the world pose is
        # preserved by the move.
        world.move_branch(layer_annotation.root, corpus)
        self.annotation = layer_annotation
        with world.modify_world():
            layer_annotation.calculate_supporting_surface()
        return layer_annotation


@dataclass
class ObjectTypeAffinity(EGBase):
    """
    How often two object types were observed sharing a shelf layer, and how they were
    typically arranged relative to each other.

    Answers questions of the form "I am holding a book -- what else usually shares a
    shelf with books, and roughly where do they sit relative to one another" without re-
    scanning every stored layer at query time.
    """

    object_type_a: ObjectType
    """
    The first type of the pair, always the lexicographically smaller of the two so that
    each unordered pair is stored exactly once.
    """

    object_type_b: ObjectType
    """
    The second type of the pair, always the lexicographically larger of the two.
    """

    co_occurrence_count: int
    """
    Number of observed pairs of objects of these two types on a common layer.
    """

    mean_relative_offset: EGPoint2D
    """
    Mean of ``b.position - a.position`` over every observed pair, in the layer's
    own content frame.
    """

    @classmethod
    def canonical_pair(
        cls, first: ObjectType, second: ObjectType
    ) -> tuple[ObjectType, ObjectType]:
        """
        Order *first* and *second* so an unordered type pair always maps onto the same
        :attr:`object_type_a`/ :attr:`object_type_b` assignment.

        :param first: One type of the pair.
        :param second: The other type of the pair.
        :return: The pair in canonical order.
        """
        return (first, second) if first <= second else (second, first)

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "object_type_a": self.object_type_a,
            "object_type_b": self.object_type_b,
            "co_occurrence_count": self.co_occurrence_count,
            "mean_relative_offset": to_json(self.mean_relative_offset),
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            object_type_a=ObjectType(data["object_type_a"]),
            object_type_b=ObjectType(data["object_type_b"]),
            co_occurrence_count=data["co_occurrence_count"],
            mean_relative_offset=EGPoint2D._from_json(
                data["mean_relative_offset"], **kwargs
            ),
        )


@dataclass
class ObjectTypeHeightProfile(EGBase):
    """
    How high on a shelf objects of one type were typically found.

    Answers "I am holding a book -- which shelf do books belong on" directly, rather
    than leaving it to be recovered by walking every stored shelf.
    """

    object_type: ObjectType
    """
    The category this profile describes.
    """

    observation_count: int
    """
    Number of observed objects of this type the profile was built from.
    """

    mean_relative_height: float
    """
    Mean of the layers' :attr:`EGShelfLayer.relative_height`, so zero is the shelf's
    base and one its top.
    """

    mean_height_above_shelf_base: float
    """
    Mean of the layers' :attr:`EGShelfLayer.height_above_shelf_base`, in metres.
    """

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "object_type": self.object_type,
            "observation_count": self.observation_count,
            "mean_relative_height": self.mean_relative_height,
            "mean_height_above_shelf_base": self.mean_height_above_shelf_base,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            object_type=ObjectType(data["object_type"]),
            observation_count=data["observation_count"],
            mean_relative_height=data["mean_relative_height"],
            mean_height_above_shelf_base=data["mean_height_above_shelf_base"],
        )


@dataclass(frozen=True)
class MeshCandidate:
    """
    A mesh asset available for rendering a sampled object, together with the generalized
    object type it was captured from.
    """

    scene_dir: Path
    """
    Directory containing the ``objects/`` sub-folder with this mesh's PLY and texture
    files.
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
    whether it fits a target space.

    ``None`` when the size is unknown, in which case the candidate is treated as always
    fitting.
    """


@dataclass
class _MeshTypeMatcher:
    """
    Selects, from a pool of candidate meshes, a random one captured from an object of
    the same :class:`ObjectType`.

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

    candidates: list[MeshCandidate]
    """
    Pool of meshes to choose from.
    """

    def random_match(
        self,
        object_type: ObjectType,
        max_extents: Scale | None = None,
        target_extents: Scale | None = None,
    ) -> MeshCandidate | None:
        """
        Return a candidate whose :attr:`MeshCandidate.object_type` equals *object_type*,
        or ``None`` when the pool holds none.

        *max_extents* is an upper bound: candidates larger than it on any axis are
        ineligible, which is how shelf contents are kept from piercing the layer above.
        *target_extents* is a size to aim for: candidates further than
        :attr:`MAXIMUM_SIZE_RATIO` from it on any axis are ineligible, and the closest
        remaining one is returned rather than a random one.

        :param object_type: The category of the object a mesh is selected for.
        :param max_extents: Upper bound on the mesh's width/length/height, as ``(length,
            width, height)`` on ``(x, y, z)``.
        :param target_extents: Size the mesh should match as closely as possible, in the
            same axis convention as *max_extents*.
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
            if mismatch <= math.log(2.0)
        ]
        if not eligible:
            return None
        return min(eligible, key=lambda scored_candidate: scored_candidate[0])[1]

    @staticmethod
    def _size_mismatch(candidate: MeshCandidate, target_extents: Scale) -> float:
        """
        How far *candidate*'s real size is from *target_extents*, as the largest
        absolute log-ratio across the three axes.

        A log-ratio is used so that being twice too large and half too large count
        equally. Candidates of unknown size score as a perfect match, since there is
        nothing to judge them on and dropping them would thin an already sparse pool.

        :param candidate: The mesh candidate to score.
        :param target_extents: The size the mesh should match, as ``(length, width,
            height)`` on ``(x, y, z)``.
        :return: The mismatch, zero being an exact match.
        """
        native = candidate.native_extents
        if native is None:
            return 0.0
        targets = (target_extents.y, target_extents.x, target_extents.z)
        return max(
            abs(math.log(measured / target))
            for measured, target in zip(native, targets)
            if measured > 0 and target > 0
        )

    @staticmethod
    def _fits(candidate: MeshCandidate, max_extents: Scale) -> bool:
        """
        Whether *candidate*'s real-world size stays within *max_extents* on every axis.
        Candidates of unknown size are treated as fitting.

        :param candidate: The mesh candidate to test.
        :param max_extents: Per-axis upper bound, as ``(length, width, height)`` on
            ``(x, y, z)``.
        :return: ``True`` if the candidate fits or its size is unknown.
        """
        native = candidate.native_extents
        if native is None:
            return True
        native_width, native_length, native_height = native
        return (
            native_width <= max_extents.y
            and native_length <= max_extents.x
            and native_height <= max_extents.z
        )


@dataclass(frozen=True)
class ShelfLayerGeometry:
    """
    Where one layer's slab sits in its shelf, and how large an object it accepts.

    Computed from the shelf alone, so a caller that wants to place something on an
    already-spawned shelf reads the same heights the spawn was built from.
    """

    height_above_shelf_base: float
    """
    Height of the slab above the base of the shelf, in metres.
    """

    relative_height: float
    """
    Where the slab sits between the shelf's base (0) and its top (1).
    """

    slab_top_height: float
    """
    Height of the slab's top face, in the shelf corpus's own frame.
    """

    maximum_object_extents: Scale
    """
    Largest object the layer accepts, as ``(length, width, height)`` on ``(x, y, z)``.

    Its height is the room up to the surface above, which is infinite for a layer
    resting on the shelf's top.
    """


@dataclass
class EGShelf(EGBase, SpawnSpecification[Cabinet]):
    """
    A shelf and the horizontal layers its contents rest on.

    The shelf defines the frame its contents are expressed in and always sits at that
    frame's origin. Where a shelf happened to stand in the room it was extracted from
    says nothing about shelves, so carrying it would only add a near-unique coordinate
    per training row for a circuit to split on -- so, unlike :class:`EGObject2D`, a
    shelf carries no pose of its own; a caller positions it by choosing :meth:`spawn`'s
    ``parent`` and ``parent_T_self``.
    """

    CONTENT_FRAME_YAW_OFFSET_DEGREES: ClassVar[float] = 90.0
    """
    Yaw offset, in degrees, between a shelf's stored orientation and the frame its
    contents are expressed in.

    In the dataset a shelf's contents spread along its wide face, which lies
    along the shelf's local x-axis -- but the spawned :class:`Cabinet` corpus
    keeps its depth on x (its opening is fixed to -x) and its face on y. This
    offset rotates the content frame so the face spread lands on the corpus's
    wide (width) axis instead of overflowing its shallow depth. Extraction and
    :meth:`spawn` must apply the *same* offset so the two stay inverses.

    ..note:: The sign decides whether the shelf's open face points toward or
        away from the viewer; it is chosen by inspecting the render, not derived.
    """

    _TOP_SURFACE_RELATIVE_HEIGHT: ClassVar[float] = 1.0
    """
    Relative height at which a layer stops describing a shelf level and starts
    describing the shelf's top.

    Extraction groups whatever stands on a piece of furniture, including what sits on
    top of it, so a quarter of the recorded layers reach the shelf's own top rather than
    a slab inside it.
    """

    _MAXIMUM_TOP_PLACEMENT_HEIGHT: ClassVar[float] = 1.6
    """
    Tallest shelf, in metres, whose top is still used as a surface.

    Things are commonly left on a low cabinet and never on a shelf taller than
    about shoulder height, so above this the recorded top layer is treated as an
    ordinary interior level instead.

    ..note:: A judgement about reach rather than a measurement -- the observed
        shelves span only 0.90 m to 1.56 m and cannot separate the two cases.
    """

    _OBJECT_VERTICAL_MARGIN: ClassVar[float] = 0.01
    """
    Headroom, in metres, subtracted from a layer's measured clearance before it is
    offered to an object, so a fitted mesh does not sit flush against the surface
    above.
    """

    scale: Scale
    """
    Scale of the shelf, as ``(length, width, height)`` on ``(x, y, z)`` -- the spawned
    :class:`Cabinet` corpus keeps its depth on x and its face on y (see
    :attr:`CONTENT_FRAME_YAW_OFFSET_DEGREES`).
    """

    layers: list[EGShelfLayer]
    """
    The layers of the Shelf.
    """

    theme_dominant_type: ObjectType
    """
    The object type that occurs most often among this shelf's own objects, which its
    dimensions, layer count and contents are conditioned on.
    """

    source_ids: list[MeshCandidate] | None = field(default=None)
    """
    Pool of candidate meshes used when placing objects on shelf layers.
    """

    annotation: Optional[Cabinet] = field(default=None, compare=False)
    """
    The shelf corpus's annotation in the world. ``None`` until :meth:`spawn` is called,
    which sets it to the annotation it creates.
    """

    name: str | None = field(default=None, kw_only=True)
    """
    Optional explicit name for the spawned corpus annotation and its body.
    """

    @property
    def world(self) -> Optional[World]:
        """
        The world this shelf was spawned into, or ``None`` before :meth:`spawn` is
        called.
        """
        return None if self.annotation is None else self.annotation.root._world

    @property
    def parent(self) -> Optional[KinematicStructureEntity]:
        """
        The frame this shelf's objects' poses are expressed relative to, or ``None``
        before :meth:`spawn` is called.
        """
        return (
            None
            if self.annotation is None
            else self.annotation.root.parent_connection.parent
        )

    @property
    def corpus(self) -> Optional[Body]:
        """
        The shelf corpus's body, so a caller can check objects for collision against
        its walls in addition to each other. ``None`` before :meth:`spawn` is called.
        """
        return None if self.annotation is None else self.annotation.root

    @property
    def corpus_footprint(self) -> Scale:
        """
        The footprint the spawned corpus occupies, in the shelf's own frame.

        Padded by twice the corpus wall thickness so the carved-out interior is
        exactly the shelf's own footprint -- otherwise a wall intrudes into the
        region objects were trained to occupy, and an object placed near the
        training data's edge margin collides with it (most visible on small
        shelves, where that margin is thinner than the wall). A caller placing a
        shelf against a room wall has to reserve this, not the bare footprint, or
        the corpus reaches through by the pad.

        Taken from the shelf's own dimensions -- a layer carries none of its own.

        .. note::
            :attr:`CONTENT_FRAME_YAW_OFFSET_DEGREES` and the corpus's own
            depth-on-x convention cancel, so the span this footprint covers when
            turned by the content frame's yaw is the span the corpus really
            occupies.
        """
        corpus_wall_thickness = 0.03
        wall_margin = 2 * corpus_wall_thickness
        return Scale(
            x=self.scale.x + wall_margin,
            y=self.scale.y + wall_margin,
            z=self.scale.z,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "scale": _scale_to_json(self.scale),
            "layers": to_json(self.layers),
            "theme_dominant_type": self.theme_dominant_type,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            scale=_scale_from_json(data["scale"]),
            layers=[EGShelfLayer._from_json(l, **kwargs) for l in data["layers"]],
            theme_dominant_type=ObjectType(data["theme_dominant_type"]),
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
        the object's on-shelf offset (``pose.x``/``y`` map straight onto the
        corpus x/y axes, which span the layer's length/width) at height
        *origin_z*, with its own yaw. Used both when first seating an object and
        when moving it to a resampled pose, so the two placements can never drift
        apart -- and it stays correct after the whole shelf is repositioned.

        :param obj: The object whose pose is computed.
        :param origin_z: Height of the object body's origin in the corpus frame.
        :param corpus: The shelf corpus body the pose is expressed relative to.
        :return: The object body's pose in the corpus frame.
        """
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            obj.pose.x,
            obj.pose.y,
            origin_z,
            yaw=obj.pose.yaw,
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
        Lower *body* so its mesh rests on the layer slab with a small contact overlap.

        Object meshes carry their own, mesh-specific origin offset, so seating them by
        their measured collision bottom -- rather than by a fixed origin height -- both
        makes them actually rest on the slab and gives the slight overlap that
        :func:`is_supported_by` needs to register support. Assumes the object is
        upright, so its body-frame vertical extent equals its corpus-frame one.

        :param obj: The object being seated, used to recompute the pose.
        :param body: The already-spawned body to lower.
        :param slab_top_z: Height of the layer slab's top face in the corpus frame.
        :param corpus: The shelf corpus body the pose is expressed relative to.
        """
        mesh_bottom = body.collision.combined_mesh.bounds[0][2]
        origin_z = slab_top_z - mesh_bottom - 0.005
        body.parent_connection.origin = self.object_local_pose(obj, origin_z, corpus)

    def _rests_on_top(self, layer: EGShelfLayer) -> bool:
        """
        Whether *layer* describes objects standing on the shelf rather than in it.

        A shelf has one top, but layers are drawn independently and several can come
        back recorded there, so only the highest takes it; the rest are ordinary levels.
        Otherwise their slabs would land on each other with no room between them.

        :param layer: The layer to classify.
        :return: ``True`` when its objects belong on the shelf's top surface.
        """
        if self.scale.z > self._MAXIMUM_TOP_PLACEMENT_HEIGHT:
            return False
        highest = max(self.layers, key=lambda candidate: candidate.relative_height)
        return (
            layer is highest
            and layer.relative_height >= self._TOP_SURFACE_RELATIVE_HEIGHT
        )

    def _layer_heights(self, corpus_height: float) -> list[float]:
        """
        Height of each layer's slab in the parent frame, aligned to :attr:`layers`.

        Slabs are spaced evenly across the corpus. A layer's ``relative_height`` records
        where its objects were *found*, not where a slab is: a shelf level holding
        nothing leaves no layer behind, so a measured gap is the distance to the next
        *occupied* level rather than to the next shelf. Placing slabs at those heights
        would carry that bias into the geometry and cluster them, while real shelves are
        evenly divided. What the data does supply is how many levels a kind of shelf
        has, and that decides how many slabs there are.

        :param corpus_height: Interior height of the shelf corpus, in metres.
        :return: One height per layer, in the order of :attr:`layers`.
        """
        interior_layers = [
            layer for layer in self.layers if not self._rests_on_top(layer)
        ]
        step = corpus_height / (len(interior_layers) + 1)
        heights_bottom_up = iter(
            step * (index + 1) for index in range(len(interior_layers))
        )
        # ``layers`` is drawn from an exchangeable RSPN template and comes back in
        # no particular order, so the grid is built by walking it sorted by height
        # and then read back out in the caller's own order -- a slab must land at
        # the layer it was built for, not at whichever grid slot the sorted pass
        # happened to produce it in.
        height_by_layer = {
            id(layer): (
                corpus_height if self._rests_on_top(layer) else next(heights_bottom_up)
            )
            for layer in sorted(self.layers, key=lambda l: l.relative_height)
        }
        return [height_by_layer[id(layer)] for layer in self.layers]

    def _surface_above_height(
        self,
        layer: EGShelfLayer,
        own_height: float,
        layer_heights: list[float],
        corpus_height: float,
    ) -> float:
        """
        Height, in the corpus frame, of the surface a *layer*'s objects would pierce.

        :param layer: The layer whose ceiling is wanted.
        :param own_height: That layer's own slab height above the shelf base, passed in
            rather than looked up: layers compare equal whenever they hold equal
            objects, so searching for one by value finds the wrong height.
        :param layer_heights: Every layer's slab height above the shelf base.
        :param corpus_height: Interior height of the shelf corpus, in metres.
        :return: The next slab's underside, the corpus interior ceiling, or infinity
            for a layer resting on the shelf's top, which has open air above it.
        """
        if self._rests_on_top(layer):
            return math.inf
        heights_above = [height for height in layer_heights if height > own_height]
        corpus_wall_thickness = 0.03
        layer_slab_thickness = EGShelfLayer._SLAB_THICKNESS
        if not heights_above:
            return corpus_height / 2 - corpus_wall_thickness
        return (min(heights_above) - corpus_height / 2) - layer_slab_thickness / 2

    def layer_geometries(self) -> list[ShelfLayerGeometry]:
        """
        Where each layer's slab sits and how large an object it accepts, in the order
        of :attr:`layers`.

        An object taller than the room above its slab would pierce the surface above,
        which no in-plane repair can fix, so that room is what a layer accepts.

        :return: One geometry per layer.
        """
        corpus_height = self.corpus_footprint.z
        layer_heights = self._layer_heights(corpus_height)
        geometries = []
        for layer, height in zip(self.layers, layer_heights):
            slab_top_height = (
                height - corpus_height / 2
            ) + EGShelfLayer._SLAB_THICKNESS / 2
            surface_above_height = self._surface_above_height(
                layer, height, layer_heights, corpus_height
            )
            geometries.append(
                ShelfLayerGeometry(
                    height_above_shelf_base=height,
                    relative_height=height / corpus_height,
                    slab_top_height=slab_top_height,
                    maximum_object_extents=Scale(
                        x=self.scale.x,
                        y=self.scale.y,
                        z=surface_above_height
                        - slab_top_height
                        - self._OBJECT_VERTICAL_MARGIN,
                    ),
                )
            )
        return geometries

    def spawn(
        self,
        world: World | None = None,
        name: str | None = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> Cabinet:
        """
        Instantiate the shelf and its objects inside a :class:`World`.

        Mutates ``self`` in place: :attr:`annotation` is set to the created corpus (so
        :attr:`world`, :attr:`parent` and :attr:`corpus` read from it), each
        :class:`EGShelfLayer`'s own :attr:`~EGShelfLayer.annotation` is set to its slab,
        and each spawned :class:`EGObject2D`'s :attr:`~EGObject2D.annotation` is set to
        its body -- so a caller keeps using this same shelf afterwards instead of a
        separate handle.

        :param world: Existing world to extend. A fresh world with a ``map`` root body
            is created when omitted.
        :param name: Overrides :attr:`name` for the spawned corpus annotation and body.
        :param parent: The parent entity the shelf is placed under. Defaults to the
            world's root when omitted.
        :param parent_T_self: Overrides the shelf's default pose (identity, offset only
            by :attr:`CONTENT_FRAME_YAW_OFFSET_DEGREES` and half its height).
        :return: The spawned corpus annotation.
        """
        _world: World = world if world is not None else World()
        if world is None:
            root = Body(name=PrefixedName(name="map"))
            with _world.modify_world():
                _world.add_body(root)

        _parent = parent if parent is not None else _world.root

        footprint = self.corpus_footprint
        # Contents are stored in the shelf's content frame (see
        # CONTENT_FRAME_YAW_OFFSET_DEGREES), so the corpus and its slabs are
        # built in that same frame -- the offset must match extraction's.
        yaw_radians = math.radians(self.CONTENT_FRAME_YAW_OFFSET_DEGREES)

        default_corpus_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.0,
            y=0.0,
            z=footprint.z / 2,
            yaw=yaw_radians,
            reference_frame=_parent,
        )
        corpus_annotation = Cabinet.get_annotation_specification(
            name or self.name or "shelf_corpus",
            Cabinet.get_default_root_kinematic_structure_entity_specification(
                scale=footprint,
                wall_thickness=0.03,
            ),
        ).spawn(
            _world, parent=_parent, parent_T_self=(parent_T_self or default_corpus_pose)
        )
        corpus_body = corpus_annotation.root
        # Make the whole shelf a movable unit: a room-level resolver repositions
        # it by setting the corpus origin, and its slabs and objects follow.
        _world.make_branch_movable(corpus_body)

        layer_geometries = self.layer_geometries()

        mesh_matcher = _MeshTypeMatcher(candidates=self.source_ids or [])

        # Every slab is created before any object is spawned onto any of them: a
        # later pass measuring a layer's own clearance against its neighbours'
        # real geometry needs every slab already standing, not just the ones
        # before it in layer order.
        for index, (layer, geometry) in enumerate(zip(self.layers, layer_geometries)):
            layer_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.0,
                y=0.0,
                z=geometry.height_above_shelf_base,
                yaw=yaw_radians,
                reference_frame=_parent,
            )
            layer.spawn(
                _world,
                name=f"layer_{index}",
                parent=_parent,
                parent_T_self=layer_pose,
                corpus=corpus_body,
                shelf_scale=self.scale,
            )

        for layer, geometry in zip(self.layers, layer_geometries):
            slab_top_z = geometry.slab_top_height
            # An object taller than the room above its slab would pierce the shelf
            # above, which the resolver (it only moves objects in the plane) can
            # never repair -- so they are dropped rather than placed.
            max_object_extents = geometry.maximum_object_extents
            for obj in layer.objects:
                if not obj.pose.x.is_constant():
                    continue
                candidate = (
                    mesh_matcher.random_match(
                        obj.object_type,
                        max_extents=max_object_extents,
                        target_extents=obj.scale,
                    )
                    if self.source_ids
                    else None
                )
                # No mesh of this type is small enough for the layer, or none is
                # cached at all. The object is either too big for the shelf or
                # simply unrenderable, so it is left out unless a stand-in was
                # asked for.
                if candidate is None:
                    continue
                obj.source_id = candidate.source_id
                body = obj.spawn(
                    _world,
                    parent=corpus_body,
                    parent_T_self=self.object_local_pose(obj, slab_top_z, corpus_body),
                    mesh_path=candidate.scene_dir,
                )
                self._seat_object_on_layer(obj, body, slab_top_z, corpus_body)

        self.annotation = corpus_annotation
        return corpus_annotation

    def create_in_world(
        self,
        world: World | None = None,
        parent: KinematicStructureEntity | None = None,
    ) -> World:
        """
        Instantiate the shelf and its objects inside a :class:`World`.

        Thin wrapper over :meth:`spawn` for callers that only need the resulting world
        and not the spawned corpus annotation.

        :param world: Existing world to extend. A fresh world with a ``map`` root body
            is created when omitted.
        :param parent: The parent entity the shelf is placed under. Defaults to the
            world's root when omitted, so standalone callers are unaffected.
        :return: The world containing the shelf.
        """
        self.spawn(world, parent=parent)
        return self.world

    def refresh_layer_annotations(self) -> None:
        """
        Re-resolve every layer's :class:`ShelfLayer` annotation from :attr:`world`.

        Merging this shelf's world into another one
        (:meth:`~semantic_digital_twin.world.World.merge_world`) replaces every
        semantic annotation on the merged bodies with a fresh instance, though the
        body itself survives unchanged -- so :attr:`EGShelfLayer.annotation` keeps
        pointing at an annotation with no world of its own once that happens. Call this
        once, right after merging this shelf's world into its final one -- the same
        point at which a caller of :class:`~semantic_digital_twin.
        semantic_annotations.semantic_annotations.Floor` or :class:`~semantic_digital_twin.
        semantic_annotations.semantic_annotations.Table` calls :meth:`~semantic_digital_twin.
        semantic_annotations.mixins.HasSupportingSurface.calculate_supporting_surface`
        on its own annotation.

        :raises MissingLiveSemanticAnnotationError: If some layer's body carries no
            live :class:`ShelfLayer` annotation in :attr:`world`.
        """
        for layer in self.layers:
            body = layer.annotation.root
            live_annotation = next(
                (
                    annotation
                    for annotation in self.world.get_semantic_annotations_by_type(
                        ShelfLayer
                    )
                    if annotation.root is body
                ),
                None,
            )
            if live_annotation is None:
                raise MissingLiveSemanticAnnotationError(
                    semantic_annotation_class=ShelfLayer, body_name=body.name
                )
            layer.annotation = live_annotation


def wrap_angle_degrees(angle: float) -> float:
    """
    Wrap *angle* into the half-open interval (-180, 180] degrees.

    :param angle: Angle in degrees.
    :return: The equivalent angle in (-180, 180].
    """
    return ((angle + 180) % 360) - 180
