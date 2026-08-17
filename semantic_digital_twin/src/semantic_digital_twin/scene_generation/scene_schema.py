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


class ShelfType(StrEnum):
    """
    Kinds of storage furniture whose layer structure and contents are modelled
    separately.

    The members are the categories the raw dataset supports well enough to
    condition on: each is backed by thousands of observed instances, and they
    differ in what the model is meant to capture -- bookcases carry several
    tightly spaced layers, cabinets and sideboards usually only one.

    .. note::
        Storage furniture the dataset describes too thinly or too ambiguously
        (dressers, wardrobes, display cases) is deliberately absent rather than
        folded into a neighbouring member. Conditioning on a member with almost
        no mass silently falls back to an unconditioned draw, which would answer
        a request for a type nothing was learned about with an arbitrary shelf.
    """

    OPEN_SHELF = "open_shelf"
    BOOKCASE = "bookcase"
    CABINET = "cabinet"
    SIDEBOARD = "sideboard"


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
            "dimensions": to_json(self.scale),
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
            scale=EGScale._from_json(data["dimensions"], **kwargs),
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
class EGObject2D(EGWithID):
    """
    An object on a shelf layer — position is 2-D since z is determined by the layer.
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
    2-D position relative to the centre of the shelf the object stands in.
    """

    orientation: EGRotation
    """
    Orientation of the object in Euler angles (degrees).
    """

    source_id: str
    """
    Identifier used to look up the PLY mesh file for this object in the dataset.
    """

    shelf_type: ShelfType
    """
    Kind of shelf this object was found in.

    Denormalized, like :attr:`EGShelfLayer.shelf_type`, because only aggregation
    statistics reach a part from its parent. Present on the layer alone it would shape
    the layer's own dimensions but leave which objects are drawn onto it independent of
    the kind of shelf -- a bookcase would fill with a cabinet's crockery. Required
    rather than defaulted, so that an extraction path which forgets it fails outright
    instead of quietly labelling its objects as belonging to some other kind of shelf.
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
            "shelf_type": self.shelf_type,
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
            shelf_type=ShelfType(data["shelf_type"]),
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

        The mesh keeps its own native real-world size, since sage10k PLY assets already
        carry their real dimensions; stretching them to an independently sampled scale
        would distort them.

        :param world: The world where the object is created.
        :param mesh_path: Directory containing the ``objects/`` sub- folder with PLY and
            texture files for this object.
        :param parent: The parent kinematic structure entity.
        :param x: Absolute x in world coordinates (defaults to ``self.position.x``).
        :param y: Absolute y in world coordinates (defaults to ``self.position.y``).
        :param z: Absolute z in world coordinates.
        :param world_pose: When given, the body is placed at this pose and *x*, *y*, *z*
            are ignored, so a caller that already computed the pose can reuse it for
            both spawning and later repositioning.
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

        mesh = _mesh_centered_on_footprint(ply_file, texture_file, body)

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

    Carries its own physical dimensions so the RSPN can learn width and length alongside
    object placement, rather than inheriting a fixed size from the parent shelf.

    It also carries where it sits vertically in its shelf. An object's own position is
    two-dimensional, since it simply rests on the slab, so without these the height at
    which a category tends to be kept -- books low, display pieces high -- is nowhere in
    the data.
    """

    SLAB_THICKNESS: ClassVar[float] = 0.02
    """
    Thickness, in metres, of a layer slab.

    One value shared by extraction, sampling and spawning: a layer recorded at one
    thickness and spawned at another would seat its objects at the wrong height.
    """

    scale: EGScale
    """
    Physical dimensions of the layer slab (width × length × height).

    Every layer of a shelf carries the shelf's own footprint -- that is how they are
    extracted and how they are drawn. It is kept per layer because object placement is
    learned relative to it and collision repair conditions on it.
    """

    objects: list[EGObject2D]
    """
    Objects placed on this layer, with positions relative to the shelf centre.
    """

    shelf_type: ShelfType
    """
    Kind of shelf this layer belongs to.

    Denormalized from the owning shelf because a fitted circuit passes only
    aggregation statistics from a parent to its parts: a type known solely to
    the shelf would leave layer contents independent of it, and a bookcase would
    draw the same objects as a cabinet.
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

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "scale": to_json(self.scale),
            "objects": to_json(self.objects),
            "shelf_type": self.shelf_type,
            "height_above_shelf_base": self.height_above_shelf_base,
            "relative_height": self.relative_height,
            "vertical_clearance": self.vertical_clearance,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            scale=EGScale._from_json(data["scale"], **kwargs),
            objects=[EGObject2D._from_json(o, **kwargs) for o in data["objects"]],
            shelf_type=ShelfType(data["shelf_type"]),
            height_above_shelf_base=data.get("height_above_shelf_base", 0.0),
            relative_height=data.get("relative_height", 0.0),
            vertical_clearance=data.get("vertical_clearance", 0.0),
        )


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
    fitting. A tuple (not an :class:`EGScale`) keeps :class:`MeshCandidate` hashable.
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

    MAXIMUM_SIZE_RATIO: ClassVar[float] = 2.0
    """
    How far a candidate's real size may differ from a requested target size, as a factor
    on each axis, before it is rejected.

    A mesh of the right category but the wrong size still looks wrong -- a sampled 0.45
    m stool spawning as a 1.2 m armchair -- so category alone is not enough once the
    circuit has sampled a size to aim for.
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
        Return a candidate whose :attr:`MeshCandidate.object_type` equals *object_type*,
        or ``None`` when the pool holds none.

        *max_extents* is an upper bound: candidates larger than it on any axis are
        ineligible, which is how shelf contents are kept from piercing the layer above.
        *target_extents* is a size to aim for: candidates further than
        :attr:`MAXIMUM_SIZE_RATIO` from it on any axis are ineligible, and the closest
        remaining one is returned rather than a random one.

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

        A log-ratio is used so that being twice too large and half too large count
        equally. Candidates of unknown size score as a perfect match, since there is
        nothing to judge them on and dropping them would thin an already sparse pool.

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
        Whether *candidate*'s real-world size stays within *max_extents* on every axis.
        Candidates of unknown size are treated as fitting.

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
class SpawnedShelf:
    """
    A shelf instantiated in a :class:`World`, with handles for in-world validation and
    repositioning of its objects.
    """

    world: World
    """
    The world the shelf was spawned into.
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
    The shelf corpus's body, so a caller can check objects for collision against its
    walls in addition to each other.
    """

    placeholder_count: int = 0
    """
    Objects standing in as plain boxes because no mesh of their type was cached.

    Reported so a render can be read honestly: a shelf that looks sparse because
    the mesh library is incomplete is a different thing from one the model drew
    that way.
    """

@dataclass
class EGShelf(EGBase):
    """
    A shelf and the horizontal layers its contents rest on.

    The shelf defines the frame its contents are expressed in and always sits at that
    frame's origin. Where a shelf happened to stand in the room it was extracted from
    says nothing about shelves, so carrying it would only add a near-unique coordinate
    per training row for a circuit to split on.
    """

    _CORPUS_WALL_THICKNESS: ClassVar[float] = 0.03
    """
    Thickness of the spawned :class:`Cabinet` corpus's walls.

    The corpus is sized larger than the layers' own footprint by this amount (see
    :meth:`spawn_in_world`), so a wall carved out of that footprint never intrudes into
    the region objects were trained to occupy.
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
    :meth:`spawn_in_world` must apply the *same* offset so the two stay inverses.

    ..note:: The sign decides whether the shelf's open face points toward or
        away from the viewer; it is chosen by inspecting the render, not derived.
    """

    _LAYER_SLAB_THICKNESS: ClassVar[float] = EGShelfLayer.SLAB_THICKNESS
    """
    Thickness, in metres, of each spawned layer slab.
    """

    _OBJECT_VERTICAL_MARGIN: ClassVar[float] = 0.01
    """
    Slack, in metres, kept between the tallest object a layer accepts and the surface
    above it, so a fitting object never grazes the next slab or ceiling.
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

    scale: EGScale
    """
    Scale of the Shelf.
    """

    layers: list[EGShelfLayer]
    """
    The layers of the Shelf.
    """

    shelf_type: ShelfType
    """
    Kind of shelf this is, which its dimensions, layer count and contents are
    conditioned on.
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
        exactly the shelf's own footprint -- otherwise a wall intrudes into the
        region objects were trained to occupy, and an object placed near the
        training data's edge margin collides with it (most visible on small
        shelves, where that margin is thinner than the wall). A caller placing a
        shelf against a room wall has to reserve this, not the bare footprint, or
        the corpus reaches through by the pad.

        Taken from the shelf's own dimensions rather than from its layers': the
        layers are drawn independently and disagree with each other, so sizing the
        corpus from them discards the shelf's learned proportions and makes every
        kind of shelf spawn the same box.

        .. note::
            :attr:`CONTENT_FRAME_YAW_OFFSET_DEGREES` and the corpus's own
            depth-on-x convention cancel, so the span this footprint covers when
            turned by :attr:`orientation` is the span the corpus really
            occupies.
        """
        wall_margin = 2 * self._CORPUS_WALL_THICKNESS
        return EGScale(
            width=self.scale.width + wall_margin,
            length=self.scale.length + wall_margin,
            height=self.scale.height,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "scale": to_json(self.scale),
            "layers": to_json(self.layers),
            "shelf_type": self.shelf_type,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            scale=EGScale._from_json(data["scale"], **kwargs),
            layers=[EGShelfLayer._from_json(l, **kwargs) for l in data["layers"]],
            shelf_type=ShelfType(data["shelf_type"]),
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
        if self.scale.height > self._MAXIMUM_TOP_PLACEMENT_HEIGHT:
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
        return [
            corpus_height if self._rests_on_top(layer) else next(heights_bottom_up)
            for layer in sorted(self.layers, key=lambda l: l.relative_height)
        ]

    def _spawn_placeholder(
        self,
        obj: EGObject2D,
        world: World,
        corpus: KinematicStructureEntity,
        slab_top_z: float,
    ) -> Body:
        """
        Stand a plain box where an object should be, at the size it was drawn.

        Used only while inspecting a render: it shows what the model placed when
        no mesh of that type exists to show it with.

        :param obj: The object that found no mesh.
        :param world: The world to spawn into.
        :param corpus: The shelf corpus the box is parented to.
        :param slab_top_z: Height of the supporting slab in the corpus frame.
        :return: The spawned placeholder body.
        """
        with world.modify_world():
            placeholder = Table.create_with_new_body_in_world(
                name=PrefixedName(name=f"placeholder_{obj.id}"),
                world=world,
                world_root_T_self=self.object_local_pose(
                    obj, slab_top_z + obj.scale.height / 2, corpus
                ),
                scale=Scale(x=obj.scale.length, y=obj.scale.width, z=obj.scale.height),
            )
        world.move_branch(placeholder.root, corpus)
        # Attached movably, like a real object: a placeholder sits in the object
        # list and collision repair repositions it the same way, which a rigid
        # attachment refuses outright.
        world.make_branch_movable(placeholder.root)
        return placeholder.root

    def spawn_in_world(
        self,
        world: World | None = None,
        parent: KinematicStructureEntity | None = None,
        placeholders_for_missing_meshes: bool = False,
    ) -> SpawnedShelf:
        """
        Instantiate the shelf and its objects inside a :class:`World`, returning handles
        to the created layer annotations and object bodies.

        The handles let a caller validate and reposition individual objects in the
        spawned world without rebuilding it.

        :param world: Existing world to extend. A fresh world with a ``map`` root body
            is created when omitted.
        :param parent: The parent entity the shelf is placed under. Defaults to the
            world's root when omitted.
        :param placeholders_for_missing_meshes: Stand a plain box in for any object
            whose type has no cached mesh, instead of leaving it out. For inspecting a
            render; off so a generation run never gains bodies that stand for nothing.
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
        yaw_radians = math.radians(self.CONTENT_FRAME_YAW_OFFSET_DEGREES)

        corpus_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.0,
            y=0.0,
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

        layer_z_heights = self._layer_heights(corpus_height)

        mesh_matcher = _MeshTypeMatcher(candidates=self.source_ids or [])

        spawned_layers: list[SpawnedShelfLayer] = []
        placeholder_count = 0
        for i, (layer, z_height) in enumerate(zip(self.layers, layer_z_heights)):
            # Every slab spans the shelf's own footprint rather than the layer's.
            # Layers are drawn independently and their footprints disagree, so a
            # slab at its own size floats clear of the corpus walls -- and its
            # objects, whose positions were drawn for a surface of the shelf's
            # width, would be placed on a differently sized one.
            layer_scale = Scale(
                x=self.scale.length, y=self.scale.width, z=self._LAYER_SLAB_THICKNESS
            )
            layer_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.0,
                y=0.0,
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
            # rather than placed. A layer resting on the shelf's top has open air
            # above it instead, and measuring it against the interior ceiling --
            # which lies below the top surface -- would reject every object.
            heights_above = [height for height in layer_z_heights if height > z_height]
            if self._rests_on_top(layer):
                surface_above_z = math.inf
            elif heights_above:
                surface_above_z = (
                    min(heights_above) - corpus_height / 2
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
                    if placeholders_for_missing_meshes:
                        object_bodies[object_index] = self._spawn_placeholder(
                            obj, _world, corpus_body, slab_top_z
                        )
                        placeholder_count += 1
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
            placeholder_count=placeholder_count,
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

        :param world: Existing world to extend. A fresh world with a ``map`` root body
            is created when omitted.
        :param parent: The parent entity the shelf's own :attr:`position`/
            :attr:`orientation` are expressed relative to. Defaults to the world's root
            when omitted, so standalone callers are unaffected.
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
