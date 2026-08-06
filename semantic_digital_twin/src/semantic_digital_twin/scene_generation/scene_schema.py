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


class RoomType(StrEnum):
    """
    Generalized room categories that unify the 227 distinct, inconsistently
    spelled ``room_type`` strings found in the raw sage10k dataset (for example
    ``"grocery store"``, ``"grocery_store"`` and ``"grocery store floor"`` all
    name the same real-world category of room).

    Room type is the strongest available predictor of which pieces a room holds
    and where they stand, so generation fits one circuit per category rather
    than pooling patient rooms, warehouses and kitchens into a single
    distribution.

    .. note::
        Assigned by
        :class:`~semantic_digital_twin.scene_generation.room_type_classifier.RoomTypeClassifier`,
        which is a best-effort heuristic rather than a guaranteed-correct
        classification.
    """

    BAKERY = "bakery"
    BAR = "bar"
    BATHROOM = "bathroom"
    BEDROOM = "bedroom"
    CASINO = "casino"
    CLASSROOM = "classroom"
    CLOSET = "closet"
    CLOTHING_STORE = "clothing_store"
    COMPUTER_LAB = "computer_lab"
    CONFERENCE_ROOM = "conference_room"
    CORRIDOR = "corridor"
    DINING_ROOM = "dining_room"
    EXAMINATION_ROOM = "examination_room"
    GAME_ROOM = "game_room"
    GARAGE = "garage"
    GREENHOUSE = "greenhouse"
    GROCERY_STORE = "grocery_store"
    GYM = "gym"
    HAIR_SALON = "hair_salon"
    KITCHEN = "kitchen"
    LAUNDRY_ROOM = "laundry_room"
    LIBRARY = "library"
    LIVING_ROOM = "living_room"
    LOBBY = "lobby"
    LOCKER_ROOM = "locker_room"
    MEDITATION_ROOM = "meditation_room"
    MUSEUM = "museum"
    NURSERY = "nursery"
    OFFICE = "office"
    OPERATING_ROOM = "operating_room"
    OTHER = "other"
    PANTRY = "pantry"
    PATIENT_ROOM = "patient_room"
    PRISON_CELL = "prison_cell"
    RESTAURANT = "restaurant"
    STORE = "store"
    STUDIO = "studio"
    WAREHOUSE = "warehouse"
    WINE_CELLAR = "wine_cellar"
    WORKSHOP = "workshop"


class PlaceId(StrEnum):
    """
    The reserved :attr:`EGObject.place_id` values that name a room's structure
    rather than another object.

    Any other ``place_id`` is the id of the piece of furniture an object rests
    on, so ``place_id == PlaceId.FLOOR`` is what distinguishes a real piece of
    furniture from a small item that merely carries a furniture word in its raw
    name (e.g. a ``"tablecloth"`` lying on a table, which the
    :class:`~semantic_digital_twin.scene_generation.object_type_classifier.ObjectTypeClassifier`
    generalizes to :attr:`ObjectType.TABLE`).
    """

    FLOOR = "floor"
    WALL = "wall"


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
    floor, table. The room-structure values are enumerated by :class:`PlaceId`.
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
class EGWall(EGWithID):
    start_point: EGPoint2D
    """
    The start point of the wall.
    """

    end_point: EGPoint2D
    """
    The end point of the wall.
    """

    height: float
    """
    The height of the wall.
    """

    thickness: float
    """
    The thickness of the wall.
    """

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "id": self.id,
            "start_point": to_json(self.start_point),
            "end_point": to_json(self.end_point),
            "height": self.height,
            "thickness": self.thickness,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs):
        return cls(
            id=data["id"],
            start_point=EGPoint2D._from_json(data["start_point"], **kwargs),
            end_point=EGPoint2D._from_json(data["end_point"], **kwargs),
            height=data["height"],
            thickness=data["thickness"],
        )

    @property
    def wall_length_and_yaw(self) -> tuple[float, float]:
        """
        :return: The length of the wall and the yaw that can be used for creating it with
        `Wall.create_with_new_body_in_world`.
        """
        # the wall length is given by x
        if self.start_point.x != self.end_point.x:
            wall_length = self.end_point.x - self.start_point.x
            yaw = math.pi / 2
        # the wall length is given by y
        elif self.start_point.y != self.end_point.y:
            wall_length = self.end_point.y - self.start_point.y
            yaw = 0
        else:
            assert_never(self)
        return wall_length, yaw

    def create_in_world(self, world: World, parent: Body, **kwargs) -> Wall:
        wall_name = PrefixedName(name=self.id)

        wall_length, yaw = self.wall_length_and_yaw

        wall_scale = Scale(x=self.thickness, y=wall_length, z=self.height)

        center_x = (self.end_point.x + self.start_point.x) / 2
        center_y = (self.end_point.y + self.start_point.y) / 2

        parent_T_wall = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=center_x,
            y=center_y,
            z=0.0,
            yaw=yaw,
            reference_frame=parent,
        )

        with world.modify_world():
            annotation = Wall.create_with_new_body_in_world(
                name=wall_name,
                scale=wall_scale,
                world=world,
                world_root_T_self=parent_T_wall,
            )

        body = annotation.root

        wall_mesh = body.collision.combined_mesh

        wall_mesh = Mesh.project_texture_coordinates(
            mesh=wall_mesh,
            projection_axis=np.array([1, 0, 0]),
            scale=np.array([self.thickness, wall_length, self.height]),
        )

        wall_length, _ = self.wall_length_and_yaw

        geometry_with_texture = ShapeCollection(
            [
                Mesh.from_trimesh(
                    origin=HomogeneousTransformationMatrix(reference_frame=body),
                    mesh=wall_mesh,
                )
            ],
            reference_frame=body,
        )
        body.collision = geometry_with_texture
        body.visual = geometry_with_texture

        return annotation


@dataclass
class EGDoor(EGWithID):
    """
    A door of a wall in Sage10k.
    """

    wall_id: str
    """
    Id of the wall where the door should be created on.
    """

    position_on_wall: float
    """
    Position on wall w.

    r. t. its starting point as percentage of the wall length.
    """

    width: float
    """
    Width of the door in meters.
    """

    height: float
    """
    Height of the door in meters.
    """

    opens_inward: bool
    """
    Rather it opens to the inside of the room or the outside.
    """

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "id": self.id,
            "wall_id": self.wall_id,
            "position_on_wall": self.position_on_wall,
            "width": self.width,
            "height": self.height,
            "opens_inward": self.opens_inward,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            id=data["id"],
            wall_id=data["wall_id"],
            position_on_wall=data["position_on_wall"],
            width=data["width"],
            height=data["height"],
            opens_inward=data["opens_inward"],
        )

    def create_in_world(
        self,
        world: World,
        parent: KinematicStructureEntity,
        **kwargs,
    ) -> Door:
        """
        The parent must always be the wall body.

        :param wall: The sage 10k wall that is referenced by
            `self.wall_id`.
        :param wall_annotation: The wall annotation created in `world`
            before this call.
        """
        name = PrefixedName(name=self.id, prefix=kwargs["wall"].id)

        scale = Scale(x=kwargs["wall"].thickness, y=self.width, z=self.height)

        wall_length, _ = kwargs["wall"].wall_length_and_yaw

        parent_T_body = HomogeneousTransformationMatrix.from_xyz_rpy(
            y=-wall_length / 2 + (self.position_on_wall * wall_length),
            z=self.height / 2,
            reference_frame=parent,
        )
        world_root_T_self = world.transform(parent_T_body, world.root)

        with world.modify_world():
            annotation = DoorWithType.create_with_new_body_in_world(
                name=name,
                scale=scale,
                world=world,
                world_root_T_self=world_root_T_self,
            )

        body = annotation.root
        door_mesh = body.collision.combined_mesh

        door_mesh = Mesh.project_texture_coordinates(
            mesh=door_mesh,
            projection_axis=np.array([1, 0, 0]),
            scale=np.array([kwargs["wall"].thickness, self.width, self.height]),
        )

        geometry_with_texture = ShapeCollection(
            [
                Mesh.from_trimesh(
                    origin=HomogeneousTransformationMatrix(reference_frame=body),
                    mesh=door_mesh,
                )
            ],
            reference_frame=body,
        )
        body.collision = geometry_with_texture
        body.visual = geometry_with_texture

        with world.modify_world():
            kwargs["wall_annotation"].add(annotation.entry_way, field_name="apertures")

        self._create_handle_in_world(world, annotation)
        self._create_hinge_in_world(world, annotation)
        return annotation

    def _create_handle_in_world(self, world: World, door: Door) -> Handle:
        """
        Create the handle of the door.

        :param world: The world where the handle is created.
        :param door: The door to create the handle for.
        :return: The handle of the door.
        """
        floor = world.get_semantic_annotations_by_type(Floor)[0]

        door_T_handle = HomogeneousTransformationMatrix.from_xyz_rpy(
            y=0.1,
            x=door.root.collision.min_point.x,
            reference_frame=door.root,
        )

        door_T_world = world.transform(door_T_handle, world.root)
        floor_bounding_box = floor.root.collision.as_bounding_box_collection_at_origin(
            world.root.global_pose
        )
        is_handle_in_room = floor_bounding_box.event.marginal(
            SpatialVariables.xy
        ).contains((door_T_world.x, door_T_world.y))

        if is_handle_in_room and self.opens_inward:
            door_T_handle = HomogeneousTransformationMatrix.from_xyz_rpy(
                y=0.1,
                x=door.root.collision.max_point.x,
                reference_frame=door.root,
                yaw=np.pi,
            )

        world_root_T_handle = world.transform(door_T_handle, world.root)
        handle_name = PrefixedName(name=f"{self.id}_handle", prefix=self.id)

        with world.modify_world():
            handle = Handle.create_with_new_body_in_world(
                name=handle_name,
                world=world,
                world_root_T_self=world_root_T_handle,
                scale=Scale(0.05, 0.02, 0.2),
            )
            door.add(handle, field_name="handle")
        return handle

    def _create_hinge_in_world(self, world: World, door: Door) -> Hinge:
        """
        Create the hinge (the joint that makes the door openable) of the door.

        :param world: The world where the hinge is created.
        :param door: The door to create the hinge for.
        :return: The hinge
        """
        world_root_T_hinge = door.calculate_world_T_hinge_based_on_handle(Vector3.Z())

        if self.opens_inward:
            lower = DerivativeMap(position=0.0)
            upper = DerivativeMap(position=np.pi / 2)
        else:
            upper = DerivativeMap(position=0.0)
            lower = DerivativeMap(position=-np.pi / 2)

        with world.modify_world():
            hinge = Hinge.create_with_new_body_in_world(
                name=PrefixedName(name="hinge", prefix=door.root.name.name),
                world=world,
                active_axis=Vector3.Z(),
                world_root_T_self=world_root_T_hinge,
                connection_limits=DegreeOfFreedomLimits(lower=lower, upper=upper),
            )
            door.add(hinge, field_name="mechanical_joint")

        return hinge


@dataclass
class EGRoom(EGWithID):
    room_type: RoomType
    """
    The generalized category of the room.
    """

    # Currently only rectangular rooms, could use footprint: list[tuple[float, float]] for L-Shaped rooms
    scale: EGScale
    """
    The scale of the room.
    """

    position: EGPosition
    """
    Position of the room's lower-left corner in the scene.
    """

    objects: list[EGObject] = field(default_factory=list)
    """
    List of the objects in the room.
    """

    walls: list[EGWall] = field(default_factory=list)
    """
    List of the walls in the room.
    """

    doors: list[EGDoor] = field(default_factory=list)
    """
    List of the doors in the room.
    """

    shelves: list[EGShelf] = field(default_factory=list)
    """
    List of the shelves in the room, each placed at its own room-frame pose.
    """

    tables: list[EGTableWithChairs] = field(default_factory=list)
    """
    List of the table-with-chairs groups in the room, each placed at its own
    room-frame pose.
    """

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "id": self.id,
            "room_type": self.room_type,
            "scale": to_json(self.scale),
            "position": to_json(self.position),
            "objects": to_json(self.objects),
            "walls": to_json(self.walls),
            "doors": to_json(self.doors),
            "shelves": to_json(self.shelves),
            "tables": to_json(self.tables),
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            id=data["id"],
            room_type=RoomType._value2member_map_.get(
                data["room_type"], RoomType.OTHER
            ),
            scale=EGScale._from_json(data["scale"], **kwargs),
            position=EGPosition._from_json(data["position"], **kwargs),
            objects=[EGObject._from_json(o, **kwargs) for o in data["objects"]],
            walls=[EGWall._from_json(w, **kwargs) for w in data["walls"]],
            doors=[EGDoor._from_json(d, **kwargs) for d in data["doors"]],
            shelves=[EGShelf._from_json(s, **kwargs) for s in data.get("shelves", [])],
            tables=[
                EGTableWithChairs._from_json(t, **kwargs)
                for t in data.get("tables", [])
            ],
        )

    def _create_floor(self, world: World, parent: KinematicStructureEntity) -> Floor:
        """
        Create the floor of this room spanning the area enclosed by the walls.

        :param world: The world to create the floor in.
        :param parent: The parent kinematic structure entity.
        :return: The annotation of the created floor.
        """
        floor_name = PrefixedName(name="floor", prefix=self.id)

        all_x = [p.x for w in self.walls for p in (w.start_point, w.end_point)]
        all_y = [p.y for w in self.walls for p in (w.start_point, w.end_point)]

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        floor_width = max_x - min_x
        floor_length = max_y - min_y
        x_center = (min_x + max_x) / 2
        y_center = (min_y + max_y) / 2

        parent_T_floor = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=x_center,
            y=y_center,
            z=self.position.z,
            reference_frame=parent,
        )

        with world.modify_world():
            floor_annotation = Floor.create_with_new_body_in_world(
                scale=Scale(x=floor_width, y=floor_length, z=0.01),
                world=world,
                name=floor_name,
                world_root_T_self=parent_T_floor,
            )

        return floor_annotation

    def spawn_in_world(
        self,
        world: World,
        object_id_to_mesh_path: dict[str, Path] | None,
        parent: KinematicStructureEntity,
    ) -> SpawnedRoom:
        """
        Instantiate the room -- floor, walls, doors, free floor objects, and
        nested shelves and tables -- returning handles for in-world validation
        and repositioning of the floor pieces.

        :param world: The world to spawn the room into.
        :param object_id_to_mesh_path: Mapping from a free floor object's id to
            its mesh directory, used to resolve per-object mesh paths. Several
            objects may map to the same directory, since one scene directory
            commonly holds many objects.
        :param parent: The parent entity the room's contents are placed under.
        :return: The spawned room and its handles.
        """
        floor_annotation = self._create_floor(world, parent)
        walls_of_room = []
        doors_of_room = []

        for wall in self.walls:
            wall_annotation = wall.create_in_world(world, parent)
            walls_of_room.append(wall_annotation)
            doors_of_this_wall = [
                door for door in self.doors if door.wall_id == wall.id
            ]
            doors_of_room += [
                door.create_in_world(
                    world,
                    wall_annotation.root,
                    wall=wall,
                    wall_annotation=wall_annotation,
                )
                for door in doors_of_this_wall
            ]

        room_annotation = RoomWithWallsAndDoors(
            floor=floor_annotation,
            walls=walls_of_room,
            doors=doors_of_room,
            room_type=self.room_type,
        )

        with world.modify_world():
            world.add_semantic_annotation(room_annotation)

        object_id_to_mesh_path = object_id_to_mesh_path or {}

        object_bodies: dict[int, Body] = {}
        for object_index, obj in enumerate(self.objects):
            mesh_path = object_id_to_mesh_path.get(obj.id)
            object_bodies[object_index] = obj.create_in_world(
                world, mesh_path, parent=parent
            )

        spawned_shelves = [shelf.spawn_in_world(world, parent) for shelf in self.shelves]
        spawned_tables = [table.spawn_in_world(world, parent) for table in self.tables]

        return SpawnedRoom(
            world=world,
            parent=parent,
            floor=floor_annotation,
            wall_bodies=[wall.root for wall in walls_of_room],
            object_bodies=object_bodies,
            spawned_shelves=spawned_shelves,
            spawned_tables=spawned_tables,
        )

    def create_in_world(
        self,
        world: World,
        object_id_to_mesh_path: dict[str, Path] | None,
        parent: KinematicStructureEntity,
        **kwargs,
    ) -> WorldEntity:
        """
        Instantiate the room inside *world*.

        Thin wrapper over :meth:`spawn_in_world` for callers that only need the
        populated world and not the per-piece handles.
        """
        self.spawn_in_world(world, object_id_to_mesh_path, parent)
        return world.root


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


class RoomWall(IntEnum):
    """
    The four walls of a rectangular room, indexed in the order
    :func:`~experiments.scene_generation_experiments.room_floor_sampling._rectangular_walls`
    builds them, so a wall index round-trips between a layout and a spawned room.
    """

    SOUTH = 0
    """
    The wall at the room's minimum y, running along the x-axis.
    """

    EAST = 1
    """
    The wall at the room's maximum x, running along the y-axis.
    """

    NORTH = 2
    """
    The wall at the room's maximum y, running along the x-axis.
    """

    WEST = 3
    """
    The wall at the room's minimum x, running along the y-axis.
    """

    @classmethod
    def nearest(cls, value: float) -> RoomWall:
        """
        Coerce a numeric wall index onto an actual wall.

        A fitted circuit models the wall index as a continuous variable, so a
        sampled pose carries a float rather than a member. Values are rounded to
        the nearest wall and clamped into range.

        :param value: The sampled wall index.
        :return: The wall it denotes.
        """
        walls = list(cls)
        return walls[min(max(int(round(value)), 0), len(walls) - 1)]

    @property
    def inward_normal_bearing_degrees(self) -> float:
        """
        Bearing, in degrees, of the direction pointing from this wall into the
        room. Measuring a piece's yaw against it puts the common case -- a
        shelf standing flat against the wall, facing the room -- at zero.
        """
        return {
            RoomWall.SOUTH: 90.0,
            RoomWall.EAST: 180.0,
            RoomWall.NORTH: -90.0,
            RoomWall.WEST: 0.0,
        }[self]

    @property
    def runs_along_x(self) -> bool:
        """
        Whether this wall extends along the room's x-axis, which decides which
        room extent measures along it and which measures away from it.
        """
        return self in (RoomWall.SOUTH, RoomWall.NORTH)


@dataclass
class EGWallRelativePose(EGBase):
    """
    Pose of a floor piece relative to the room wall it stands nearest, so that
    "shelves stand against a wall" is learnable as a single distance rather
    than as a relationship between two coordinates.

    A probability tree's leaf models every variable independently, so in
    Cartesian coordinates "against a wall" is the disjunction *x near an edge or
    y near an edge*, which a product of univariate marginals cannot express. As
    a perpendicular distance it collapses to one marginal concentrated near
    zero: measured over the dataset, shelves sit 0.25 m from a wall and
    cabinets 0.27 m, against 1.15 m for chairs and 1.25 m for tables.

    Together the three spatial fields re-parametrise the room rectangle
    completely -- every interior point has a nearest wall -- so a piece needs no
    "free-standing" special case: a table in the middle of the room simply has a
    large :attr:`distance_from_wall`.
    """

    wall: RoomWall
    """
    The room wall this piece stands nearest.
    """

    distance_from_wall: float
    """
    Perpendicular distance, in metres, from that wall to the piece centre.

    Kept in absolute metres rather than as a fraction because a shelf stands the
    same distance from a wall whatever the room's size. Nothing in a fitted
    circuit bounds it by the room, so :meth:`to_absolute_pose` clamps it.
    """

    position_along_wall: float
    """
    Position of the piece along that wall, as a fraction of the wall's length
    running from its minimum-coordinate end.

    A fraction rather than metres, so a position two thirds along a wall stays
    two thirds along it in a room of any size.
    """

    yaw_relative_to_wall: float
    """
    Yaw of the piece, in degrees, relative to the wall's inward normal.

    Zero means the piece faces straight into the room.
    """

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "wall": int(self.wall),
            "distance_from_wall": self.distance_from_wall,
            "position_along_wall": self.position_along_wall,
            "yaw_relative_to_wall": self.yaw_relative_to_wall,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            wall=RoomWall(data["wall"]),
            distance_from_wall=data["distance_from_wall"],
            position_along_wall=data["position_along_wall"],
            yaw_relative_to_wall=data["yaw_relative_to_wall"],
        )

    @classmethod
    def from_absolute_pose(
        cls, x: float, y: float, yaw_degrees: float, room_scale: EGScale
    ) -> Self:
        """
        Compute a piece's pose relative to its nearest wall, from a pose given
        in the room-centred frame.

        :param x: Position of the piece along the room's x-axis, measured from
            the room centre.
        :param y: Position of the piece along the room's y-axis, measured from
            the room centre.
        :param yaw_degrees: Absolute yaw of the piece, in degrees.
        :param room_scale: Footprint of the room the piece stands in.
        :return: The piece's pose relative to its nearest wall.
        """
        half_width = room_scale.width / 2
        half_length = room_scale.length / 2
        distance_to_wall = {
            RoomWall.SOUTH: y + half_length,
            RoomWall.EAST: half_width - x,
            RoomWall.NORTH: half_length - y,
            RoomWall.WEST: x + half_width,
        }
        wall = min(distance_to_wall, key=distance_to_wall.get)

        if wall.runs_along_x:
            position_along_wall = (x + half_width) / room_scale.width
        else:
            position_along_wall = (y + half_length) / room_scale.length

        return cls(
            wall=wall,
            distance_from_wall=distance_to_wall[wall],
            position_along_wall=position_along_wall,
            yaw_relative_to_wall=wrap_angle_degrees(
                yaw_degrees - wall.inward_normal_bearing_degrees
            ),
        )

    def to_absolute_pose(self, room_scale: EGScale) -> tuple[float, float, float]:
        """
        Convert this wall-relative pose back into a room-centred pose.

        The wall index is coerced with :meth:`RoomWall.nearest`, since a
        circuit samples it as a continuous value. The distance is clamped to
        ``min(half_width, half_length)``, so a
        distance drawn from a large room's marginal cannot place a piece outside
        a small one. That bound is the variable's true support:
        :meth:`from_absolute_pose` reports the distance to the *nearest* wall,
        which can never exceed the smaller half-extent, so clamping to it also
        keeps :attr:`wall` the nearest wall of the pose it produces.

        :param room_scale: Footprint of the room the piece stands in.
        :return: ``(x, y, yaw_degrees)`` of the piece in the room-centred frame.
        """
        wall = RoomWall.nearest(self.wall)
        half_width = room_scale.width / 2
        half_length = room_scale.length / 2
        distance = min(
            max(self.distance_from_wall, 0.0), min(half_width, half_length)
        )
        fraction = min(max(self.position_along_wall, 0.0), 1.0)

        if wall.runs_along_x:
            x = -half_width + fraction * room_scale.width
            y = (
                -half_length + distance
                if wall is RoomWall.SOUTH
                else half_length - distance
            )
        else:
            y = -half_length + fraction * room_scale.length
            x = (
                half_width - distance
                if wall is RoomWall.EAST
                else -half_width + distance
            )

        return (
            x,
            y,
            wrap_angle_degrees(
                self.yaw_relative_to_wall + wall.inward_normal_bearing_degrees
            ),
        )


@dataclass
class EGFloorPiece(EGBase):
    """
    One placeable resting on a room's floor, posed relative to the wall it
    stands nearest.

    Deliberately narrower than :class:`EGObject2D`, which describes an object on
    a shelf layer and carries the identifiers that come with a dataset row. A
    floor piece models only what generation actually learns -- what kind of
    thing it is, how big it is, and where it stands -- because the identifiers
    are near-unique per row and would otherwise grow the fitted circuit one leaf
    per training piece. Its mesh and identity are resolved after sampling, from
    the candidate pool.
    """

    object_type: ObjectType
    """
    The category of the piece.
    """

    scale: EGScale
    """
    Physical dimensions of the piece.
    """

    pose: EGWallRelativePose
    """
    Pose of the piece relative to its nearest wall.
    """

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "object_type": self.object_type,
            "scale": to_json(self.scale),
            "pose": to_json(self.pose),
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            object_type=ObjectType._value2member_map_.get(
                data["object_type"], ObjectType.OTHER
            ),
            scale=EGScale._from_json(data["scale"], **kwargs),
            pose=EGWallRelativePose._from_json(data["pose"], **kwargs),
        )


@dataclass(frozen=True)
class RoomInterior:
    """
    The rectangle a floor piece's centre may occupy for the piece to stay clear
    of the room's walls.

    :class:`EGWallRelativePose` bounds a piece's *centre* only, and each wall is
    built centred on the room's boundary, so it reaches half its thickness back
    into the room. A piece standing the measured 0.25 m from a wall therefore
    cuts into that wall as soon as it is deeper than 0.4 m, which most furniture
    is. Nothing downstream recovers from that: the collision resolver keeps a
    piece on the wall the circuit chose for it, so sliding it along that wall
    never clears the overlap and the piece is eventually dropped.
    """

    scale: EGScale
    """
    Footprint of the room.

    .. warning::
        Taken to be centred on the origin, as
        :func:`~experiments.scene_generation_experiments.room_floor_sampling._rectangular_walls`
        builds a generated room. Stored sage10k rooms put their lower-left
        corner at the origin instead, so their positions must be re-centred --
        as extraction already does -- before being measured against this.
    """

    wall_thickness: float
    """
    Thickness of the room's walls, each centred on the room's boundary.
    """

    WALL_CLEARANCE: ClassVar[float] = 0.001
    """
    Gap, in metres, kept between a contained footprint and a wall's inner face.

    Containing a piece exactly flush leaves the two surfaces at zero distance,
    which a collision check at ``distance=0.0`` reports as contact -- so a piece
    pushed precisely to its limit is flagged against the very wall it was just
    cleared of, and the repair loop cannot win.
    """

    @classmethod
    def of_room(cls, room: EGRoom) -> Self:
        """
        Read the interior off a room's own footprint and walls.

        :param room: The room to measure. Its footprint must be centred on the
            origin, see :attr:`scale`.
        :return: The interior its pieces must stay within.
        """
        return cls(
            scale=room.scale,
            wall_thickness=max((wall.thickness for wall in room.walls), default=0.0),
        )

    def centre_limits(
        self, footprint: EGScale, yaw_degrees: float
    ) -> tuple[float, float]:
        """
        Return how far from the room centre, along x and along y, the centre of a
        *footprint*-sized piece turned by *yaw_degrees* may sit.

        The footprint is bounded by the axis-aligned span its rotation makes it
        occupy, so a piece turned diagonally is held further from the walls than
        one standing square to them.

        :param footprint: Size of the piece.
        :param yaw_degrees: Yaw of the piece, in degrees.
        :return: The largest ``abs(x)`` and ``abs(y)`` its centre may have.
        """
        yaw_radians = math.radians(yaw_degrees)
        half_width = footprint.width / 2
        half_length = footprint.length / 2
        overhang_x = abs(half_width * math.cos(yaw_radians)) + abs(
            half_length * math.sin(yaw_radians)
        )
        overhang_y = abs(half_width * math.sin(yaw_radians)) + abs(
            half_length * math.cos(yaw_radians)
        )
        inner_face = self.wall_thickness / 2 + self.WALL_CLEARANCE
        return (
            max(self.scale.width / 2 - inner_face - overhang_x, 0.0),
            max(self.scale.length / 2 - inner_face - overhang_y, 0.0),
        )

    def contained_position(
        self, x: float, y: float, footprint: EGScale, yaw_degrees: float
    ) -> tuple[float, float]:
        """
        Return ``(x, y)`` moved the shortest distance that keeps a
        *footprint*-sized piece turned by *yaw_degrees* clear of every wall.

        :param x: Position of the piece centre along the room's x-axis.
        :param y: Position of the piece centre along the room's y-axis.
        :param footprint: Size of the piece.
        :param yaw_degrees: Yaw of the piece, in degrees.
        :return: The contained position, unchanged when the piece already fits.
        """
        limit_x, limit_y = self.centre_limits(footprint, yaw_degrees)
        return (
            min(max(x, -limit_x), limit_x),
            min(max(y, -limit_y), limit_y),
        )


@dataclass
class EGRoomFloorLayout(EGBase):
    """
    A room's floor arrangement for environment generation: the placeables
    resting directly on its floor, each posed relative to its nearest wall.

    Mirrors :class:`EGShelfLayer` -- it carries the room's own footprint so the
    circuit can condition on floor dimensions alongside which pieces a room
    holds and where, rather than inheriting a fixed size.
    """

    scale: EGScale
    """
    Footprint of the room floor (width × length × height).
    """

    pieces: list[EGFloorPiece]
    """
    Placeables resting on the floor, with positions relative to the room centre.
    """

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "scale": to_json(self.scale),
            "pieces": to_json(self.pieces),
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            scale=EGScale._from_json(data["scale"], **kwargs),
            pieces=[EGFloorPiece._from_json(p, **kwargs) for p in data["pieces"]],
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
class SpawnedRoom(SpawnedLayout):
    """
    A room instantiated in a :class:`World`, with handles for in-world
    validation and repositioning of its floor pieces before their contents are
    sampled.
    """

    parent: KinematicStructureEntity
    """
    The frame the room's contents' poses are expressed relative to.
    """

    floor: Floor
    """
    The room's floor annotation, whose surface the free floor objects and
    furniture pieces are placed on and resolved against.
    """

    wall_bodies: list[Body]
    """
    The room's wall bodies, kept as static obstacles the floor pieces must not
    collide with.
    """

    object_bodies: dict[int, Body]
    """
    Bodies spawned for the room's free floor objects, keyed by their index in
    :attr:`EGRoom.objects`.
    """

    spawned_shelves: list[SpawnedShelf]
    """
    Per-shelf spawn handles, in :attr:`EGRoom.shelves` order.
    """

    spawned_tables: list[SpawnedTableWithChairs]
    """
    Per-table spawn handles, in :attr:`EGRoom.tables` order.
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


@dataclass
class EGRelativePolarPose(EGBase):
    """
    Pose of a chair relative to its table, expressed in the table's own local
    frame (after subtracting the table's yaw), so that "evenly spaced, facing
    the table" is learnable independent of a table's absolute position or
    orientation in the room.
    """

    distance_from_table_center: float
    """
    Radial distance, in metres, from the table centre to the chair centre.
    """

    angle_from_table_center: float
    """
    Angle, in degrees, of the chair's position around the table centre,
    measured counter-clockwise from the table's own local x-axis.
    """

    facing_angle_relative_to_table: float
    """
    Yaw of the chair, in degrees, relative to the bearing that points from the
    chair straight at the table centre.

    Zero means the chair faces the table dead-on.
    """

    def to_json(self) -> dict[str, Any]:
        return {
            "distance_from_table_center": self.distance_from_table_center,
            "angle_from_table_center": self.angle_from_table_center,
            "facing_angle_relative_to_table": self.facing_angle_relative_to_table,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            distance_from_table_center=data["distance_from_table_center"],
            angle_from_table_center=data["angle_from_table_center"],
            facing_angle_relative_to_table=data["facing_angle_relative_to_table"],
        )

    @classmethod
    def from_absolute_poses(
        cls,
        chair_x: float,
        chair_y: float,
        chair_yaw_degrees: float,
        table_x: float,
        table_y: float,
        table_yaw_degrees: float,
    ) -> Self:
        """
        Compute a chair's pose relative to its table from both poses expressed
        in a shared absolute frame.

        :param chair_x: Absolute x position of the chair.
        :param chair_y: Absolute y position of the chair.
        :param chair_yaw_degrees: Absolute yaw of the chair, in degrees.
        :param table_x: Absolute x position of the table centre.
        :param table_y: Absolute y position of the table centre.
        :param table_yaw_degrees: Absolute yaw of the table, in degrees.
        :return: The chair's pose relative to the table.
        """
        local_offset = EGPoint2D(
            x=chair_x - table_x, y=chair_y - table_y
        ).rotated_into_frame(table_yaw_degrees)

        distance_from_table_center = math.hypot(local_offset.x, local_offset.y)
        angle_from_table_center = math.degrees(
            math.atan2(local_offset.y, local_offset.x)
        )

        bearing_to_table = wrap_angle_degrees(angle_from_table_center + 180)
        chair_yaw_relative_to_table = wrap_angle_degrees(
            chair_yaw_degrees - table_yaw_degrees
        )
        facing_angle_relative_to_table = wrap_angle_degrees(
            chair_yaw_relative_to_table - bearing_to_table
        )

        return cls(
            distance_from_table_center=distance_from_table_center,
            angle_from_table_center=angle_from_table_center,
            facing_angle_relative_to_table=facing_angle_relative_to_table,
        )

    def to_absolute_pose(
        self, table_x: float, table_y: float, table_yaw_degrees: float
    ) -> tuple[float, float, float]:
        """
        Convert this polar pose back into an absolute pose, given the table's
        own absolute pose.

        :param table_x: Absolute x position of the table centre.
        :param table_y: Absolute y position of the table centre.
        :param table_yaw_degrees: Absolute yaw of the table, in degrees.
        :return:``(x, y, yaw_degrees)`` of the chair in the table's
            absolute frame.
        """
        table_yaw_radians = math.radians(table_yaw_degrees)
        angle_radians = math.radians(self.angle_from_table_center)

        local_x = self.distance_from_table_center * math.cos(angle_radians)
        local_y = self.distance_from_table_center * math.sin(angle_radians)

        world_dx = local_x * math.cos(table_yaw_radians) - local_y * math.sin(
            table_yaw_radians
        )
        world_dy = local_x * math.sin(table_yaw_radians) + local_y * math.cos(
            table_yaw_radians
        )

        bearing_to_table = wrap_angle_degrees(self.angle_from_table_center + 180)
        chair_yaw_relative_to_table = wrap_angle_degrees(
            self.facing_angle_relative_to_table + bearing_to_table
        )
        chair_yaw_world = table_yaw_degrees + chair_yaw_relative_to_table

        return table_x + world_dx, table_y + world_dy, chair_yaw_world


@dataclass
class EGChair(EGWithID):
    """
    A chair belonging to a table-with-chairs group, positioned relative to the
    table via a polar pose rather than absolute Cartesian coordinates.
    """

    room_id: str
    """
    The id of the room where the chair is located.
    """

    object_type: ObjectType
    """
    The category of the object (normally :attr:`ObjectType.CHAIR`).
    """

    scale: EGScale
    """
    Physical dimensions of the chair.
    """

    relative_pose: EGRelativePolarPose
    """
    Pose of the chair relative to the table centre, in the table's local frame.
    """

    source_id: str
    """
    Identifier used to look up the PLY mesh file for this chair in the dataset.
    """

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "id": self.id,
            "room_id": self.room_id,
            "type": self.object_type,
            "scale": to_json(self.scale),
            "relative_pose": to_json(self.relative_pose),
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
            scale=EGScale._from_json(data["scale"], **kwargs),
            relative_pose=EGRelativePolarPose._from_json(
                data["relative_pose"], **kwargs
            ),
            source_id=data["source_id"],
        )

    def create_in_world(
        self,
        world: World,
        mesh_path: Path | None,
        parent: KinematicStructureEntity,
        table_position: EGPoint2D | None = None,
        table_orientation: EGRotation | None = None,
        world_pose: HomogeneousTransformationMatrix | None = None,
        **kwargs,
    ) -> Body:
        """
        Instantiate this chair in *world*, converting its table-relative polar
        pose into an absolute pose using the table's own position and
        orientation.

        :param world: The world where the chair is created.
        :param mesh_path: Directory containing the ``objects/`` sub-
            folder with PLY and texture files for this chair.
        :param parent: The parent kinematic structure entity.
        :param table_position: Absolute position of the table centre; required
            unless *world_pose* is given.
        :param table_orientation: Absolute orientation of the table; required
            unless *world_pose* is given.
        :param world_pose: When given, the chair is placed at this pose and the
            table pose is ignored, so a caller that already computed the pose
            can reuse it for both spawning and later repositioning.
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
            absolute_x, absolute_y, chair_yaw_world = (
                self.relative_pose.to_absolute_pose(
                    table_position.x, table_position.y, table_orientation.z
                )
            )
            root_T_body = HomogeneousTransformationMatrix.from_xyz_rpy(
                absolute_x,
                absolute_y,
                0.0,
                0.0,
                0.0,
                math.radians(chair_yaw_world),
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
        # a fixed parent expression keeps the chair movable: the ``.origin``
        # setter can later reposition it in place.
        body.parent_connection.origin = root_T_body

        annotation = NaturalLanguageWithTypeDescription(
            root=body, description=None, type_description=self.object_type
        )

        with world.modify_world():
            world.add_semantic_annotation(annotation)

        return body


@dataclass
class SpawnedTableWithChairs(SpawnedLayout):
    """
    A table-with-chairs group instantiated in a :class:`World`, with handles for
    in-world validation and repositioning of its chairs.
    """

    parent: KinematicStructureEntity
    """
    The frame the table's own pose is expressed relative to.
    """

    table: Body
    """
    The table's body; the chairs hang under it, so moving it moves the whole
    group as one unit.
    """

    chair_bodies: dict[int, Body]
    """
    Bodies spawned for the chairs, keyed by their index in
    :attr:`EGTableWithChairs.chairs`; chairs skipped at spawn have no entry.
    """


@dataclass
class EGTableWithChairs(EGBase):
    """
    A table together with the chairs clustered around it via spatial proximity,
    since chairs do not carry a ``place_id`` link to their table in the source
    data (unlike shelf contents, which do).
    """

    position: EGPoint2D
    """
    Position of the table's centre, relative to its parent frame.
    """

    scale: EGScale
    """
    Scale of the table.
    """

    orientation: EGRotation
    """
    Orientation of the table relative to its parent frame; every chair's
    :attr:`EGChair.relative_pose` is expressed relative to this table's own
    yaw.
    """

    chairs: list[EGChair]
    """
    Chairs clustered around this table, with poses relative to the table centre
    and yaw.
    """

    source_ids: list[MeshCandidate] | None = field(default=None)
    """
    Pool of candidate meshes used when placing chairs around the table.
    """

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "position": to_json(self.position),
            "scale": to_json(self.scale),
            "orientation": to_json(self.orientation),
            "chairs": to_json(self.chairs),
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls(
            position=EGPoint2D._from_json(data["position"], **kwargs),
            scale=EGScale._from_json(data["scale"], **kwargs),
            orientation=EGRotation._from_json(data["orientation"], **kwargs),
            chairs=[EGChair._from_json(c, **kwargs) for c in data["chairs"]],
        )

    def chair_local_pose(
        self, chair: EGChair, table: KinematicStructureEntity
    ) -> HomogeneousTransformationMatrix:
        """
        Compute a chair's pose in the table's own frame from its table-relative
        polar pose.

        Chairs are children of the table body, so their pose is expressed
        relative to it: the table carries the group's world position and yaw, and
        moving the table moves every chair with it. The chair's polar pose is
        evaluated in a table-at-origin frame, and lowered by half the table
        height so the chair still stands on the floor (the table body sits at
        half its height). Used both when first spawning a chair and when moving
        it to a resampled pose, so the two placements never drift -- and it stays
        correct after the whole group is repositioned.

        :param chair: The chair whose pose is computed.
        :param table: The table body the pose is expressed relative to.
        :return: The chair body's pose in the table frame.
        """
        local_x, local_y, chair_yaw = chair.relative_pose.to_absolute_pose(0.0, 0.0, 0.0)
        return HomogeneousTransformationMatrix.from_xyz_rpy(
            local_x,
            local_y,
            -self.scale.height / 2,
            0.0,
            0.0,
            math.radians(chair_yaw),
            reference_frame=table,
        )

    def spawn_in_world(
        self,
        world: World | None = None,
        parent: KinematicStructureEntity | None = None,
    ) -> SpawnedTableWithChairs:
        """
        Instantiate the table and its chairs inside a :class:`World`, returning
        handles to the table annotation and chair bodies.

        The handles let a caller validate and reposition individual chairs in
        the spawned world without rebuilding it.

        :param world: Existing world to extend. A fresh world with a ``map``
            root body is created when omitted.
        :param parent: The parent entity the table's own
            :attr:`position`/ :attr:`orientation` are expressed relative to.
            Defaults to the world's root when omitted.
        :return: The spawned group, its world, and per-chair handles.
        """
        _world: World = world if world is not None else World()
        if world is None:
            root = Body(name=PrefixedName(name="map"))
            with _world.modify_world():
                _world.add_body(root)

        _parent = parent if parent is not None else _world.root

        table_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=self.position.x,
            y=self.position.y,
            z=self.scale.height / 2,
            yaw=math.radians(self.orientation.z),
            reference_frame=_parent,
        )
        with _world.modify_world():
            table_annotation = Table.create_with_new_body_in_world(
                name=PrefixedName(name="table"),
                world=_world,
                world_root_T_self=table_pose,
                scale=Scale(
                    x=self.scale.length, y=self.scale.width, z=self.scale.height
                ),
            )
        table_body = table_annotation.root
        # Make the whole group a movable unit: a room-level resolver repositions
        # it by setting the table origin, and its chairs follow.
        _world.make_branch_movable(table_body)

        mesh_matcher = _MeshTypeMatcher(candidates=self.source_ids or [])

        chair_bodies: dict[int, Body] = {}
        for i, chair in enumerate(self.chairs):
            if not self.source_ids:
                continue
            candidate = mesh_matcher.random_match(chair.object_type)
            if candidate is None:
                continue
            chair.source_id = candidate.source_id
            chair_bodies[i] = chair.create_in_world(
                _world,
                candidate.scene_dir,
                parent=table_body,
                world_pose=self.chair_local_pose(chair, table_body),
            )

        return SpawnedTableWithChairs(
            world=_world,
            parent=_parent,
            table=table_body,
            chair_bodies=chair_bodies,
        )

    def create_in_world(
        self,
        world: World | None = None,
        parent: KinematicStructureEntity | None = None,
    ) -> World:
        """
        Instantiate the table and its chairs inside a :class:`World`.

        Thin wrapper over :meth:`spawn_in_world` for callers that only need the
        resulting world and not the per-chair body handles.

        :param world: Existing world to extend. A fresh world with a ``map``
            root body is created when omitted.
        :param parent: The parent entity the table's own
            :attr:`position`/ :attr:`orientation` are expressed relative to.
            Defaults to the world's root when omitted, so standalone callers
            are unaffected.
        :return: The world containing the table and its chairs.
        """
        return self.spawn_in_world(world, parent).world


@dataclass
class SceneGenerator(EGWithID):
    room: EGRoom
    """
    The room of the scene.

    Currently only one room is supported for simplicity.
    """

    object_id_to_mesh_path: dict[str, Path] = field(default_factory=dict)
    """
    A mapping from a free floor object's id to its mesh directory path.

    Used to resolve per-object mesh paths when creating the world.
    """

    def to_json(self) -> dict[str, Any]:
        return {
            **super().to_json(),
            "id": self.id,
            "room": to_json(self.room),
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs):
        return cls(
            id=data["id"],
            room=EGRoom._from_json(data["rooms"][0], **kwargs),
        )

    def create_world(self) -> World:
        world = World()
        root = Body(name=PrefixedName(name="map"))

        with world.modify_world():
            world.add_body(root)

        self.room.create_in_world(world, self.object_id_to_mesh_path, root)

        return world
