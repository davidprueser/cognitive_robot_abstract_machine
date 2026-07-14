from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Self, assert_never

import numpy as np
import trimesh

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
        mesh_to_object_mapping: dict[Path, EGObject] | None,
        parent: KinematicStructureEntity,
        **kwargs,
    ) -> WorldEntity:
        """
        Create the object in the world by getting its geometry from the
        provided information.

        :param world: The world where the object is created.
        :param mesh_to_object_mapping: A mapping from mesh paths to
            object information.
        :param parent: The parent of the object in the world.
        :param kwargs: Additional keyword arguments.
        :return: The relevant created body
        """


@dataclass
class EGSize(EGBase):
    """
    The scale of an object.
    """

    height: float
    """
    Scale in z (vertical axis).
    """

    length: float
    """
    Depth of the object (shelf depth direction, world x-axis when placed as a
    corpus).
    """

    width: float
    """
    Face width of the object (shelf face direction, world y-axis when placed as
    a corpus).
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


class ObjectType(enum.Enum):
    """
    Canonical object types present in the sage10k dataset.
    """

    ADJUSTABLEWRENCH = "adjustablewrench"
    ART = "art"
    BAKINGPOWDER1 = "bakingpowder1"
    BAKINGPOWDER2 = "bakingpowder2"
    BOOK = "book"
    BOOK1 = "book1"
    BOOK2 = "book2"
    BOOK4E33D6C6 = "book4e33d6c6"
    BOOK_SHELF_D8061277 = "book_shelf_d8061277"
    BOOK_SHELF_F9F248CD = "book_shelf_f9f248cd"
    BOOK_TABLE1 = "book_table1"
    BOOK_TABLE2 = "book_table2"
    BOOKCHAIR6 = "bookchair6"
    BOOKCHAIR8 = "bookchair8"
    BOOKCHAIR8EBA7FDC = "bookchair8eba7fdc"
    BOOKCHAIR9 = "bookchair9"
    BOOKMUSTARD = "bookmustard"
    BOOKMUSTARD4E33D6C6 = "bookmustard4e33d6c6"
    BOOKOLIVE2 = "bookolive2"
    CABINET = "cabinet"
    CANDLE2 = "candle2"
    CART = "cart"
    CHAIR = "chair"
    CHANGEJAR = "changejar"
    CLOCK = "clock"
    CONTAINER = "container"
    CONTAINER2 = "container2"
    CONTAINER_1 = "container_1"
    CONTAINER_2 = "container_2"
    CONTAINER_3 = "container_3"
    CONTAINER_CABINET_250E2E93 = "container_cabinet_250e2e93"
    CONTAINER_CABINET_88534706 = "container_cabinet_88534706"
    CONTAINER_CABINET_B7A01281 = "container_cabinet_b7a01281"
    CONTAINER_SHELF = "container_shelf"
    CONTAINERCABINET250 = "containercabinet250"
    CONTAINERCABINET88534706 = "containercabinet88534706"
    COUNTER = "counter"
    CROISSANT = "croissant"
    CROISSANT1 = "croissant1"
    CUP = "cup"
    CUP1 = "cup1"
    CUP2 = "cup2"
    CUP_TABLE1 = "cup_table1"
    CUP_TABLE2 = "cup_table2"
    DISPLAYCASE = "displaycase"
    DRILL = "drill"
    DRYER = "dryer"
    FLOURBAG = "flourbag"
    FOUNTAIN = "fountain"
    HAMMER = "hammer"
    LADDER = "ladder"
    LAUNDRYBASKET = "laundrybasket"
    LIGHT_FIXTURE = "light_fixture"
    LIGHTFIXTURE = "lightfixture"
    LIGHTING = "lighting"
    MEASURINGCUP1 = "measuringcup1"
    MEASURINGCUP2 = "measuringcup2"
    MEASURINGCUP3 = "measuringcup3"
    MEASURINGCUP4 = "measuringcup4"
    MEASURINGCUPS = "measuringcups"
    MIRROR = "mirror"
    MIXINGBOWL = "mixingbowl"
    MIXINGBOWL2 = "mixingbowl2"
    NEON = "neon"
    NOTEBOOK = "notebook"
    NOTEBOOK1 = "notebook1"
    NOTEBOOKEXTRA = "notebookextra"
    NOTEPAD = "notepad"
    OVEN = "oven"
    PAINTING = "painting"
    PAPERTOWELDISPENSER = "papertoweldispenser"
    PEGBOARD = "pegboard"
    PEN = "pen"
    PEN2 = "pen2"
    PEN_TABLE2 = "pen_table2"
    PENCOUNTER = "pencounter"
    PENEXTRA = "penextra"
    PENSHELF = "penshelf"
    PIPINGBAG = "pipingbag"
    PIPINGBAG1 = "pipingbag1"
    PLANT = "plant"
    PLANT1 = "plant1"
    PLANTFLOOR = "plantfloor"
    PLASTICBIN_2 = "plasticbin_2"
    PLIERS = "pliers"
    POSTER = "poster"
    POSTERWALL = "posterwall"
    PRINT = "print"
    RADIO = "radio"
    ROLLINGPIN = "rollingpin"
    ROLLINGPIN1 = "rollingpin1"
    ROLLINGPIN2 = "rollingpin2"
    SANDER = "sander"
    SCONCE = "sconce"
    SCONCEWALL = "sconcewall"
    SCREWDRIVER = "screwdriver"
    SHELF = "shelf"
    SHELFBOOK_3 = "shelfbook_3"
    SHELFPEN = "shelfpen"
    SHOWCASE = "showcase"
    SIGN = "sign"
    SIGNWALL = "signwall"
    SOAPDISPENSER = "soapdispenser"
    SPATULA = "spatula"
    STAINEDGLASS = "stainedglass"
    STOOL = "stool"
    STORAGEBIN = "storagebin"
    STORAGEBIN_FLOOR = "storagebin_floor"
    SUCCULENT = "succulent"
    SUGARJAR = "sugarjar"
    SUGARJAR1 = "sugarjar1"
    SUGARJAR2 = "sugarjar2"
    SUGARJAR3 = "sugarjar3"
    TABLE = "table"
    TIRE = "tire"
    TOOLBOX = "toolbox"
    TOOLBOX_FLOOR = "toolbox_floor"
    TRASH = "trash"
    WALLART = "wallart"
    WALLART_1 = "wallart_1"
    WALLPAINTING = "wallpainting"
    WALLSCONCE = "wallsconce"
    WALLSIGN = "wallsign"
    WASHER = "washer"
    WORKBENCH = "workbench"
    WORKBENCHCUP = "workbenchcup"
    WORKBENCHNOTEBOOK = "workbenchnotebook"
    WRENCH = "wrench"
    OTHER = "other"


class BookObjectType(enum.Enum):
    """
    Object types that represent actual books (not furniture named with 'book').
    """

    BOOK = "book"
    BOOK1 = "book1"
    BOOK2 = "book2"
    BOOK4E33D6C6 = "book4e33d6c6"
    BOOKMUSTARD = "bookmustard"
    BOOKMUSTARD4E33D6C6 = "bookmustard4e33d6c6"
    BOOKOLIVE2 = "bookolive2"
    SHELFBOOK_3 = "shelfbook_3"
    NOTEBOOK = "notebook"
    NOTEBOOK1 = "notebook1"
    NOTEBOOKEXTRA = "notebookextra"
    NOTEPAD = "notepad"

    @classmethod
    def contains(cls, object_type: "ObjectType") -> bool:
        """
        Return True if *object_type* represents a book.
        """
        return object_type.value in cls._value2member_map_


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
    floor, table.
    """

    object_type: ObjectType
    """
    The type of the object.
    """

    scale: EGSize
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
            scale=EGSize._from_json(data["dimensions"], **kwargs),
            source_id=data["source_id"],
        )

    def create_in_world(
        self,
        world: World,
        mesh_path: Path | None,
        parent: KinematicStructureEntity,
        **kwargs,
    ) -> Body:
        """
        Instantiate this object in *world* by loading its PLY mesh from
        *mesh_path*.

        :param world: The world where the object is created.
        :param mesh_path: Directory containing the ``objects/`` sub-
            folder with PLY and texture files for this object.
        :param parent: The parent kinematic structure entity.
        :raises ValueError: If *mesh_path* does not exist.
        :return: The created :class:`Body`.
        """
        if mesh_path is None:
            mesh_path = (
                Path.home()
                / "Documents"
                / "sage-10k-scenes"
                / "20251230_060038_layout_fd6894a7"
            )
        if not mesh_path.exists():
            raise ValueError(f"Directory {mesh_path} does not exist.")
        ply_file = mesh_path / "objects" / f"{self.source_id}.ply"
        texture_file = mesh_path / "objects" / f"{self.source_id}_texture.png"

        body = Body()
        body.name = PrefixedName(name=str(body.id), prefix=self.id)

        root_T_body = HomogeneousTransformationMatrix.from_xyz_rpy(
            self.position.x,
            self.position.y,
            self.position.z,
            *self.orientation.as_roll_pitch_yaw_in_radians(),
            reference_frame=parent,
            child_frame=body,
        )

        mesh = Mesh.from_ply_file(
            ply_file_path=str(ply_file),
            texture_file_path=str(texture_file),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=body),
        )

        geometry = ShapeCollection([mesh], reference_frame=body)
        body.visual = geometry
        body.collision = geometry

        if self.place_id in ["floor", "wall"]:
            connection_type = FixedConnection
        else:
            connection_type = Connection6DoF

        with world.modify_world():
            root_C_body = connection_type.create_with_dofs(
                world=world,
                parent=parent,
                child=body,
                parent_T_connection_expression=root_T_body,
            )
            world.add_body(body)
            world.add_connection(root_C_body)

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

    scale: EGSize
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
            scale=EGSize._from_json(data["dimensions"], **kwargs),
            source_id=data["source_id"],
        )

    def _scale_to_match_extents(self, native_extents: np.ndarray) -> Scale:
        """
        Compute the per-axis factor that rescales a mesh with *native_extents*
        so its bounding box matches this object's declared :attr:`scale`.

        Mirrors the width/length/height to x/y/z axis convention used by
        the collision-resolution box proxy, so the rendered mesh agrees
        with the geometry that was already checked for collisions. Axes
        with zero native extent are left unscaled.

        :param native_extents: The mesh's raw (x, y, z) bounding-box
            size, read before any scale is applied.
        :return: A :class:`Scale` with one factor per axis.
        """
        target_extents = (self.scale.width, self.scale.length, self.scale.height)
        return Scale(
            *(
                target / native if native > 0 else 1.0
                for target, native in zip(target_extents, native_extents)
            )
        )

    def create_in_world(
        self,
        world: World,
        mesh_path: Path | None,
        parent: KinematicStructureEntity,
        x: float | None = None,
        y: float | None = None,
        z: float = 0.0,
        **kwargs,
    ) -> Body:
        """
        Instantiate this object in *world* at the given absolute pose.

        The mesh is rescaled so its bounding box matches this object's
        declared :attr:`scale`, since PLY assets are otherwise rendered
        at their native size regardless of the sampled/declared
        dimensions.

        :param world: The world where the object is created.
        :param mesh_path: Directory containing the ``objects/`` sub-
            folder with PLY and texture files for this object.
        :param parent: The parent kinematic structure entity.
        :param x: Absolute x in world coordinates (defaults to
            ``self.position.x``).
        :param y: Absolute y in world coordinates (defaults to
            ``self.position.y``).
        :param z: Absolute z in world coordinates.
        :raises ValueError: If *mesh_path* does not exist.
        :return: The created :class:`Body`.
        """
        if mesh_path is None:
            mesh_path = (
                Path.home()
                / "Documents"
                / "sage-10k-scenes"
                / "20251230_060038_layout_fd6894a7"
            )
        if not mesh_path.exists():
            raise ValueError(f"Directory {mesh_path} does not exist.")
        ply_file = mesh_path / "objects" / f"{self.source_id}.ply"
        texture_file = mesh_path / "objects" / f"{self.source_id}_texture.png"

        body = Body()
        body.name = PrefixedName(name=str(body.id), prefix=self.id)

        root_T_body = HomogeneousTransformationMatrix.from_xyz_rpy(
            self.position.x if x is None else x,
            self.position.y if y is None else y,
            z,
            *self.orientation.as_roll_pitch_yaw_in_radians(),
            reference_frame=parent,
            child_frame=body,
        )

        native_extents = trimesh.load(str(ply_file), process=False).extents
        mesh = Mesh.from_ply_file(
            ply_file_path=str(ply_file),
            texture_file_path=str(texture_file),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=body),
            scale=self._scale_to_match_extents(native_extents),
        )

        geometry = ShapeCollection([mesh], reference_frame=body)
        body.visual = geometry
        body.collision = geometry

        with world.modify_world():
            root_C_body = Connection6DoF.create_with_dofs(
                world=world,
                parent=parent,
                child=body,
                parent_T_connection_expression=root_T_body,
            )
            world.add_body(body)
            world.add_connection(root_C_body)

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
    room_type: str
    """
    The type of the room.
    """

    # Currently only rectangular rooms, could use footprint: list[tuple[float, float]] for L-Shaped rooms
    scale: EGSize
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
            room_type=data["room_type"],
            scale=EGSize._from_json(data["scale"], **kwargs),
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

    def create_in_world(
        self,
        world: World,
        mesh_to_object_mapping: dict[Path, EGObject] | None,
        parent: KinematicStructureEntity,
        **kwargs,
    ) -> WorldEntity:

        floor_annotation = self._create_floor(world, parent)
        walls_of_room = []
        doors_of_room = []

        for wall in self.walls:
            wall_annotation = wall.create_in_world(world, parent)
            walls_of_room.append(wall_annotation)
            doors_of_this_wall = [door for door in self.doors if door.wall_id == wall.id]
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

        object_to_mesh_path: dict[str, Path] = (
            {obj.id: path for path, obj in mesh_to_object_mapping.items()}
            if mesh_to_object_mapping
            else {}
        )

        for obj in self.objects:
            mesh_path = object_to_mesh_path.get(obj.id)
            obj.create_in_world(world, mesh_path, parent=parent)

        for shelf in self.shelves:
            shelf.create_in_world(world, parent=parent)

        for table in self.tables:
            table.create_in_world(world, parent=parent)

        return world.root


@dataclass
class EGShelfLayer(EGBase):
    """
    A shelf layer for environment generation.

    Carries its own physical dimensions so the RSPN can learn width and
    length alongside object placement, rather than inheriting a fixed
    size from the parent shelf.
    """

    scale: EGSize
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
            scale=EGSize._from_json(data["scale"], **kwargs),
            objects=[EGObject2D._from_json(o, **kwargs) for o in data["objects"]],
        )


@dataclass
class _MeshSizeMatcher:
    """
    Selects, from a pool of candidate meshes, the one whose native bounding box
    is closest to a target :class:`EGSize`.

    Object-type labels in the source dataset are effectively per-
    instance identifiers rather than real categories, so matching by
    declared size instead of by label is what keeps a randomly-drawn
    mesh visually plausible for the scale an object was sampled at.
    """

    candidates: list[tuple[Path, str]]
    """
    (scene_dir, source_id) pairs to choose from.
    """

    _native_extents_by_source_id: dict[str, np.ndarray] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    Cache of source_id -> native (x, y, z) mesh extents, populated lazily since
    reading every candidate's PLY file is only needed once per pool.
    """

    def _native_extents(self, scene_dir: Path, source_id: str) -> np.ndarray:
        if source_id not in self._native_extents_by_source_id:
            ply_file = scene_dir / "objects" / f"{source_id}.ply"
            self._native_extents_by_source_id[source_id] = trimesh.load(
                str(ply_file), process=False
            ).extents
        return self._native_extents_by_source_id[source_id]

    def closest_match(self, target_scale: EGSize) -> tuple[Path, str]:
        """
        Return the candidate whose native bounding box is closest to
        *target_scale*, using the same width/length/height to x/y/z axis
        convention as the collision-resolution box proxy.

        :param target_scale: The declared size to match against.
        :return: The best-matching (scene_dir, source_id) pair.
        """
        if len(self.candidates) == 1:
            return self.candidates[0]
        target_extents = np.array(
            [target_scale.width, target_scale.length, target_scale.height]
        )
        return min(
            self.candidates,
            key=lambda candidate: np.linalg.norm(
                self._native_extents(*candidate) - target_extents
            ),
        )


@dataclass
class EGShelf(EGBase):
    """
    A shelf with four explicit horizontal layers.
    """

    position: EGPoint2D
    """
    Position of the Shelf, relative to its parent frame.
    """

    scale: EGSize
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

    source_ids: list[tuple[Path, str]] | None = field(default=None)
    """
    List of (scene_dir, source_id) pairs for meshes used when placing objects
    on shelf layers.
    """

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
            scale=EGSize._from_json(data["scale"], **kwargs),
            orientation=EGRotation._from_json(data["orientation"], **kwargs),
            layers=[EGShelfLayer._from_json(l, **kwargs) for l in data["layers"]],
        )

    def create_in_world(
        self,
        world: World | None = None,
        parent: KinematicStructureEntity | None = None,
    ) -> World:
        """
        Instantiate the shelf and its objects inside a :class:`World`.

        :param world: Existing world to extend. A fresh world with a
            ``map`` root body is created when omitted.
        :param parent: The parent entity the shelf's own
            :attr:`position`/ :attr:`orientation` are expressed relative
            to. Defaults to the world's root when omitted, so standalone
            callers are unaffected.
        :return: The world containing the shelf.
        """
        _world: World = world if world is not None else World()
        if world is None:
            root = Body(name=PrefixedName(name="map"))
            with _world.modify_world():
                _world.add_body(root)

        _parent = parent if parent is not None else _world.root

        corpus_face = max(layer.scale.width for layer in self.layers)
        corpus_depth = max(layer.scale.length for layer in self.layers)
        corpus_height = self.scale.height
        yaw_radians = math.radians(self.orientation.z)

        corpus_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=self.position.x,
            y=self.position.y,
            z=corpus_height / 2,
            yaw=yaw_radians,
            reference_frame=_parent,
        )
        with _world.modify_world():
            Cabinet.create_with_new_body_in_world(
                name=PrefixedName(name="shelf_corpus"),
                world=_world,
                world_root_T_self=corpus_pose,
                scale=Scale(x=corpus_depth, y=corpus_face, z=corpus_height),
                wall_thickness=0.03,
            )

        step = corpus_height / (len(self.layers) + 1)
        layer_z_heights = [step * (i + 1) for i in range(len(self.layers))]

        mesh_matcher = _MeshSizeMatcher(candidates=self.source_ids or [])
        cos_yaw = math.cos(yaw_radians)
        sin_yaw = math.sin(yaw_radians)

        for i, (layer, z_height) in enumerate(zip(self.layers, layer_z_heights)):
            layer_scale = Scale(x=layer.scale.length, y=layer.scale.width, z=0.02)
            layer_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=self.position.x,
                y=self.position.y,
                z=z_height,
                yaw=yaw_radians,
                reference_frame=_parent,
            )
            with _world.modify_world():
                ShelfLayer.create_with_new_body_in_world(
                    name=PrefixedName(name=f"layer_{i}"),
                    world=_world,
                    world_root_T_self=layer_pose,
                    scale=layer_scale,
                )

            for obj in layer.objects:
                if not isinstance(obj.position.x, (int, float)):
                    continue
                # obj.position.y/x map to the shelf's local x/y axes (the
                # shelf's own width=y-face/length=x-depth convention); rotate
                # that local offset by the shelf's own yaw before translating
                # by self.position, so objects turn with the shelf.
                local_dx = obj.position.y
                local_dy = obj.position.x
                rotated_dx = local_dx * cos_yaw - local_dy * sin_yaw
                rotated_dy = local_dx * sin_yaw + local_dy * cos_yaw
                absolute_x = self.position.x + rotated_dx
                absolute_y = self.position.y + rotated_dy
                absolute_z = z_height + layer_scale.z / 2

                if not self.source_ids:
                    continue
                scene_dir, source_id = mesh_matcher.closest_match(obj.scale)
                obj.source_id = source_id
                # Compound the shelf's own yaw into the object's orientation
                # so it turns together with the shelf, not just its position.
                rotated_object = replace(
                    obj,
                    orientation=EGRotation(
                        x=obj.orientation.x,
                        y=obj.orientation.y,
                        z=obj.orientation.z + self.orientation.z,
                    ),
                )
                rotated_object.create_in_world(
                    _world,
                    scene_dir,
                    parent=_parent,
                    x=absolute_x,
                    y=absolute_y,
                    z=absolute_z,
                )

        return _world


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
        delta_x = chair_x - table_x
        delta_y = chair_y - table_y
        table_yaw_radians = math.radians(table_yaw_degrees)

        local_x = delta_x * math.cos(table_yaw_radians) + delta_y * math.sin(
            table_yaw_radians
        )
        local_y = -delta_x * math.sin(table_yaw_radians) + delta_y * math.cos(
            table_yaw_radians
        )

        distance_from_table_center = math.hypot(local_x, local_y)
        angle_from_table_center = math.degrees(math.atan2(local_y, local_x))

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

    scale: EGSize
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
            scale=EGSize._from_json(data["scale"], **kwargs),
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
        table_position: EGPoint2D,
        table_orientation: EGRotation,
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
        :param table_position: Absolute position of the table centre.
        :param table_orientation: Absolute orientation of the table.
        :raises ValueError: If *mesh_path* does not exist.
        :return: The created :class:`Body`.
        """
        if mesh_path is None:
            mesh_path = (
                Path.home()
                / "Documents"
                / "sage-10k-scenes"
                / "20251230_060038_layout_fd6894a7"
            )
        if not mesh_path.exists():
            raise ValueError(f"Directory {mesh_path} does not exist.")
        ply_file = mesh_path / "objects" / f"{self.source_id}.ply"
        texture_file = mesh_path / "objects" / f"{self.source_id}_texture.png"

        absolute_x, absolute_y, chair_yaw_world = self.relative_pose.to_absolute_pose(
            table_position.x, table_position.y, table_orientation.z
        )

        body = Body()
        body.name = PrefixedName(name=str(body.id), prefix=self.id)

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

        native_extents = trimesh.load(str(ply_file), process=False).extents
        target_extents = (self.scale.width, self.scale.length, self.scale.height)
        mesh_scale = Scale(
            *(
                target / native if native > 0 else 1.0
                for target, native in zip(target_extents, native_extents)
            )
        )
        mesh = Mesh.from_ply_file(
            ply_file_path=str(ply_file),
            texture_file_path=str(texture_file),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=body),
            scale=mesh_scale,
        )

        geometry = ShapeCollection([mesh], reference_frame=body)
        body.visual = geometry
        body.collision = geometry

        with world.modify_world():
            root_C_body = Connection6DoF.create_with_dofs(
                world=world,
                parent=parent,
                child=body,
                parent_T_connection_expression=root_T_body,
            )
            world.add_body(body)
            world.add_connection(root_C_body)

        annotation = NaturalLanguageWithTypeDescription(
            root=body, description=None, type_description=self.object_type
        )

        with world.modify_world():
            world.add_semantic_annotation(annotation)

        return body


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

    scale: EGSize
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

    source_ids: list[tuple[Path, str]] | None = field(default=None)
    """
    List of (scene_dir, source_id) pairs for meshes used when placing chairs
    around the table.
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
            scale=EGSize._from_json(data["scale"], **kwargs),
            orientation=EGRotation._from_json(data["orientation"], **kwargs),
            chairs=[EGChair._from_json(c, **kwargs) for c in data["chairs"]],
        )

    def create_in_world(
        self,
        world: World | None = None,
        parent: KinematicStructureEntity | None = None,
    ) -> World:
        """
        Instantiate the table and its chairs inside a :class:`World`.

        :param world: Existing world to extend. A fresh world with a
            ``map`` root body is created when omitted.
        :param parent: The parent entity the table's own
            :attr:`position`/ :attr:`orientation` are expressed relative
            to. Defaults to the world's root when omitted, so standalone
            callers are unaffected.
        :return: The world containing the table and its chairs.
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
            Table.create_with_new_body_in_world(
                name=PrefixedName(name="table"),
                world=_world,
                world_root_T_self=table_pose,
                scale=Scale(x=self.scale.length, y=self.scale.width, z=self.scale.height),
            )

        mesh_matcher = _MeshSizeMatcher(candidates=self.source_ids or [])

        for i, chair in enumerate(self.chairs):
            if not self.source_ids:
                continue
            scene_dir, source_id = mesh_matcher.closest_match(chair.scale)
            chair.source_id = source_id
            chair.create_in_world(
                _world,
                scene_dir,
                parent=_parent,
                table_position=self.position,
                table_orientation=self.orientation,
            )

        return _world


@dataclass
class SceneGenerator(EGWithID):
    room: EGRoom
    """
    The room of the scene.

    Currently only one room is supported for simplicity.
    """

    mesh_to_object_mapping: dict[Path, EGObject] = field(default_factory=dict)
    """
    A mapping from the mesh directory path to the corresponding object in the
    scene.

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

        self.room.create_in_world(world, self.mesh_to_object_mapping, root)

        return world
