import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Self, Type, Dict, Any, Tuple, assert_never, Optional

import numpy as np

from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageWithTypeDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Room,
    RoomWithWallsAndDoors,
    Floor,
    Wall,
    Door,
    DoorWithType,
    Handle,
    Hinge,
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
from semantic_digital_twin.world_description.geometry import Mesh, Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
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
        mesh_to_object_mapping: Optional[Dict[Path, "EGObject"]],
        parent: KinematicStructureEntity,
        **kwargs,
    ) -> WorldEntity:
        """
        Create the object in the world by getting its geometry from the provided information.

        :param world: The world where the object is created.
        :param mesh_to_object_mapping: A mapping from mesh paths to object information.
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
    Scale in z
    """

    length: float
    """
    Scale in y
    """

    width: float
    """
    Scale in x
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            "height": self.height,
            "length": self.length,
            "width": self.width,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            height=data["height"],
            length=data["length"],
            width=data["width"],
        )


@dataclass
class EGPoint2D(EGBase):
    x: float
    y: float

    def to_json(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            x=data["x"],
            y=data["y"],
        )


@dataclass
class EGPosition(EGPoint2D):
    z: float

    def to_json(self) -> Dict[str, Any]:
        json = super().to_json()
        json.update({"z": self.z})
        return json

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:

        return cls(
            x=data["x"],
            y=data["y"],
            z=data["z"],
        )


@dataclass
class EGOrientation(EGPoint2D):
    z: float

    def as_roll_pitch_yaw_in_radians(self):
        conversion_factor = np.pi / 180
        return (
            self.x * conversion_factor,
            self.y * conversion_factor,
            self.z * conversion_factor,
        )

    def to_json(self) -> Dict[str, Any]:
        json = super().to_json()
        json.update({"z": self.z})
        return json

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            x=data["x"],
            y=data["y"],
            z=data["z"],
        )


# %%
@dataclass
class EGObject(EGWithID):
    room_id: str
    """
    The id of the room where the object is located.
    """

    place_id: str
    """
    The id of the object where the object is located/placed on/at, e.g. wall, floor, table.
    """

    object_type: str
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

    orientation: EGOrientation
    """
    The orientation of the object.
    """

    source_id: str
    """
    id of the object. This is used to identify the object in the dataset.
    """

    # children: List[Self] = field(default_factory=list)
    # """
    # List of the children of the object.
    # Children are objects that are placed inside, on or beneath the object.
    # """

    def to_json(self) -> Dict[str, Any]:
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
    def _from_json(cls, data: Dict[str, Any], **kwargs):
        return cls(
            id=data["id"],
            room_id=data["room_id"],
            object_type=data["type"],
            place_id=data["place_id"],
            position=EGPosition._from_json(data["position"], **kwargs),
            orientation=EGOrientation._from_json(data["rotation"], **kwargs),
            scale=EGSize._from_json(data["dimensions"], **kwargs),
            source_id=data["source_id"],
        )

    def create_in_world(
        self,
        world: World,
        mesh_path: Optional[Path],
        parent: KinematicStructureEntity,
        **kwargs,
    ):
        if mesh_path is None:
            raise ValueError(
                f"No mesh path found for object '{self.id}' (source_id='{self.source_id}'). "
                "Ensure the object is present in the mesh_to_object_mapping."
            )
        if not mesh_path.exists():
            raise ValueError(f"Directory {mesh_path} does not exist.")
        ply_file = mesh_path / "objects" / f"{self.source_id}.ply"
        texture_file = mesh_path / "objects" / f"{self.source_id}_texture.png"

        body = Body()
        body.name = PrefixedName(name=str(body.id), prefix=self.id)

        # Define the pose for the object in the world
        root_T_body = HomogeneousTransformationMatrix.from_xyz_rpy(
            self.position.x,
            self.position.y,
            self.position.z,
            *self.orientation.as_roll_pitch_yaw_in_radians(),
            reference_frame=parent,
            child_frame=body,
        )

        # Load the mesh and texture
        mesh = Mesh.from_ply_file(
            ply_file_path=str(ply_file),
            texture_file_path=str(texture_file),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(reference_frame=body),
        )

        # Create a Body with the loaded mesh as both visual and collision geometry
        visual = ShapeCollection([mesh], reference_frame=body)
        collision = ShapeCollection([mesh], reference_frame=body)
        body.visual = visual
        body.collision = collision

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
            # Add the body to the world
            world.add_body(body)
            world.add_connection(root_C_body)

        # create semantic annotation
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

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "id": self.id,
            "start_point": to_json(self.start_point),
            "end_point": to_json(self.end_point),
            "height": self.height,
            "thickness": self.thickness,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs):
        return cls(
            id=data["id"],
            start_point=EGPosition._from_json(data["start_point"], **kwargs),
            end_point=EGPosition._from_json(data["end_point"], **kwargs),
            height=data["height"],
            thickness=data["thickness"],
        )

    @property
    def wall_length_and_yaw(self) -> Tuple[float, float]:
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
    Position on wall w. r. t. its starting point as percentage of the wall length.
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

    def to_json(self) -> Dict[str, Any]:
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
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
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

        :param wall: The sage 10k wall that is referenced by `self.wall_id`.
        :param wall_annotation: The wall annotation created in `world` before this call.
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
            kwargs["wall_annotation"].add_aperture(annotation.entry_way)

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
            door.add_handle(handle)
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
            door.add_hinge(hinge)

        return hinge


@dataclass
class EGRoom(EGWithID):
    room_type: str
    """
    The type of the room.
    """

    # Currently only rectangular rooms, could use footprint: List[Tuple[float, float]] for L-Shaped rooms
    scale: EGSize
    """
    The scale of the room.
    """

    position: EGPosition
    """
    The position of the rooms lower left corner? in the scene.
    """
    # floor_material: str

    objects: List[EGObject] = field(default_factory=list)
    """
    List of the objects in the room.
    """

    walls: List[EGWall] = field(default_factory=list)
    """
    List of the walls in the room.
    """

    doors: List[EGDoor] = field(default_factory=list)
    """
    List of the doors in the room.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "id": self.id,
            "room_type": self.room_type,
            "scale": to_json(self.scale),
            "position": to_json(self.position),
            "objects": to_json(self.objects),
            "walls": to_json(self.walls),
            "doors": to_json(self.doors),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            id=data["id"],
            room_type=data["room_type"],
            scale=EGSize._from_json(data["dimensions"], **kwargs),
            position=EGPosition._from_json(data["position"], **kwargs),
            objects=[EGObject._from_json(o, **kwargs) for o in data["objects"]],
            walls=[EGWall._from_json(w, **kwargs) for w in data["walls"]],
            doors=[EGDoor._from_json(d, **kwargs) for d in data["doors"]],
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
        mesh_to_object_mapping: Optional[Dict[Path, "EGObject"]],
        parent: KinematicStructureEntity,
        **kwargs,
    ) -> WorldEntity:

        floor_annotation = self._create_floor(world, parent)
        walls_of_room = []
        doors_of_room = []

        for wall in self.walls:
            wall_annotation = wall.create_in_world(world, parent)
            walls_of_room.append(wall_annotation)
            doors_of_this_wall = [
                door for door in self.doors if door.wall_id == wall.id
            ]  # join doors on this wall

            # create doors
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

        object_to_mesh_path: Dict[str, Path] = (
            {obj.id: path for path, obj in mesh_to_object_mapping.items()}
            if mesh_to_object_mapping
            else {}
        )

        for obj in self.objects:
            mesh_path = object_to_mesh_path.get(obj.id)
            obj.create_in_world(world, mesh_path, parent=parent)

        return world.root


@dataclass
class SceneGenerator(EGWithID):
    room: EGRoom
    """
    The room of the scene.
    Currently only one room is supported for simplicity.
    """

    mesh_to_object_mapping: Dict[Path, "EGObject"] = field(default_factory=dict)
    """
    A mapping from the mesh directory path to the corresponding object in the scene.
    Used to resolve per-object mesh paths when creating the world.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "id": self.id,
            "room": to_json(self.room),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs):
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
