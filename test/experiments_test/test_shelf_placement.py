# from __future__ import annotations
#
# import math
# import dataclasses
# from dataclasses import dataclass
#
# import pytest
# from random_events.interval import closed
# from random_events.product_algebra import SimpleEvent
# from random_events.variable import Continuous
#
# import experiments.orm.ormatic_interface  # noqa: F401  registers ORM mappers
# from experiments.scene_generation_experiments.shelf_placement import (
#     LayerVariable,
#     ObjectSlotVariable,
#     PlacementRefusal,
#     _free_positions,
#     _layer_and_object_evidence,
#     _neighbour_evidence,
#     _occupied_footprints,
#     _placement_query,
#     mode_query,
# )
# from krrood.parametrization.model_registries import RelationalCircuitRegistry
# from probabilistic_model.probabilistic_circuit.rx.helper import (
#     uniform_measure_of_simple_event,
# )
# from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
#     ProbabilisticCircuit,
# )
# from krrood.parametrization.parameterizer import UnderspecifiedParameters
# from krrood.ormatic.data_access_objects.helper import to_dao
# from probabilistic_model.probabilistic_circuit.relational.rspn import (
#     RelationalProbabilisticCircuit,
# )
# from semantic_digital_twin.scene_generation.scene_schema import (
#     EGObject2D,
#     EGPoint2D,
#     EGRotation,
#     EGScale,
#     EGShelf,
#     EGShelfLayer,
#     ObjectType,
#     ShelfLayerGeometry,
# )
# from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
# from semantic_digital_twin.world_description.geometry import BoundingBox
#
# _SHELF_SCALE = EGScale(height=1.2, length=0.4, width=0.8)
# """
# Dimensions of every shelf in this module, so a query can pin what was fitted.
# """
#
# _BOOK_SCALE = EGScale(width=0.12, length=0.04, height=0.2)
# """
# Size of the book that is fitted and later placed; the same value in both, since a size
# the fit never saw has no density to compare layers by.
# """
#
# _LOWER_RELATIVE_HEIGHT = 1 / 3
# """
# Relative height of the lower of two evenly spaced layers.
#
# Slabs are spread evenly over the corpus regardless of the heights a layer was
# recorded at, so a two-layer shelf always puts its layers here and at
# :data:`_UPPER_RELATIVE_HEIGHT` -- and a fit that used other values would leave the
# spawned shelf's own heights with no density.
# """
#
# _UPPER_RELATIVE_HEIGHT = 2 / 3
# """
# Relative height of the upper of two evenly spaced layers.
# """
#
# _NEIGHBOUR_SCALE = EGScale(width=0.09, length=0.09, height=0.175)
# """
# Size of the objects a layer already holds, roughly a jar.
# """
#
# _LOW_SHELF_SCALE = EGScale(
#     height=_SHELF_SCALE.height / 2, length=_SHELF_SCALE.length, width=_SHELF_SCALE.width
# )
# """
# Dimensions of the shelves that make one absolute layer height common.
#
# Half the height of :data:`_SHELF_SCALE`, so its upper layer sits at exactly the height
# the taller shelf's lower layer does.
# """
#
# # %% fixtures
#
#
# def _object_of(
#     object_type: ObjectType, scale: EGScale, index: int, theme: ObjectType
# ) -> EGObject2D:
#     return EGObject2D(
#         id=f"{object_type.value}_{index}",
#         room_id="room_1",
#         place_id="shelf_1",
#         object_type=object_type,
#         scale=scale,
#         position=EGPoint2D(x=0.0, y=0.0),
#         orientation=EGRotation(x=0.0, y=0.0, z=0.0),
#         source_id=f"{object_type.value}_source",
#     )
#
#
# def _layer_holding(
#     scale: EGScale, index: int, relative_height: float, shelf_height: float
# ) -> EGShelfLayer:
#     return EGShelfLayer(
#         objects=[_object_of(ObjectType.BOOK, scale, index, ObjectType.BOOK)],
#         theme_dominant_type=ObjectType.BOOK,
#         height_above_shelf_base=relative_height * shelf_height,
#         relative_height=relative_height,
#         vertical_clearance=0.3,
#     )
#
#
# def _two_layer_shelf(scale: EGScale, index: int) -> EGShelf:
#     """
#     A shelf whose two layers sit where an evenly spaced spawn would put them.
#     """
#     return EGShelf(
#         scale=scale,
#         layers=[
#             _layer_holding(_BOOK_SCALE, index, height, scale.height)
#             for height in (_LOWER_RELATIVE_HEIGHT, _UPPER_RELATIVE_HEIGHT)
#         ],
#         theme_dominant_type=ObjectType.BOOK,
#     )
#
#
# @pytest.fixture
# def low_shelf_model() -> RelationalProbabilisticCircuit:
#     """
#     A model fitted mostly on low shelves, so a layer around ``_LOW_SHELF_SCALE.height *
#     _UPPER_RELATIVE_HEIGHT`` above a shelf's base is a height it has seen far more often
#     than any other.
#     """
#     shelves = [_two_layer_shelf(_LOW_SHELF_SCALE, index) for index in range(8)]
#     shelves += [_two_layer_shelf(_SHELF_SCALE, index) for index in range(8, 10)]
#     return RelationalProbabilisticCircuit(EGShelf, min_samples_per_leaf=0.1).fit(
#         [to_dao(shelf) for shelf in shelves]
#     )
#
#
# @pytest.fixture
# def empty_two_layer_shelf() -> EGShelf:
#     """
#     A shelf of the fitted size whose layers spawn without any bodies on them, so every
#     position on them is free.
#
#     The upper layer comes first, so a placement that ignored what the model says and
#     simply took the first layer it could would be the wrong answer rather than
#     accidentally the right one.
#     """
#     return EGShelf(
#         scale=_SHELF_SCALE,
#         layers=[
#             _layer_holding(_BOOK_SCALE, 0, height, _SHELF_SCALE.height)
#             for height in (_UPPER_RELATIVE_HEIGHT, _LOWER_RELATIVE_HEIGHT)
#         ],
#         theme_dominant_type=ObjectType.BOOK,
#         source_ids=[],
#     )
#
#
# def _held_book(scale: EGScale = _BOOK_SCALE) -> EGObject2D:
#     return _object_of(ObjectType.BOOK, scale, 0, ObjectType.BOOK)
#
#
# def _footprint(min_x: float, min_y: float, max_x: float, max_y: float) -> BoundingBox:
#     return BoundingBox(
#         min_x, min_y, 0.0, max_x, max_y, 0.1, HomogeneousTransformationMatrix()
#     )
#
#
# # %% free positions on a layer
#
#
# def _stands_within_layer(
#     shelf: EGShelf, scale: EGScale, x: float, y: float, yaw_degrees: float
# ) -> bool:
#     """
#     Whether an object of *scale* turned to *yaw_degrees* lies inside the layer when its
#     centre is at ``(x, y)``.
#     """
#     angle = math.radians(yaw_degrees)
#     depth = scale.length * abs(math.cos(angle)) + scale.width * abs(math.sin(angle))
#     face = scale.length * abs(math.sin(angle)) + scale.width * abs(math.cos(angle))
#     return (
#         abs(x) + depth / 2 <= shelf.scale.length / 2 + 1e-9
#         and abs(y) + face / 2 <= shelf.scale.width / 2 + 1e-9
#     )
#
#
# def test_every_free_pose_keeps_the_object_inside_the_layer() -> None:
#     """
#     The region exists to stop an object hanging over an edge, so no pose it admits may
#     put any part of the object outside the layer -- at the yaw that pose carries.
#     """
#     position_x, position_y, yaw = Continuous("x"), Continuous("y"), Continuous("yaw")
#     shelf = EGShelf(scale=_SHELF_SCALE, layers=[], theme_dominant_type=ObjectType.BOOK)
#
#     free = _free_positions(shelf, [], _BOOK_SCALE, position_x, position_y, yaw)
#
#     overhanging = [
#         (x, y, yaw_degrees)
#         for yaw_degrees in range(-180, 180, 7)
#         for x in (-0.19, -0.1, 0.0, 0.1, 0.19)
#         for y in (-0.42, -0.2, 0.0, 0.2, 0.42)
#         if free.contains((x, y, yaw_degrees))
#         and not _stands_within_layer(shelf, _BOOK_SCALE, x, y, yaw_degrees)
#     ]
#     assert overhanging == []
#
#
# def test_the_middle_of_an_empty_layer_is_free() -> None:
#     """
#     A region that admitted nothing would satisfy every other rule here vacuously.
#     """
#     position_x, position_y, yaw = Continuous("x"), Continuous("y"), Continuous("yaw")
#     shelf = EGShelf(scale=_SHELF_SCALE, layers=[], theme_dominant_type=ObjectType.BOOK)
#
#     free = _free_positions(shelf, [], _BOOK_SCALE, position_x, position_y, yaw)
#
#     assert free.contains((0.0, 0.0, 90.0))
#     assert free.contains((0.0, 0.0, 0.0))
#
#
# def test_an_object_longer_than_the_layer_is_deep_can_lie_along_it() -> None:
#     """
#     A flat book is wider than a shelf is deep along its diagonal but fits easily lying
#     along the shelf, so a bound that had to hold at every yaw would turn it away from a
#     layer it plainly fits.
#     """
#     position_x, position_y, yaw = Continuous("x"), Continuous("y"), Continuous("yaw")
#     shelf = EGShelf(scale=_SHELF_SCALE, layers=[], theme_dominant_type=ObjectType.BOOK)
#     # Depth 0.25 and face 0.37 on a layer 0.40 deep: the diagonal, 0.45, does not fit.
#     flat_book = EGScale(width=0.37, length=0.25, height=0.06)
#     assert math.hypot(flat_book.length, flat_book.width) > _SHELF_SCALE.length
#
#     free = _free_positions(shelf, [], flat_book, position_x, position_y, yaw)
#
#     assert not free.is_empty()
#     assert free.contains((0.0, 0.0, 0.0))
#
#
# def test_a_position_the_object_would_reach_into_its_neighbour_from_is_not_free() -> (
#     None
# ):
#     """
#     Positions are truncated before the yaw is drawn, so a neighbour is kept clear by the
#     object's whole reach rather than by the footprint it happens to have at yaw zero.
#     """
#     position_x, position_y, yaw = Continuous("x"), Continuous("y"), Continuous("yaw")
#     shelf = EGShelf(scale=_SHELF_SCALE, layers=[], theme_dominant_type=ObjectType.BOOK)
#     reach = math.hypot(_BOOK_SCALE.length, _BOOK_SCALE.width) / 2
#     neighbour = _footprint(-0.05, -0.05, 0.05, 0.05)
#
#     free = _free_positions(shelf, [neighbour], _BOOK_SCALE, position_x, position_y, yaw)
#
#     # Along y, where the shelf is wide enough for both the neighbour's reach and a
#     # position clear of it.
#     assert not free.contains((0.0, 0.05 + reach / 2))
#     assert free.contains((0.0, 0.05 + reach * 1.5))
#
#
# def test_a_layer_covered_by_what_stands_on_it_has_no_free_position() -> None:
#     """
#     A layer with nowhere left to stand must be skipped rather than answered with a pose
#     on top of something.
#     """
#     position_x, position_y, yaw = Continuous("x"), Continuous("y"), Continuous("yaw")
#     shelf = EGShelf(scale=_SHELF_SCALE, layers=[], theme_dominant_type=ObjectType.BOOK)
#     covering_everything = _footprint(-1.0, -1.0, 1.0, 1.0)
#
#     free = _free_positions(
#         shelf, [covering_everything], _BOOK_SCALE, position_x, position_y, yaw
#     )
#
#     assert free.is_empty()
#
#
# # %% asking the model
#
#
# def test_the_placement_is_a_pose_on_one_of_the_shelfs_own_layers(
#     low_shelf_model: RelationalProbabilisticCircuit,
#     empty_two_layer_shelf: EGShelf,
# ) -> None:
#     """
#     The answer has to be somewhere the object can actually be put down: on a layer of
#     this shelf, within its footprint, resting on that layer's slab.
#     """
#     spawned = empty_two_layer_shelf.spawn_in_world()
#     geometries = empty_two_layer_shelf.layer_geometries()
#
#     placement = mode_query(spawned, low_shelf_model, _held_book())
#
#     geometry = geometries[placement.layer_index]
#     assert placement.layer_index in range(len(empty_two_layer_shelf.layers))
#     assert abs(placement.placed_object.position.x) <= _SHELF_SCALE.length / 2
#     assert abs(placement.placed_object.position.y) <= _SHELF_SCALE.width / 2
#     assert placement.pose.reference_frame is spawned.corpus
#     assert float(placement.pose.z) == pytest.approx(geometry.slab_top_height)
#
#
# def test_the_layer_at_a_height_the_model_knows_wins(
#     low_shelf_model: RelationalProbabilisticCircuit,
#     empty_two_layer_shelf: EGShelf,
# ) -> None:
#     """
#     Which layer to use is the model's answer, not the caller's: the fit saw layers
#     around one height far more often than any other, so the layer sitting there wins
#     over the one that does not.
#
#     A layer's own height reaches the circuit only when it is conditioned on under the
#     name the circuit gives it, so this also pins down that the evidence arrives at all.
#     """
#     spawned = empty_two_layer_shelf.spawn_in_world()
#     geometries = empty_two_layer_shelf.layer_geometries()
#     common_height = _LOW_SHELF_SCALE.height * _UPPER_RELATIVE_HEIGHT
#
#     placement = mode_query(spawned, low_shelf_model, _held_book())
#
#     assert geometries[placement.layer_index].height_above_shelf_base == pytest.approx(
#         common_height
#     )
#
#
# # %% what a layer is scored on
#
#
# def _placement_parameters(occupied_count: int) -> tuple[UnderspecifiedParameters, str]:
#     parameters = UnderspecifiedParameters(
#         _placement_query(ObjectType.BOOK, _held_book(), occupied_count)
#     )
#     return parameters, f"EGShelfLayer.objects[{occupied_count}]."
#
#
# def test_what_already_stands_on_a_layer_is_not_part_of_its_score() -> None:
#     """
#     Every object slot pins the same upright orientation and theme, each worth the same
#     fixed amount of log density, so counting them in would mark a layer down for holding
#     more -- which says nothing about where the held object belongs.
#     """
#     parameters, held_slot_prefix = _placement_parameters(occupied_count=3)
#
#     scored = _layer_and_object_evidence(
#         parameters,
#         {variable: variable for variable in LayerVariable},
#         ShelfLayerGeometry(
#             height_above_shelf_base=0.4,
#             relative_height=_LOWER_RELATIVE_HEIGHT,
#             slab_top_height=0.0,
#             maximum_object_extents=_BOOK_SCALE,
#         ),
#         ObjectType.BOOK,
#         held_slot_prefix,
#     )
#
#     slot_assignments = [
#         variable.name
#         for variable in scored
#         if variable.name.startswith("EGShelfLayer.objects[")
#     ]
#     assert slot_assignments
#     assert all(name.startswith(held_slot_prefix) for name in slot_assignments)
#
#
# def test_the_objects_already_on_a_layer_are_conditioned_away_first() -> None:
#     """
#     Their pins are still conditioned on, only separately, so the layer the score is
#     taken over is the one that really holds them.
#     """
#     parameters, held_slot_prefix = _placement_parameters(occupied_count=3)
#
#     discarded = _neighbour_evidence(parameters, held_slot_prefix)
#
#     assert discarded
#     assert all(
#         variable.name.startswith("EGShelfLayer.objects[")
#         and not variable.name.startswith(held_slot_prefix)
#         for variable in discarded
#     )
#
#
# def test_an_object_wider_than_the_layer_has_nowhere_to_stand() -> None:
#     """
#     Pulling the layer's bounds in by more than half its own width leaves a single point
#     dead centre, which an object that size overhangs on every side.
#     """
#     position_x, position_y, yaw = Continuous("x"), Continuous("y"), Continuous("yaw")
#     shelf = EGShelf(scale=_SHELF_SCALE, layers=[], theme_dominant_type=ObjectType.BOOK)
#     too_wide = EGScale(
#         width=_SHELF_SCALE.width * 2, length=_SHELF_SCALE.length, height=0.1
#     )
#
#     free = _free_positions(shelf, [], too_wide, position_x, position_y, yaw)
#
#     assert free.is_empty()
#
#
# # %% picking the densest pose
#
#
# @dataclass
# class _PoseModelWhoseModeCannotBeRead:
#     """
#     A pose distribution that answers every question the real one does, except that its
#     mode cannot be read off -- which is what a truncated circuit sometimes looks like.
#     """
#
#     circuit: ProbabilisticCircuit
#     """
#     The distribution every call is passed on to.
#     """
#
#     @property
#     def variables(self):
#         return self.circuit.variables
#
#     def is_deterministic(self) -> bool:
#         return False
#
#     def sample(self, amount: int):
#         return self.circuit.sample(amount)
#
#     def log_likelihood(self, points):
#         return self.circuit.log_likelihood(points)
#
#
# def _pose_distributions(
#     model: RelationalProbabilisticCircuit, shelf: EGShelf, layer_index: int
# ) -> tuple[ProbabilisticCircuit, ProbabilisticCircuit]:
#     """
#     The pose distribution for a book on one of *shelf*'s layers, untruncated and
#     restricted to the free space, as :func:`_layer_placement` builds them.
#     """
#     geometry = shelf.layer_geometries()[layer_index]
#     theme = shelf.theme_dominant_type
#     parameters = UnderspecifiedParameters(_placement_query(theme, _held_book(), 0))
#     layers = model.exchangeable_distribution_templates["layers"].template_distribution
#     grounded = RelationalCircuitRegistry(
#         relational_probabilistic_circuit=layers
#     ).get_model(parameters)
#     variables = {variable.name: variable for variable in grounded.variables}
#     prefix = "EGShelfLayer.objects[0]."
#     without, _ = grounded.log_conditional(_neighbour_evidence(parameters, prefix))
#     conditioned, _ = without.log_conditional(
#         _layer_and_object_evidence(parameters, variables, geometry, theme, prefix)
#     )
#     pose_variables = [variables[prefix + name] for name in ObjectSlotVariable]
#     pose_model = conditioned.marginal(pose_variables)
#     free = _free_positions(shelf, [], _held_book().scale, *pose_variables)
#     free.fill_missing_variables(pose_model.variables)
#     truncated, _ = pose_model.log_truncated(free)
#     return pose_model, truncated
#
#
# def test_the_chosen_pose_stands_inside_the_layer(
#     low_shelf_model: RelationalProbabilisticCircuit,
#     empty_two_layer_shelf: EGShelf,
# ) -> None:
#     """
#     The pose handed back is what a place action aims at, so the object must actually be
#     on the layer when it gets there.
#     """
#     spawned = empty_two_layer_shelf.spawn_in_world()
#
#     placement = mode_query(spawned, low_shelf_model, _held_book())
#
#     placed = placement.placed_object
#     assert _stands_within_layer(
#         empty_two_layer_shelf,
#         placed.scale,
#         placed.position.x,
#         placed.position.y,
#         placed.orientation.z,
#     )
#
#
# # %% clearing what is already there
#
#
# def _layer_holding_objects_at(
#     face_positions: tuple[float, ...],
#     relative_height: float,
#     shelf_height: float,
#     scale: EGScale = _NEIGHBOUR_SCALE,
# ) -> EGShelfLayer:
#     """
#     A layer holding one object at each of *face_positions* along the shelf's face.
#     """
#     return EGShelfLayer(
#         objects=[
#             dataclasses.replace(
#                 _object_of(ObjectType.BOOK, scale, index, ObjectType.BOOK),
#                 position=EGPoint2D(x=0.0, y=face_position),
#             )
#             for index, face_position in enumerate(face_positions)
#         ],
#         theme_dominant_type=ObjectType.BOOK,
#         height_above_shelf_base=relative_height * shelf_height,
#         relative_height=relative_height,
#         vertical_clearance=0.3,
#     )
#
#
# _CROWDED_FACE_POSITIONS = (0.0, 0.0, 0.30)
# """
# Where the objects of a crowded layer stand along the shelf's face.
#
# Two of the three sit at the middle, so that is where the fit puts an object most often,
# and a placement ignoring what already stands there would land exactly on the neighbour
# :func:`occupied_two_layer_shelf` parks in the middle.
#
# Three objects is also what makes the query answerable at all: the count reaches the
# model through the layer's aggregation statistics, and one the fit never saw is turned
# down before the free space is ever consulted.
# """
#
#
# @pytest.fixture
# def crowded_shelf_model() -> RelationalProbabilisticCircuit:
#     """
#     A model fitted on layers holding three books, one of them at the middle of the face.
#     """
#     shelves = [
#         EGShelf(
#             scale=_SHELF_SCALE,
#             layers=[
#                 _layer_holding_objects_at(
#                     _CROWDED_FACE_POSITIONS, height, _SHELF_SCALE.height, _BOOK_SCALE
#                 )
#                 for height in (_LOWER_RELATIVE_HEIGHT, _UPPER_RELATIVE_HEIGHT)
#             ],
#             theme_dominant_type=ObjectType.BOOK,
#         )
#         for _ in range(10)
#     ]
#     return RelationalProbabilisticCircuit(EGShelf, min_samples_per_leaf=0.1).fit(
#         [to_dao(shelf) for shelf in shelves]
#     )
#
#
# @pytest.fixture
# def occupied_two_layer_shelf() -> EGShelf:
#     """
#     A shelf whose every layer already holds two objects, one of them standing exactly
#     where the fit puts a lone object.
#
#     They spawn as plain boxes, which is enough for the placement to read their
#     footprints out of the world. Blocking the middle is the point: every object in the
#     fit sits at the origin, so a placement that ignored what was already there would
#     land on top of that neighbour.
#     """
#     return EGShelf(
#         scale=_SHELF_SCALE,
#         layers=[
#             _layer_holding_objects_at((0.0, 0.30), height, _SHELF_SCALE.height)
#             for height in (_UPPER_RELATIVE_HEIGHT, _LOWER_RELATIVE_HEIGHT)
#         ],
#         theme_dominant_type=ObjectType.BOOK,
#         source_ids=[],
#     )
#
#
# def _footprints_overlap(placed: EGObject2D, taken: BoundingBox) -> bool:
#     """
#     Whether *placed*, turned to its own yaw, shares any ground with *taken*.
#
#     The placed object is measured by the box its turned footprint fits inside, which is
#     never smaller than the footprint itself -- so no overlap here means no overlap in
#     truth.
#     """
#     angle = math.radians(placed.orientation.z)
#     depth = placed.scale.length * abs(math.cos(angle)) + placed.scale.width * abs(
#         math.sin(angle)
#     )
#     face = placed.scale.length * abs(math.sin(angle)) + placed.scale.width * abs(
#         math.cos(angle)
#     )
#     taken_x, taken_y = taken.x_interval, taken.y_interval
#     return (
#         abs(placed.position.x - (taken_x.lower + taken_x.upper) / 2)
#         < (depth + (taken_x.upper - taken_x.lower)) / 2
#         and abs(placed.position.y - (taken_y.lower + taken_y.upper) / 2)
#         < (face + (taken_y.upper - taken_y.lower)) / 2
#     )
#
#
# def test_the_chosen_pose_clears_what_already_stands_on_the_layer(
#     crowded_shelf_model: RelationalProbabilisticCircuit,
#     occupied_two_layer_shelf: EGShelf,
# ) -> None:
#     """
#     Keeping clear of the layer's other objects is the one thing the model cannot do --
#     its object slots are independent, so conditioning on them changes nothing -- and it
#     falls to the free region instead.
#
#     A pose landing on a neighbour would place the object inside it.
#     """
#     spawned = occupied_two_layer_shelf.spawn_in_world(
#         placeholders_for_missing_meshes=True
#     )
#
#     placement = mode_query(spawned, crowded_shelf_model, _held_book())
#
#     standing_there = _occupied_footprints(spawned, placement.layer_index)
#     assert standing_there
#     assert [
#         taken
#         for taken in standing_there
#         if _footprints_overlap(placement.placed_object, taken)
#     ] == []
