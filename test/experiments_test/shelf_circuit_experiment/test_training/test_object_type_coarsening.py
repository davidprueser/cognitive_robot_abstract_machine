from __future__ import annotations

from pathlib import Path

from experiments.shelf_generation_experiments.training.object_type_coarsening import (
    ObjectTypeCoarsening,
)
from experiments.shelf_generation_experiments.utils import MeshCandidate, ObjectType

from ..shelf_dataset import random_shelves


# %% fixtures
def _book_dominated_shelves():
    """
    :return: Shelves themed by books and bottles holding books, bottles and boxes, plus
        one shelf themed by a rare type holding a rare object.
    """
    shelves = random_shelves()
    rare_shelf = random_shelves(
        themes=(ObjectType.LAMP,), object_types=(ObjectType.LAMP,), shelves_per_theme=1
    )
    return shelves + rare_shelf


# %% counting
def test_the_most_frequent_object_types_and_themes_are_kept() -> None:
    coarsening = ObjectTypeCoarsening.from_shelves(
        _book_dominated_shelves(), keep_count=2
    )

    assert coarsening.frequent_theme_types == {ObjectType.BOOK, ObjectType.BOTTLE}
    assert len(coarsening.frequent_object_types) == 2
    assert ObjectType.LAMP not in coarsening.frequent_object_types


def test_placeable_types_are_kept_as_objects_and_themes_but_never_other() -> None:
    coarsening = ObjectTypeCoarsening(
        frequent_object_types={ObjectType.BOOK, ObjectType.BOX, ObjectType.OTHER},
        frequent_theme_types={ObjectType.BOOK, ObjectType.OTHER},
    )

    assert coarsening.placeable_object_types == {ObjectType.BOOK}


# %% coarsening shelves
def test_a_rare_theme_is_coarsened_on_the_shelf_and_every_layer() -> None:
    coarsening = ObjectTypeCoarsening(
        frequent_object_types={ObjectType.LAMP}, frequent_theme_types={ObjectType.BOOK}
    )
    [rare_shelf] = random_shelves(
        themes=(ObjectType.LAMP,), object_types=(ObjectType.LAMP,), shelves_per_theme=1
    )

    [coarsened] = coarsening.coarsen_shelves([rare_shelf])

    assert coarsened.theme_dominant_type is ObjectType.OTHER
    assert {layer.theme_dominant_type for layer in coarsened.layers} == {
        ObjectType.OTHER
    }
    assert {
        object_.object_type for layer in coarsened.layers for object_ in layer.objects
    } == {ObjectType.LAMP}


def test_a_rare_object_type_is_coarsened_and_a_frequent_one_kept() -> None:
    coarsening = ObjectTypeCoarsening(
        frequent_object_types={ObjectType.BOOK},
        frequent_theme_types={ObjectType.BOOK},
    )
    [shelf] = random_shelves(
        themes=(ObjectType.BOOK,),
        object_types=(ObjectType.BOOK, ObjectType.LAMP),
        shelves_per_theme=1,
    )

    [coarsened] = coarsening.coarsen_shelves([shelf])

    assert {
        object_.object_type for layer in coarsened.layers for object_ in layer.objects
    } == {ObjectType.BOOK, ObjectType.OTHER}
    assert coarsened.theme_dominant_type is ObjectType.BOOK


def test_coarsening_leaves_the_given_shelves_unchanged() -> None:
    coarsening = ObjectTypeCoarsening(
        frequent_object_types=set(), frequent_theme_types=set()
    )
    [shelf] = random_shelves(themes=(ObjectType.BOOK,), shelves_per_theme=1)

    coarsening.coarsen_shelves([shelf])

    assert shelf.theme_dominant_type is ObjectType.BOOK


# %% mesh candidates
def test_mesh_candidates_are_relabelled_like_the_objects() -> None:
    coarsening = ObjectTypeCoarsening(
        frequent_object_types={ObjectType.BOOK}, frequent_theme_types=set()
    )
    candidates = [
        MeshCandidate(Path("scene"), "book_mesh", ObjectType.BOOK),
        MeshCandidate(Path("scene"), "lamp_mesh", ObjectType.LAMP),
    ]

    coarsened = coarsening.coarsen_mesh_candidates(candidates)

    assert [candidate.object_type for candidate in coarsened] == [
        ObjectType.BOOK,
        ObjectType.OTHER,
    ]


def test_stored_types_of_a_shelf_are_its_sampled_types() -> None:
    coarsening = ObjectTypeCoarsening(
        frequent_object_types={ObjectType.BOOK, ObjectType.BOTTLE},
        frequent_theme_types=set(),
    )
    [shelf] = random_shelves(
        themes=(ObjectType.BOOK,),
        object_types=(ObjectType.BOOK, ObjectType.BOTTLE),
        shelves_per_theme=1,
    )

    assert coarsening.stored_object_types_of(shelf) == {
        ObjectType.BOOK,
        ObjectType.BOTTLE,
    }


def test_a_sampled_other_stands_for_every_type_not_kept() -> None:
    """
    Stored objects the classifier could not place are themselves of type
    :attr:`ObjectType.OTHER`, so that type is among those a sampled OTHER stands for.
    """
    coarsening = ObjectTypeCoarsening(
        frequent_object_types={ObjectType.BOOK}, frequent_theme_types=set()
    )
    [shelf] = random_shelves(
        themes=(ObjectType.BOOK,),
        object_types=(ObjectType.BOOK, ObjectType.OTHER),
        shelves_per_theme=1,
    )

    assert coarsening.stored_object_types_of(shelf) == set(ObjectType)
