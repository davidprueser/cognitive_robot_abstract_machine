from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


# %% colliding pairs
@dataclass
class CollidingPairs:
    """
    Which members of a group of objects, identified by their index, collide with each
    other.
    """

    pairs: set[frozenset[int]] = field(default_factory=set)
    """
    One set of two member indices per colliding pair.
    """

    def add(self, first_index: int, second_index: int) -> None:
        """
        Record that the members at *first_index* and *second_index* collide.

        :param first_index: Index of one member of the pair.
        :param second_index: Index of the other member of the pair.
        """
        self.pairs.add(frozenset((first_index, second_index)))

    def minimal_resample_set(self) -> set[int]:
        """
        A small set of member indices whose removal breaks every colliding pair.

        The member involved in the most remaining pairs is removed first, ties going to
        the higher index, so the result depends only on which members collide and not on
        the order the collisions were reported in.

        :return: The indices of the members to move.
        """
        remaining_pairs = set(self.pairs)
        indices_to_resample: set[int] = set()
        while remaining_pairs:
            involvement_counts = Counter(
                index for pair in remaining_pairs for index in pair
            )
            most_colliding_index = min(
                involvement_counts,
                key=lambda index: (-involvement_counts[index], -index),
            )
            indices_to_resample.add(most_colliding_index)
            remaining_pairs = {
                pair for pair in remaining_pairs if most_colliding_index not in pair
            }
        return indices_to_resample
