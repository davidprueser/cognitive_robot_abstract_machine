from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from semantic_digital_twin.scene_generation.scene_schema import RoomType


@dataclass(frozen=True)
class RoomTypeClassifier:
    """
    Maps the free-form, inconsistently spelled ``room_type`` strings found in
    the raw sage10k dataset (e.g. ``"grocery store"``, ``"grocery_store"``,
    ``"Baroque warehouse"``) onto the generalized :class:`RoomType` categories.

    Matching is a case-insensitive, ordered keyword lookup: the raw string is
    tested against each category's keywords in turn, and the first category with
    a matching keyword wins. Compound names are resolved by placing the more
    specific category first, so ``"warehouse office"`` classifies as a warehouse
    and ``"dentist_office"`` as an examination room rather than both collapsing
    onto :attr:`RoomType.OFFICE`. This is a best-effort heuristic, not a
    guaranteed-correct classification.
    """

    _KEYWORDS_BY_TYPE: ClassVar[tuple[tuple[RoomType, tuple[str, ...]], ...]] = (
        # -- Medical (before the generic "office"/"room" they contain) ---------
        (RoomType.OPERATING_ROOM, ("operating",)),
        (RoomType.EXAMINATION_ROOM, ("examination", "treatment room", "dentist", "dental")),
        (RoomType.PATIENT_ROOM, ("patient", "hospital")),
        # -- Categories that name the whole building, which outranks the room
        # it contains: "warehouse office" is a warehouse, "greenhouse lounge" a
        # greenhouse. "prison" is checked before "cellar" so neither keyword
        # swallows the other. ---------------------------------------------
        (RoomType.PRISON_CELL, ("prison",)),
        (RoomType.WINE_CELLAR, ("wine cellar", "cellar")),
        (RoomType.GREENHOUSE, ("greenhouse", "conservatory")),
        (RoomType.WAREHOUSE, ("warehouse", "storage")),
        # -- Checked before retail because "workshop" contains "shop" ---------
        (RoomType.WORKSHOP, ("workshop", "workspace", "workstation", "work area")),
        # -- Retail, most specific first --------------------------------------
        (RoomType.GROCERY_STORE, ("grocery", "supermarket", "delicatessen")),
        (RoomType.CLOTHING_STORE, ("clothing", "apparel", "shoe store", "shoe_store")),
        (RoomType.BAKERY, ("bakery",)),
        (RoomType.STORE, ("store", "shop", "showroom", "sales floor", "kiosk",
                          "customer area", "service counter")),
        # -- Food and drink. "bar" is checked after the building categories so
        # "Baroque warehouse" stays a warehouse. -----------------------------
        (RoomType.BAR, ("bar", "pub", "tavern")),
        (RoomType.RESTAURANT, ("restaurant", "diner", "buffet", "fast food", "fast_food",
                               "dining area", "dining_area", "cafeteria", "cafe")),
        (RoomType.CASINO, ("casino",)),
        (RoomType.MUSEUM, ("museum", "gallery", "exhibition")),
        (RoomType.HAIR_SALON, ("salon", "barber")),
        (RoomType.GYM, ("gym", "fitness")),
        (RoomType.LOCKER_ROOM, ("locker",)),
        (RoomType.MEDITATION_ROOM, ("meditation",)),
        (RoomType.GAME_ROOM, ("game", "gaming", "arcade")),
        # -- Education and work -----------------------------------------------
        (RoomType.CLASSROOM, ("classroom", "kindergarten", "lecture")),
        (RoomType.COMPUTER_LAB, ("computer lab", "computer_lab", "lab")),
        (RoomType.LIBRARY, ("library",)),
        (RoomType.CONFERENCE_ROOM, ("conference", "meeting room", "boardroom")),
        (RoomType.STUDIO, ("studio",)),
        (RoomType.OFFICE, ("office", "administrative", "staff room", "staff_room")),
        # -- Circulation and reception. "hall" is safe here because every
        # compound using it ("exhibition hall", "reception hall", "storage
        # hall") is claimed by an earlier category. -------------------------
        (RoomType.LOBBY, ("lobby", "reception", "waiting", "foyer")),
        (RoomType.CORRIDOR, ("corridor", "hallway", "hall", "entryway", "entrance")),
        # -- Residential --------------------------------------------------------
        (RoomType.NURSERY, ("nursery",)),
        (RoomType.BEDROOM, ("bedroom", "children")),
        (RoomType.KITCHEN, ("kitchen",)),
        (RoomType.BATHROOM, ("bathroom", "restroom", "washroom", "toilet")),
        (RoomType.DINING_ROOM, ("dining",)),
        (RoomType.LIVING_ROOM, ("living", "lounge", "family room", "sitting room")),
        (RoomType.PANTRY, ("pantry",)),
        (RoomType.CLOSET, ("closet", "wardrobe")),
        (RoomType.LAUNDRY_ROOM, ("laundromat", "laundry")),
        (RoomType.GARAGE, ("garage",)),
    )

    def classify(self, raw_type: str) -> RoomType:
        """
        Return the :class:`RoomType` category whose keywords best match
        *raw_type*.

        :param raw_type: A raw ``room_type`` string from the sage10k dataset
            (e.g. ``"restaurant_dining_area"``).
        :return: The best-matching generalized category, or
            :attr:`RoomType.OTHER` if no keyword matches.
        """
        normalized = raw_type.strip().lower()
        for room_type, keywords in self._KEYWORDS_BY_TYPE:
            if any(keyword in normalized for keyword in keywords):
                return room_type
        return RoomType.OTHER
