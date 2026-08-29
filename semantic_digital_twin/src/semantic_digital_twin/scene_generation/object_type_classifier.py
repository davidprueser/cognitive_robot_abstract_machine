from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from semantic_digital_twin.scene_generation.scene_schema import ObjectType


@dataclass(frozen=True)
class ObjectTypeClassifier:
    """
    Maps the free-form, near-instance-specific ``object_type`` strings found in
    the raw sage10k dataset (e.g. ``"book2"``, ``"bookchair8eba7fdc"``) onto
    the generalized :class:`ObjectType` categories.

    Matching is a case-insensitive, ordered keyword lookup: the raw
    string is tested against each category's keywords in turn, and the
    first category with a matching keyword wins. Furniture/surface
    categories (shelf, table, desk, ...) are checked before small-item
    categories, since the dataset frequently names an item together with
    the furniture it sits on (e.g. ``"bookshelf"``, ``"candletable"``)
    and the furniture is usually the more useful category for scene-
    layout purposes. This is a best-effort heuristic, not a guaranteed-
    correct classification -- raw strings that combine two plausible
    categories in an unusual order may be mapped to the "wrong" one.
    """

    _KEYWORDS_BY_TYPE: ClassVar[tuple[tuple[ObjectType, tuple[str, ...]], ...]] = (
        # -- Furniture -----------------------------------------------------
        (ObjectType.WORKBENCH, ("workbench",)),
        (ObjectType.DISPLAYCASE, ("displaycase", "showcase")),
        (ObjectType.WARDROBE, ("wardrobe", "closet")),
        (ObjectType.DRESSER, ("dresser",)),
        (ObjectType.LOCKER, ("locker",)),
        (ObjectType.PANTRY, ("pantry",)),
        (ObjectType.VANITY, ("vanity",)),
        (ObjectType.NIGHTSTAND, ("nightstand",)),
        (ObjectType.SIDEBOARD, ("sideboard", "console", "credenza")),
        (ObjectType.SHELF, ("shelf", "shelv", "rack", "bookcase")),
        (ObjectType.CABINET, ("cabinet",)),
        (ObjectType.DESK, ("desk",)),
        (ObjectType.COUNTER, ("counter", "countertop")),
        (ObjectType.SOFA, ("sofa", "couch")),
        (ObjectType.BENCH, ("bench",)),
        (ObjectType.BED, ("bed", "crib")),
        (ObjectType.CHAIR, ("chair", "stool", "armchair", "ottoman", "pouf", "barstool")),
        (ObjectType.TABLE, ("table", "island")),
        (ObjectType.CART, ("cart", "trolley")),
        (ObjectType.CRATE, ("crate", "pallet")),
        (ObjectType.TOOLBOX, ("toolbox",)),
        (ObjectType.PEDESTAL, ("pedestal", "podium", "plinth")),
        (ObjectType.STAND, ("stand", "holder", "hanger", "easel", "coatrack", "clothingrack")),
        # -- Plants (checked early: "pot" and "table" are common substrings of
        # "pottedplant"/"planttable"-style compounds, and the plant is the more
        # useful category for those) --------------------------------------
        (ObjectType.PLANT, ("plant", "succulent", "fern", "cactus", "ficus", "orchid", "palm",
                             "bamboo", "flower", "tree")),
        # -- Kitchen / dining ------------------------------------------------
        (ObjectType.CUTTING_BOARD, ("cuttingboard", "cutting_board")),
        (ObjectType.DISHWASHER, ("dishwasher",)),
        (ObjectType.REFRIGERATOR, ("fridge", "refrigerator", "freezer")),
        (ObjectType.SINK, ("sink",)),
        (ObjectType.OVEN, ("oven", "stove")),
        (ObjectType.MICROWAVE, ("microwave",)),
        (ObjectType.SMALL_APPLIANCE, ("toaster", "coffeemaker", "kettle", "blender")),
        (ObjectType.DISPENSER, ("dispenser",)),
        (ObjectType.CUTLERY, ("cutlery", "fork", "spoon", "spatula", "rollingpin")),
        (ObjectType.KNIFE, ("knife",)),
        (ObjectType.CUP, ("cup", "mug", "tumbler", "teacup")),
        (ObjectType.GLASS, ("glass", "wineglass")),
        (ObjectType.PLATE, ("plate",)),
        (ObjectType.BOWL, ("bowl",)),
        (ObjectType.BOTTLE, ("bottle",)),
        (ObjectType.JAR, ("jar", "shaker", "spicejar")),
        (ObjectType.UTENSIL, ("utensil",)),
        (ObjectType.POT, ("pot", "peppergrinder")),
        (ObjectType.TRAY, ("tray",)),
        # -- Lighting --------------------------------------------------------
        (ObjectType.CHANDELIER, ("chandelier",)),
        (ObjectType.NEON_SIGN, ("neon",)),
        (ObjectType.CANDLE, ("candle", "candelabra", "candlestick", "lantern")),
        (ObjectType.LAMP, ("lamp",)),
        (ObjectType.LIGHT_FIXTURE, ("light", "sconce", "fixture", "pendant", "ledstrip", "lightstrip")),
        # -- Electronics (checked before decor/art: "printer" and
        # "smartphone" would otherwise match ART's "print"/"art" substrings)
        # ---------------------------------------------------------------
        (ObjectType.TELEVISION, ("tv", "television")),
        (ObjectType.PROJECTOR, ("projector",)),
        (ObjectType.COMPUTER, ("computer", "laptop")),
        (ObjectType.KEYBOARD, ("keyboard",)),
        (ObjectType.MOUSE, ("mouse",)),
        (ObjectType.MONITOR, ("monitor", "screen")),
        (ObjectType.CAMERA, ("camera",)),
        (ObjectType.SPEAKER, ("speaker",)),
        (ObjectType.PHONE, ("phone", "smartphone")),
        (ObjectType.PRINTER, ("printer",)),
        (ObjectType.REMOTE_CONTROL, ("remote", "controller")),
        # -- Decor / art -------------------------------------------------------
        (ObjectType.MIRROR, ("mirror",)),
        (ObjectType.CLOCK, ("clock",)),
        (ObjectType.SCULPTURE, ("sculpture", "figurine", "statue", "bust", "mannequin")),
        (ObjectType.VASE, ("vase", "urn", "planter")),
        (ObjectType.TAPESTRY, ("tapestry", "wallhanging", "banner", "flag")),
        (ObjectType.FRAME, ("frame", "pictureframe")),
        (ObjectType.PEGBOARD, ("pegboard",)),
        (ObjectType.SIGN, ("sign", "menuboard", "whiteboard", "blackboard", "chart", "map")),
        (ObjectType.ART, ("art", "painting", "poster", "print", "picture", "canvas", "mural",
                          "decor", "ornament", "brassdecor", "stainedglass", "globe", "seashell")),
        # -- Food --------------------------------------------------------------
        (ObjectType.FOOD, ("apple", "fig", "pastry", "cannedgood", "canned", "condiment",
                            "croissant", "bakingpowder", "flourbag", "bread", "herb", "spice")),
        # -- Reading / office --------------------------------------------------
        (ObjectType.BOOK, ("book", "notebook", "magazine", "notepad", "tome", "volume", "folio",
                           "textbook", "cookbook", "hardcover", "novel", "codex")),
        (ObjectType.PEN, ("pen", "pencil", "crayon", "quill")),
        (ObjectType.OFFICE_SUPPLY, ("stapler", "paperclip", "ruler", "folder", "eraser", "tape",
                                     "scissors", "businesscard")),
        # -- Bath / personal care ----------------------------------------------
        (ObjectType.TOILET, ("toilet",)),
        (ObjectType.BATHTUB, ("bathtub", "shower")),
        (ObjectType.TOWEL, ("towel", "napkin")),
        (ObjectType.PERSONAL_CARE_PRODUCT, ("soap", "shampoo", "lotion", "conditioner",
                                             "toothbrush", "toothpaste", "cosmetic", "perfume",
                                             "sanitizer", "bodywash", "hairproduct", "comb",
                                             "brush", "diaper", "syringe", "medicalsupply",
                                             "stethoscope")),
        # -- Tools / hardware ----------------------------------------------------
        (ObjectType.TOOL, ("tool", "wrench", "hammer", "screwdriver", "drill", "pliers",
                            "sander", "scale", "gauge")),
        (ObjectType.HARDWARE, ("gear", "wire", "pipe", "hook", "outlet", "cable", "circuit",
                                "socket", "cog", "chip", "sensor", "router", "key", "button")),
        (ObjectType.LADDER, ("ladder",)),
        (ObjectType.SAFETY_EQUIPMENT, ("extinguisher", "smokedetector", "firealarm")),
        # -- Containers ----------------------------------------------------------
        (ObjectType.TRASH, ("trash", "waste")),
        (ObjectType.BASKET, ("basket",)),
        (ObjectType.BIN, ("bin",)),
        (ObjectType.BOX, ("box",)),
        (ObjectType.BUCKET, ("bucket",)),
        (ObjectType.CONTAINER, ("container", "case", "can", "barrel", "tub", "trunk", "caddy")),
        # -- Structural / architectural --------------------------------------
        (ObjectType.WINDOW, ("window",)),
        (ObjectType.DOOR, ("door",)),
        (ObjectType.FIREPLACE, ("fireplace",)),
        (ObjectType.VENT, ("vent", "radiator")),
        (ObjectType.PANEL, ("panel", "tile", "wallpaper", "molding", "column", "beam", "arch",
                             "grille", "trim")),
        # -- Textiles --------------------------------------------------------
        (ObjectType.PILLOW, ("pillow", "cushion")),
        (ObjectType.TEXTILE, ("textile", "fabric", "rug", "carpet", "blanket")),
        # -- Misc ---------------------------------------------------------------
        (ObjectType.APPAREL, ("shoe", "watch", "glasses")),
        (ObjectType.SPORTS_EQUIPMENT, ("dumbbell", "treadmill", "elliptical", "kettlebell")),
        (ObjectType.VEHICLE, ("car", "bike", "tire")),
        (ObjectType.RETAIL_FIXTURE, ("cashregister", "register", "checkout", "pricetag", "coin",
                                      "display", "kiosk", "station", "booth")),
        (ObjectType.TOY, ("toy",)),
        (ObjectType.WASHING_MACHINE, ("washingmachine", "washer")),
        (ObjectType.DRYER, ("dryer",)),
    )

    def classify(self, raw_type: str) -> ObjectType:
        """
        Return the :class:`ObjectType` category whose keywords best match
        *raw_type*.

        :param raw_type: A raw, near-instance-specific ``object_type``
            string from the sage10k dataset (e.g. ``"book2"``).
        :return: The best-matching generalized category, or
            :attr:`ObjectType.OTHER` if no keyword matches.
        """
        normalized = raw_type.strip().lower()
        for object_type, keywords in self._KEYWORDS_BY_TYPE:
            if any(keyword in normalized for keyword in keywords):
                return object_type
        return ObjectType.OTHER
