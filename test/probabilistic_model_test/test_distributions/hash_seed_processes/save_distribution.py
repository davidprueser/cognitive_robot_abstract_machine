from __future__ import annotations

import json
import sys
from pathlib import Path

from krrood.adapters.json_serializer import to_json

from .string_backed_category import string_backed_distribution


# %% export in a process of its own
def main() -> None:
    """
    Export :func:`string_backed_distribution` as JSON to the path given as the first
    argument.
    """
    Path(sys.argv[1]).write_text(json.dumps(to_json(string_backed_distribution())))


if __name__ == "__main__":
    main()
