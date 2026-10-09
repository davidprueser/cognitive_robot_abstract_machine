from __future__ import annotations

import json
import sys
from pathlib import Path

from krrood.adapters.json_serializer import from_json

from .string_backed_category import StringBackedCategory


# %% import in a process of its own
def main() -> None:
    """
    Load the distribution exported to the path given as the first argument and write the
    probability of every :class:`StringBackedCategory` member, keyed by the member, as
    JSON to the path given as the second argument.
    """
    distribution = from_json(json.loads(Path(sys.argv[1]).read_text()))
    probabilities = {
        member.value: distribution.probabilities[hash(member)]
        for member in StringBackedCategory
    }
    Path(sys.argv[2]).write_text(json.dumps(probabilities))


if __name__ == "__main__":
    main()
