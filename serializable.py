from dataclasses import dataclass, fields
from typing import Union, Dict, Any

class Serializable:
    """Base class providing automatic serialization for HMoE dataclasses.

    This interface allows dataclass instances to be converted into a
    dictionary or string representation suitable for YAML or JSON output.
    It automatically drops default values, recursively serializes nested
    Serializable objects, and simplifies single-property objects.

    Methods:
        serialize() -> Union[str, Dict[str, Any]]:
            Converts the dataclass instance into a dictionary or string.
    """

    def serialize(self) -> Union[str, Dict[str, Any]]:
        """Serializes the dataclass into a dictionary or string.

        The method performs the following steps:

        1. Iterates over all dataclass fields.
        2. Skips fields that are equal to their default value (except 'name').
        3. Recursively serializes any nested objects that implement `serialize()`.
        4. If the only customized property is 'name', returns it as a string.
        5. Otherwise, returns a dictionary of all non-default properties.

        Returns:
            Union[str, Dict[str, Any]]: Serialized representation of the object.
        """
        f_dict = {}

        # Iterate through all dataclass fields
        for f in fields(self):
            val = getattr(self, f.name)

            # Skip fields set to default values to reduce YAML/JSON clutter
            if val == f.default and f.name != "name":
                continue

            # Recursively serialize nested Serializable objects
            if hasattr(val, 'serialize') and callable(val.serialize):
                f_dict[f.name] = val.serialize()
            else:
                f_dict[f.name] = val

        # If only 'name' is customized, return it as a string
        if len(f_dict) == 1 and "name" in f_dict:
            return f_dict["name"]

        return f_dict