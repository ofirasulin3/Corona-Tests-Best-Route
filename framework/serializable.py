import json
import dataclasses
from typing import Optional, Union

from framework.ways.streets_map import *


__all__ = ['Serializable']


PrimitiveType = Union[int, str, float]
CollectionOfPrimitiveType = Union[Tuple[PrimitiveType], List[PrimitiveType]]


class Serializable:
    def to_dict(self) -> dict:
        assert dataclasses.is_dataclass(self)

        def serialize_field(field: dataclasses.Field) -> Union[dict, PrimitiveType, CollectionOfPrimitiveType]:
            field_value = getattr(self, field.name)
            if isinstance(field_value, Serializable):
                return field_value.to_dict()
            if isinstance(field_value, Junction):
                return field_value.index
            if isinstance(field_value, (tuple, list)):
                assert all(isinstance(val, (str, int, float)) for val in field_value)
                return list(field_value)
            assert any(issubclass(field.type, primitive_type) for primitive_type in (str, int, float))
            return field_value

        return {field.name: serialize_field(field) for field in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, dct: dict, streets_map: Optional[StreetsMap] = None) -> dict:
        assert dataclasses.is_dataclass(cls)
        assert len(set(field.name for field in dataclasses.fields(cls)) ^ set(dct.keys())) == 0

        def deserialize_field(
                field: dataclasses.Field,
                serialized_value: Union[dict, PrimitiveType, CollectionOfPrimitiveType]):
            if issubclass(field.type, Serializable):
                return field.type.from_dict(dct=serialized_value, streets_map=streets_map)
            if issubclass(field.type, Junction):
                assert streets_map is not None
                return streets_map[int(serialized_value)]
            if issubclass(field.type, (list, tuple)):
                assert all(isinstance(val, (str, int, float)) for val in serialized_value)
                return field.type(serialized_value)
            assert any(issubclass(field.type, primitive_type) for primitive_type in (str, int, float))
            return field.type(serialized_value)

        return cls(**{field.name: deserialize_field(field, dct[field.name]) for field in dataclasses.fields(cls)})

    def serialize(self) -> str:
        def convert(o):
            if isinstance(o, np.generic): return o.item()
            raise TypeError

        return json.dumps(self.to_dict(), default=convert)

    @classmethod
    def deserialize(cls, serialized: str, streets_map: Optional[StreetsMap] = None):
        assert dataclasses.is_dataclass(cls)
        return cls.from_dict(json.loads(serialized), streets_map)
