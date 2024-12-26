## Frome the https://github.com/xdanny/pyspark_types/blob/main/pyspark_types/dataclass.py
## with some changes

from typing import Type, get_type_hints, Union
from dataclasses import is_dataclass, fields
import datetime
from pyspark.sql.types import *

from scaledp.utils.auxiliary import LongT, ShortT, ByteT, BinaryT, BoundDecimal

type_mapping = {
    str: StringType,
    int: IntegerType,
    LongT: LongType,
    ShortT: ShortType,
    ByteT: ByteType,
    float: DoubleType,
    datetime.datetime: TimestampType,
    datetime.date: DateType,
    bool: BooleanType,
    BinaryT: BinaryType,
}


def register_type(new_type, spark_type):
    type_mapping[new_type] = spark_type


def map_dataclass_to_struct(dataclass_type: Type) -> StructType:
    """
    Map a Python data class to a PySpark struct.

    :param dataclass_type: The Python data class to be mapped.
    :return: A PySpark struct that corresponds to the data class.
    """
    fields_list = []
    hints = get_type_hints(dataclass_type)

    for field in fields(dataclass_type):
        field_name = field.name
        field_type = field.type

        if is_dataclass(field_type):
            # Recursively map nested data classes to PySpark structs
            sub_struct = map_dataclass_to_struct(field_type)
            nullable = is_field_nullable(field_name, hints)
            fields_list.append(StructField(field_name, sub_struct, nullable))
        elif hasattr(field_type, "__origin__") and field_type.__origin__ is list:
            # Handle lists of elements
            elem_type = field_type.__args__[0]
            if is_dataclass(elem_type):
                # Handle lists of data classes
                sub_struct = map_dataclass_to_struct(elem_type)
                nullable = is_field_nullable(field_name, hints)
                fields_list.append(StructField(field_name, ArrayType(sub_struct), nullable))
            else:
                # Handle lists of primitive types and dicts
                spark_type = get_spark_type(elem_type)
                nullable = is_field_nullable(field_name, hints)
                if spark_type == MapType(StringType(), StringType()):
                    # Special case for dictionaries with any value type
                    fields_list.append(StructField(field_name, spark_type, nullable))
                else:
                    fields_list.append(StructField(field_name, ArrayType(spark_type), nullable))
        elif hasattr(field_type, "__origin__") and field_type.__origin__ is dict:
            # Handle dictionaries
            key_type, value_type = field_type.__args__
            if is_dataclass(value_type):
                sub_struct = map_dataclass_to_struct(value_type)
                nullable = is_field_nullable(field_name, hints)
                fields_list.append(
                    StructField(
                        field_name,
                        MapType(get_spark_type(key_type), sub_struct),
                        nullable,
                    )
                )
            else:
                spark_type = get_spark_type(value_type)
                nullable = is_field_nullable(field_name, hints)
                fields_list.append(
                    StructField(
                        field_name,
                        MapType(get_spark_type(key_type), spark_type),
                        nullable,
                    )
                )
        else:
            # Handle primitive types and BoundDecimal custom type
            spark_type = get_spark_type(field_type)
            nullable = is_field_nullable(field_name, hints)
            fields_list.append(StructField(field_name, spark_type, nullable))

    return StructType(fields_list)


def get_spark_type(py_type: Type, type_mapping=type_mapping) -> DataType:
    """
    Creates a mapping from a python type to a pyspark data type
    :param py_type:
    :return:
    """
    # Check if the type exists in the mapping
    if py_type in type_mapping:
        # If it's a function, call it (e.g., for Box.getSchema())
        return (
            type_mapping[py_type]() if callable(type_mapping[py_type]) else type_mapping[py_type]()
        )

    # Handle BoundDecimal separately as it needs specific attributes
    if isinstance(py_type, type) and issubclass(py_type, BoundDecimal):
        return DecimalType(precision=py_type.precision, scale=py_type.scale)

    # Handle Optional types
    if is_optional_type(py_type):
        elem_type = py_type.__args__[0]
        return get_spark_type(elem_type)

    raise Exception(f"Type {py_type} is not supported by PySpark")


def is_field_nullable(field_name: str, hints: dict) -> bool:
    """
    Returns True if the given field name is nullable, based on the type hint for the field in the given hints dictionary.
    """
    if field_name not in hints:
        return True
    field_type = hints[field_name]
    if is_optional_type(field_type):
        return True
    return False


def apply_nullability(dtype: DataType, is_nullable: bool) -> DataType:
    """
    Returns a new PySpark DataType with the nullable flag set to the given value.
    """
    if is_nullable:
        if isinstance(dtype, StructType):
            # Wrap the nullable field in a struct with a single field
            return StructType([StructField("value", dtype, True)])
        elif hasattr(dtype, "add_nullable"):
            return dtype.add_nullable()
        else:
            raise TypeError(f"Type {dtype} does not support nullability")
    else:
        return dtype


def is_optional_type(py_type: Type) -> bool:
    """
    Returns True if the given type is an Optional type.
    """
    if hasattr(py_type, "__origin__") and py_type.__origin__ is Union:
        args = py_type.__args__
        if len(args) == 2 and args[1] is type(None):
            return True
    return False
