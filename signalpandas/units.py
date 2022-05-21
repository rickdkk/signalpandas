from typing import Optional

from signalpandas.sigtyping import Pandas


def set_units(data: Pandas, units: list[str], columns: Optional[list[str]] = None, copy: bool = True) -> Pandas:
    data = data.copy() if copy else data

    if columns is not None:
        unit_dict = {column: unit for column, unit in zip(columns, units)}
    else:
        unit_dict = {column: unit for column, unit in zip(data.columns, units)}
    data.attrs["units"] = unit_dict

    return data


def get_units():
    pass


def convert_units():
    pass


def to_base_units():
    pass
