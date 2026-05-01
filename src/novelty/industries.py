"""Mapping NICE class codes to human-readable industry labels used in the paper."""

INDUSTRY_NAME = {
    "009": "Software & Electronics",
    "032": "Beer & Soft Drinks",
}

INDUSTRY_SHORT = {
    "009": "software",
    "032": "beverages",
}


def name(cls: str) -> str:
    return INDUSTRY_NAME.get(cls, f"NICE Class {cls}")
