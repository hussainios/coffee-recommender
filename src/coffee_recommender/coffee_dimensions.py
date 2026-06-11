from __future__ import annotations


PROFILE_DIMENSIONS: tuple[str, ...] = (
    "acidity",
    "sweetness",
    "body",
    "bitterness",
    "fruitiness",
    "chocolate_nutty",
    "floral",
    "funky_fermented",
    "roasty",
)

SENSORY_DIMENSIONS: tuple[str, ...] = (*PROFILE_DIMENSIONS, "clean_cup")

PROCESS_DIMENSIONS: tuple[str, ...] = (
    "process_washed",
    "process_natural",
    "process_honey",
    "process_anaerobic",
    "process_cofermented",
)
