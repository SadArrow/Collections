from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HaloTaskSpec:
    task_name: str
    script_name: str
    output_dir_name: str
    stage_count: int

    @property
    def data_dir_rel(self) -> str:
        return f"Data/{self.output_dir_name}"


TASK_SPECS: tuple[HaloTaskSpec, ...] = (
    HaloTaskSpec("Fling_Dress", "Fling_Dress_HALO.py", "Fling_Dress_Validation_HALO", 2),
    HaloTaskSpec("Fling_Tops", "Fling_Tops_HALO.py", "Fling_Tops_Validation_HALO", 2),
    HaloTaskSpec("Fling_Trousers", "Fling_Trousers_HALO.py", "Fling_Trousers_Validation_HALO", 1),
    HaloTaskSpec("Fold_Dress", "Fold_Dress_HALO.py", "Fold_Dress_Validation_HALO", 3),
    HaloTaskSpec("Fold_Tops", "Fold_Tops_HALO.py", "Fold_Tops_Validation_HALO", 3),
    HaloTaskSpec("Fold_Trousers", "Fold_Trousers_HALO.py", "Fold_Trousers_Validation_HALO", 2),
    HaloTaskSpec("Hang_Coat", "Hang_Coat_HALO.py", "Hang_Coat_Validation_HALO", 1),
    HaloTaskSpec("Hang_Dress", "Hang_Dress_HALO.py", "Hang_Dress_Validation_HALO", 1),
    HaloTaskSpec("Hang_Tops", "Hang_Tops_HALO.py", "Hang_Tops_Validation_HALO", 1),
    HaloTaskSpec("Hang_Trousers", "Hang_Trousers_HALO.py", "Hang_Trousers_Validation_HALO", 1),
    HaloTaskSpec("Store_Tops", "Store_Tops_HALO.py", "Store_Tops_Validation_HALO", 1),
    HaloTaskSpec("Wear_Baseballcap", "Wear_Baseballcap_HALO.py", "Wear_Baseballcap_Validation_HALO", 1),
    HaloTaskSpec("Wear_Bowlhat", "Wear_Bowlhat_HALO.py", "Wear_Bowlhat_Validation_HALO", 1),
    HaloTaskSpec("Wear_Scarf", "Wear_Scarf_HALO.py", "Wear_Scarf_Validation_HALO", 1),
)

TASK_BY_NAME: dict[str, HaloTaskSpec] = {item.task_name: item for item in TASK_SPECS}


def parse_task_names(raw: str) -> list[HaloTaskSpec]:
    text = str(raw or "").strip()
    if not text or text.lower() == "all":
        return list(TASK_SPECS)
    specs: list[HaloTaskSpec] = []
    for token in text.split(","):
        name = token.strip()
        if not name:
            continue
        if name not in TASK_BY_NAME:
            known = ", ".join(item.task_name for item in TASK_SPECS)
            raise KeyError(f"Unknown HALO task: {name!r}. Known tasks: {known}")
        specs.append(TASK_BY_NAME[name])
    if not specs:
        raise ValueError("No HALO tasks selected.")
    return specs
