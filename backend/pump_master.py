# backend/pump_master.py

import os
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "..", "csv")

PUMP_MASTER_CSV = os.path.join(CSV_DIR, "pump_master.csv")


# =========================================================
# Load master pump list (cached)
# =========================================================

_PUMPS_CACHE = None
_NAME_TO_ID = {}
_ID_TO_NAME = {}


def _load_pumps():
    global _PUMPS_CACHE, _NAME_TO_ID, _ID_TO_NAME

    if _PUMPS_CACHE is not None:
        return

    pumps = []

    with open(PUMP_MASTER_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            model_id = row.get("model_id", "").strip()
            model_name = row.get("model_name", "").strip()

            if not model_id or not model_name:
                continue

            pumps.append({
                "id": model_id,
                "label": model_name
            })

            _NAME_TO_ID[model_name.lower()] = model_id
            _ID_TO_NAME[model_id] = model_name

    _PUMPS_CACHE = pumps


# =========================================================
# Public API
# =========================================================

def get_all_pumps():
    """
    Returns list of pumps for frontend dropdowns.

    Format:
    [
      { "id": "P00001", "label": "CSP-521-T" },
      ...
    ]
    """
    _load_pumps()
    return _PUMPS_CACHE


def get_model_id_by_name(name: str | None) -> str | None:
    if not name:
        return None

    _load_pumps()
    return _NAME_TO_ID.get(name.strip().lower())


def get_model_name_by_id(model_id: str | None) -> str | None:
    if not model_id:
        return None

    _load_pumps()
    return _ID_TO_NAME.get(model_id)


def resolve_model_identifier(value: str | None) -> str | None:
    """
    Accepts either:
    - model_id (P00001)
    - model_name (CSP-521-T)

    Returns:
    - model_id or None
    """
    if not value:
        return None

    value = value.strip()

    # Already a model_id
    if value.upper().startswith("P"):
        return value if get_model_name_by_id(value) else None

    # Otherwise try name → id
    return get_model_id_by_name(value)

if __name__ == "__main__":
    pumps = get_all_pumps()
    print(f"Loaded {len(pumps)} pumps")
    print("CSP-521-T →", resolve_model_identifier("CSP-521-T"))
    print("Sample:", pumps[:3])