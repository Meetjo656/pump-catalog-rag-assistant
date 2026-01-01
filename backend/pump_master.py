import os
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MASTER_CSV = os.path.join(PROJECT_ROOT, "csv", "pump_master.csv")

_model_name_to_id = {}
_model_id_to_name = {}
_all_pumps = None


def _load_master():
    global _all_pumps

    if _all_pumps is not None:
        return

    _all_pumps = []

    with open(MASTER_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("is_active", "").lower() != "yes":
                continue

            model_id = row["model_id"].strip()
            model_name = row["model_name"].strip()

            _model_name_to_id[model_name.lower()] = model_id
            _model_id_to_name[model_id] = model_name

            _all_pumps.append({
                "id": model_id,
                "label": model_name,
            })


def get_all_pumps():
    _load_master()
    return _all_pumps


def get_model_id_by_name(model_name: str) -> str:
    _load_master()
    key = model_name.strip().lower()
    if key not in _model_name_to_id:
        raise ValueError(f"Unknown pump model name: {model_name}")
    return _model_name_to_id[key]


def get_model_name_by_id(model_id: str) -> str:
    _load_master()
    return _model_id_to_name.get(model_id, model_id)
