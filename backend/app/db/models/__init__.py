"""Package aggregating ORM models.
Automatically loads definitions from the original `models.py` file so that
`import app.db.models` continues to expose core models like `User`, `Dataset`, etc.
"""
from importlib import util as _import_util
from importlib import machinery as _import_machinery
from pathlib import Path as _Path
import sys as _sys

_legacy_path = (_Path(__file__).resolve().parent.parent / "models.py").as_posix()
_spec = _import_util.spec_from_file_location("app.db._legacy_models", _legacy_path)
_legacy_module = _import_util.module_from_spec(_spec)  # type: ignore
_sys.modules[_spec.name] = _legacy_module  # type: ignore
assert _spec.loader is not None
_spec.loader.exec_module(_legacy_module)  # type: ignore

# re-export everything except private names
for _name in dir(_legacy_module):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_legacy_module, _name)

del _import_util, _import_machinery, _Path, _sys, _legacy_path, _spec, _legacy_module, _name
