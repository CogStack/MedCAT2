import json
from typing import Any, cast, Optional
import logging

from pydantic import BaseModel

from medcat2.config import Config


logger = logging.getLogger(__name__)


SET_IDENTIFIER = "==SET=="
CONFIG_KEEP_IDENTICAL = {
    'cdb_maker', 'preprocessing'
}
CONFIG_MOVE = {
    'linking': 'components.linking',
    'ner': 'components.ner',
    'version.description': 'meta.description',
    'version.id': 'meta.hash',
    'version.ontology': 'meta.ontology',
    'general.spacy_model': 'general.nlp.modelname',
    'general.spacy_disabled_components': 'general.nlp.disabled_components',
}
MOVE_WITH_REMOVES = {
    'general': {'checkpoint',  # TODO: Start supporitn checkpoints again
                'spacy_model', 'spacy_disabled_components', 'usage_monitor'},
    'annotation_output': {'doc_extended_info'},
}


def get_val_and_parent_model(old_data: Optional[dict],
                             cnf: Optional[Config],
                             path: str
                             ) -> tuple[Optional[Any], Optional[BaseModel]]:
    val = old_data
    target_model: Optional[BaseModel] = cnf
    name = path
    while name:
        parts = name.split(".", 1)
        cname = parts[0]
        if len(parts) == 2:
            name = parts[1]
            if target_model is not None:
                target_model = cast(BaseModel, getattr(target_model, cname))
        else:
            name = ''
        if val is not None:
            val = val[cname]
    return val, target_model


def _safe_setattr(target_model: BaseModel, fname: str, val: Any) -> None:
    mval = getattr(target_model, fname)
    if isinstance(mval, BaseModel) and isinstance(val, dict):
        for k, v in val.items():
            setattr(mval, k, v)
    else:
        setattr(target_model, fname, val)


def _move_identicals(cnf: Config, old_data: dict) -> Config:
    for name in CONFIG_KEEP_IDENTICAL:
        val, target_model = get_val_and_parent_model(old_data, cnf, name)
        val = cast(Any, val)
        target_model = cast(BaseModel, target_model)
        fname = name.split(".")[-1]
        logger.info("Setting %s.%s to %s", type(target_model).__name__, fname,
                    type(val).__name__)
        _safe_setattr(target_model, fname, val)
    return cnf


def _move_partials(cnf: Config, old_data: dict) -> Config:
    for path, to_remove in MOVE_WITH_REMOVES.items():
        val, target_model = get_val_and_parent_model(old_data, cnf, path)
        val = cast(Any, val)
        target_model = cast(BaseModel, target_model)
        val = cast(dict, val).copy()
        for remove in to_remove:
            del val[remove]
        fname = path.split(".")[-1]
        logger.info("Setting %s while removing %d", path, len(to_remove))
        _safe_setattr(target_model, fname, val)
    return cnf


def _relocate(cnf: Config, old_data: dict) -> Config:
    for orig_path, new_path in CONFIG_MOVE.items():
        orig_val, _ = get_val_and_parent_model(old_data, cnf=None,
                                               path=orig_path)
        _, target_model = get_val_and_parent_model(None, cnf=cnf,
                                                   path=new_path)
        orig_val = cast(Any, orig_val)
        target_model = cast(BaseModel, target_model)
        fname = new_path.split(".")[-1]
        logger.info("Relocating from %s to %s (%s)", orig_path, new_path,
                    type(orig_val).__name__)
        _safe_setattr(target_model, fname, orig_val)
    return cnf


def _sanitise_sets(old_data: dict) -> dict:
    for k in list(old_data):
        v = old_data[k]
        if isinstance(v, dict) and len(v) == 1 and SET_IDENTIFIER in v:
            logger.info("Moving ")
            old_data[k] = set(v[SET_IDENTIFIER])
        elif isinstance(v, dict):
            # in place anyway
            _sanitise_sets(v)
    return old_data


def _make_changes(cnf: Config, old_data: dict) -> Config:
    old_data = _sanitise_sets(old_data)
    cnf = _move_identicals(cnf, old_data)
    cnf = _move_partials(cnf, old_data)
    cnf = _relocate(cnf, old_data)
    return cnf


def get_config_from_old(path: str) -> Config:
    cnf = Config()
    with open(path) as f:
        old_cnf_data = json.load(f)
    return _make_changes(cnf, old_cnf_data)
