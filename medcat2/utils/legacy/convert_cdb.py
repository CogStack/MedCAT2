import dill
import logging

from medcat2.cdb import CDB
from medcat2.config import Config
from medcat2.cdb.concepts import get_new_cui_info, get_new_name_info
from medcat2.utils.legacy.convert_config import get_config_from_nested_dict


logger = logging.getLogger(__name__)


class LegacyClassNotFound:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return f"<LegacyClassNotFound args={self.args} kwargs={self.kwargs}>"


class CustomUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            logger.warning(
                "Missing class %s.%s, replacing with LegacyClassNotFound.",
                module, name)
            return LegacyClassNotFound


def load_old_raw_data(old_path: str) -> dict:
    """Looads the raw data from old file.

    This uses a wrapper that allows loading the data even if the classes
    do not exist.

    Args:
        old_path (str): The path of the file to read.

    Returns:
        dict: The resulting raw data.
    """
    with open(old_path, 'rb') as f:
        # NOTE: custom unpickler needed because we
        #       do not have access to original modules within medcat(v1)
        data = CustomUnpickler(f).load()
    return data


EXPECTED_USEFUL_KEYS = [
    'name2cuis', 'name2cuis2status', 'name2count_train', 'name_isupper',

    'snames',

    'cui2names', 'cui2snames', 'cui2context_vectors', 'cui2count_train',
    'cui2tags', 'cui2type_ids', 'cui2preferred_name', 'cui2average_confidence',

    'addl_info',
    'vocab',
]
NAME2KEYS = {'name2cuis', 'name2cuis2status', 'name2count_train',
             'name_isupper'}
CUI2KEYS = {'cui2names', 'cui2snames', 'cui2context_vectors',
            'cui2count_train', 'cui2info', 'cui2tags', 'cui2type_ids',
            'cui2preferred_name', 'cui2average_confidence', }
TO_RENAME = {'vocab': 'token_counts'}


def _add_cui_info(cdb: CDB, data: dict) -> CDB:
    all_cuis = set()
    for key in CUI2KEYS:
        ccuis = data[key].keys()
        logger.debug("Adding %d cuis based on '%s", len(ccuis), key)
        all_cuis.update(ccuis)
    logger.info("A total of %d CUIs identified", len(all_cuis))
    cui2names, cui2snames = data['cui2names'], data['cui2snames']
    cui2cv, cui2ct = data['cui2context_vectors'], data['cui2count_train']
    cui2tags, cui2type_ids = data['cui2tags'], data['cui2type_ids']
    cui2prefname = data['cui2preferred_name']
    cui2av_conf = data['cui2average_confidence']
    for cui in all_cuis:
        names = cui2names.get(cui, set())
        snames = cui2snames.get(cui, set())
        vecs = cui2cv.get(cui, None)
        count_train = cui2ct.get(cui, 0)
        tags = cui2tags.get(cui, None)
        type_ids = cui2type_ids.get(cui, None)
        prefname = cui2prefname.get(cui, None)
        av_conf = cui2av_conf.get(cui, 0.0)
        info = get_new_cui_info(
            cui=cui, preferred_name=prefname, names=names, subnames=snames,
            type_ids=type_ids, tags=tags, count_train=count_train,
            context_vectors=vecs, average_confidence=av_conf,
        )
        cdb.cui2info[cui] = info
    cdb.addl_info.update(data['addl_info'])
    # for cui, ontos in data['addl_info']['cui2ontologies'].items():
    #     cdb.cui2info[cui]['in_other_ontology'] = ontos
    return cdb


def _add_name_info(cdb: CDB, data: dict) -> CDB:
    all_names = set()
    for key in NAME2KEYS:
        cnames = data[key].keys()
        logger.debug("Adding %d names based on '%s", len(cnames), key)
        all_names.update(cnames)
    logger.info("A total of %d names found", len(all_names))
    logger.info("Adding names from cui2names")
    # add from cui2names
    for cui_infos in cdb.cui2info.values():
        all_names.update(cui_infos['names'])
    logger.info("A total of %d names found after adding from cui2names",
                len(all_names))
    # NOTE: name2cuis has the same cuis as name2cuis2status
    #       so v2 only uses the latter since it provides extra information
    name2cuis2status = data['name2cuis2status']
    name2cnt_train = data['name2count_train']
    name2is_upper = data['name_isupper']
    for name in all_names:
        cuis2status: dict[str, str] = {}
        _cuis2status = name2cuis2status.get(name, {})
        cuis2status.update(_cuis2status)
        cnt_train = name2cnt_train.get(name, 0)
        is_upper = name2is_upper.get(name, False)
        info = get_new_name_info(name, per_cui_status=cuis2status,
                                 is_upper=is_upper, count_train=cnt_train)
        cdb.name2info[name] = info
    return cdb


def update_names(cdb: CDB, data: dict):
    for name_from, name_to in TO_RENAME.items():
        setattr(cdb, name_to, data[name_from])


def convert_data(all_data: dict) -> CDB:
    """Convert the raw v1 data into a CDB.

    Args:
        all_data (dict): The raw v1 data off disk.

    Returns:
        CDB: The v2 CDB.
    """
    data = all_data['cdb']
    cdb = CDB(Config())
    cdb = _add_cui_info(cdb, data)
    cdb = _add_name_info(cdb, data)
    update_names(cdb, data)
    if 'config' in all_data:
        logger.info("Loading old style CDB with config included.")
        cdb.config = get_config_from_nested_dict(all_data['config'])
    return cdb


def get_cdb_from_old(old_path: str) -> CDB:
    """Get the v2 CDB from a v1 CDB path.

    Args:
        old_path (str): The v1 CDB path.

    Returns:
        CDB: The v2 CDB.
    """
    data = load_old_raw_data(old_path)
    return convert_data(data)
