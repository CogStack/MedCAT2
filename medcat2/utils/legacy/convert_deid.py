import os
import json

from medcat2.cdb.cdb import CDB
from medcat2.tokenizing.tokenizers import BaseTokenizer
from medcat2.components.ner.trf.transformers_ner import TransformersNER
from medcat2.config.config_transformers_ner import ConfigTransformersNER
from medcat2.utils.legacy.convert_cdb import get_cdb_from_old


exp_cat_config_name = "cat_config.json"
exp_trf_ner_config_name = "config.json"
exp_cdb_name = "cdb.dat"


def get_cnf(cnf_path: str) -> ConfigTransformersNER:
    cnf = ConfigTransformersNER()
    with open(cnf_path) as f:
        old_stuff = json.load(f)
    cnf.merge_config(old_stuff)
    return cnf


def get_trf_ner_from_old(old_path: str, tokenizer: BaseTokenizer
                         ) -> TransformersNER:
    config = get_cnf(os.path.join(old_path, exp_cat_config_name))
    config.general.model_name = old_path
    cdb: CDB = get_cdb_from_old(os.path.join(old_path, exp_cdb_name))
    return TransformersNER.create_new(cdb, tokenizer, config)
