import os
import shutil
import logging
from typing import Optional

from medcat2.cat import CAT

from medcat2.utils.legacy.convert_cdb import get_cdb_from_old
from medcat2.utils.legacy.convert_config import get_config_from_old
from medcat2.utils.legacy.convert_vocab import get_vocab_from_old
from medcat2.storage.serialisers import AvailableSerialisers


logger = logging.getLogger(__name__)


class Converter:
    """Converts v1 models to v2 models."""
    cdb_name = 'cdb.dat'
    vocab_name = 'vocab.dat'
    config_name = 'config.dat'

    def __init__(self, medcat1_model_pack_path: str,
                 new_model_pack_path: Optional[str],
                 ser_type: AvailableSerialisers = AvailableSerialisers.dill):
        if medcat1_model_pack_path.endswith(".zip"):
            folder_path = medcat1_model_pack_path[:-4]
            unpack(medcat1_model_pack_path, folder_path)
            medcat1_model_pack_path = folder_path
        if not os.path.isdir(medcat1_model_pack_path):
            raise ValueError(
                "Provided model path is not a directory: "
                f"{medcat1_model_pack_path}")
        self.old_model_folder = medcat1_model_pack_path
        self.new_model_folder = new_model_pack_path
        self.ser_type = ser_type
        self._validate()

    @property
    def expected_files_in_folder(self):
        """The base names of the required files in a folder for a v1 model."""
        return [self.cdb_name, self.vocab_name, ]

    def _validate(self):
        for fn in self.expected_files_in_folder:
            path = os.path.join(self.old_model_folder, fn)
            if not os.path.exists(path):
                raise ValueError(f"Unable to find {fn} in model folder "
                                 f"{self.old_model_folder}")

    def convert(self) -> CAT:
        """Use the gathered information to convert to a v2 model.

        This converts the CDB, Vocab, and Config, in order and then
        created the model pack.

        If `self.new_model_folder` is set, the model will be saved as well.

        Returns:
            CAT: The model pack.
        """
        cdb = get_cdb_from_old(
            os.path.join(self.old_model_folder, self.cdb_name))
        vocab = get_vocab_from_old(
            os.path.join(self.old_model_folder, self.vocab_name))
        cnf_path = os.path.join(self.old_model_folder, self.config_name)
        if os.path.exists(cnf_path):
            config = get_config_from_old(cnf_path)
        else:
            config = cdb.config
        cat = CAT(cdb, vocab, config)
        if self.new_model_folder:
            logger.info("Saving converted model to '%s'",
                        self.new_model_folder)
            cat.save_model_pack(self.new_model_folder,
                                serialiser_type=self.ser_type)
        return cat


def unpack(model_zip_path: str, target_folder: str):
    """Unpack v1 model into target folder.

    Args:
        model_zip_path (str): ZIP path.
        target_folder (str): Target folder.
    """
    shutil.unpack_archive(model_zip_path, extract_dir=target_folder)
