from typing import Optional
from contextlib import nullcontext

from medcat2.config.config import LinkingFilters
from medcat2.data.mctexport import MedCATTrainerExportProject
from medcat2.utils.config_utils import temp_changed_config


def project_filters(filters: LinkingFilters,
                    project: MedCATTrainerExportProject,
                    extra_cui_filter: Optional[set[str]],
                    use_project_filters: bool):
    if extra_cui_filter is not None and not use_project_filters:
        return temp_changed_config(filters, 'cuis', extra_cui_filter)
    if use_project_filters:
        cuis = project.get('cuis', None)
        if cuis is None or not cuis:
            return nullcontext()
        return temp_changed_config(filters, 'cuis', cuis)
    return temp_changed_config(filters, 'cuis', set())
