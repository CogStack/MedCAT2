from typing import Optional
import copy

from medcat.components.addons.meta_cat.mctokenizers.tokenizers import (
    TokenizerWrapperBase)
import logging

logger = logging.getLogger(__name__)


def prepare_from_json(data: dict,
                      cntx_left: int,
                      cntx_right: int,
                      tokenizer: TokenizerWrapperBase,
                      cui_filter: Optional[set] = None,
                      replace_center: Optional[str] = None,
                      prerequisites: dict = {},
                      lowercase: bool = True) -> dict:
    """Convert the data from a json format into a CSV-like format for
    training. This function is not very efficient (the one working with
    documents as part of the meta_cat.pipe method is much better).
    If your dataset is > 1M documents think about rewriting this function
    - but would be strange to have more than 1M manually annotated documents.

    Args:
        data (dict):
            Loaded output of MedCATtrainer. If we have a `my_export.json`
            from MedCATtrainer, than data = json.load(<my_export>).
        cntx_left (int):
            Size of context to get from the left of the concept
        cntx_right (int):
            Size of context to get from the right of the concept
        tokenizer (TokenizerWrapperBase):
            Something to split text into tokens for the LSTM/BERT/whatever
            meta models.
        replace_center (Optional[str]):
            If not None the center word (concept) will be replaced with
            whatever this is.
        prerequisites (dict):
            A map of prerequisites, for example our data has two
            meta-annotations (experiencer, negation). Assume I want to create
            a dataset for `negation` but only in those cases where
            `experiencer=patient`, my prerequisites would be:
                {'Experiencer': 'Patient'} - Take care that the CASE has to
                            match whatever is in the data. Defaults to `{}`.
        lowercase (bool):
            Should the text be lowercased before tokenization.
            Defaults to True.
        cui_filter (Optional[set]):
            CUI filter if set. Defaults to None.

    Returns:
        out_data (dict):
            Example: {'category_name': [('<category_value>', '<[tokens]>',
                        '<center_token>'), ...], ...}
    """
    out_data: dict = {}

    for project in data['projects']:
        for document in project['documents']:
            text = str(document['text'])
            if lowercase:
                text = text.lower()

            if len(text) > 0:
                doc_text = tokenizer(text)

                for ann in document.get('annotations', document.get(
                        # A hack to support entities and annotations
                        'entities', {}).values()):
                    cui = ann['cui']
                    skip = False
                    if 'meta_anns' in ann and prerequisites:
                        # It is possible to require certain meta_anns to exist
                        # and have a specific value
                        for meta_ann in prerequisites:
                            if (meta_ann not in ann['meta_anns'] or
                                    ann['meta_anns'][meta_ann][
                                        'value'] != prerequisites[meta_ann]):
                                # Skip this annotation as the prerequisite
                                # is not met
                                skip = True
                                break

                    if not skip and (cui_filter is None or
                                     not cui_filter or cui in cui_filter):
                        if ann.get('validated', True) and (
                                not ann.get('deleted', False) and
                                not ann.get('killed', False)
                                and not ann.get('irrelevant', False)):
                            start = ann['start']
                            end = ann['end']

                            # Updated implementation to extract all the tokens
                            # for the medical entity (rather than the one)
                            ctoken_idx = []
                            for ind, pair in enumerate(
                                    doc_text['offset_mapping']):
                                if start <= pair[0] or start <= pair[1]:
                                    if end <= pair[1]:
                                        ctoken_idx.append(ind)
                                        break
                                    else:
                                        ctoken_idx.append(ind)

                            _start = max(0, ctoken_idx[0] - cntx_left)
                            _end = min(len(doc_text['input_ids']),
                                       ctoken_idx[-1] + 1 + cntx_right)

                            cpos = cntx_left + min(0, ind - cntx_left)
                            cpos_new = [x - _start for x in ctoken_idx]
                            tkns = doc_text['input_ids'][_start:_end]

                            if replace_center is not None:
                                if lowercase:
                                    replace_center = replace_center.lower()
                                for p_ind, pair in enumerate(
                                        doc_text['offset_mapping']):
                                    if start >= pair[0] and start < pair[1]:
                                        s_ind = p_ind
                                    if end > pair[0] and end <= pair[1]:
                                        e_ind = p_ind

                                ln = e_ind - s_ind
                                tkns = tkns[:cpos] + tokenizer(
                                    replace_center)['input_ids'] + tkns[
                                        cpos + ln + 1:]

                            # Backward compatibility if meta_anns is a list vs
                            # dict in the new approach
                            meta_anns: list[dict] = []
                            if 'meta_anns' in ann:
                                if isinstance(ann['meta_anns'], dict):
                                    meta_anns.extend(ann['meta_anns'].values())
                                else:
                                    meta_anns.extend(ann['meta_anns'])

                            # If the annotation is validated
                            for meta_ann in meta_anns:
                                name = meta_ann['name']
                                value = meta_ann['value']

                                sample = [tkns, cpos_new, value]

                                if name in out_data:
                                    out_data[name].append(sample)
                                else:
                                    out_data[name] = [sample]
    return out_data


def prepare_for_oversampled_data(data: list,
                                 tokenizer: TokenizerWrapperBase) -> list:
    """Convert the data from a json format into a CSV-like format for
       training. This function is not very efficient (the one working with
       documents as part of the meta_cat.pipe method is much better).
       If your dataset is > 1M documents think about rewriting this function -
       but would be strange to have more than 1M manually annotated documents.

       Args:
           data (list):
               Oversampled data expected in the following format:
               [[['text','of','the','document'], [index of medical entity],
                    "label" ],
                ['text','of','the','document'], [index of medical entity],
                    "label" ]]
           tokenizer (TokenizerWrapperBase):
                Something to split text into tokens for the LSTM/BERT/whatever
                meta models.

       Returns:
            data_sampled (list):
                The processed data in the format that can be merged with the
                output from prepare_from_json.
                [[<[tokens]>, [index of medical entity], "label" ],
                <[tokens]>, [index of medical entity], "label" ]]
                """

    data_sampled = []
    for sample in data:
        # Checking if the input is already tokenized
        if isinstance(sample[0][0], str):
            doc_text = tokenizer(sample[0])
            data_sampled.append([
                doc_text[0]['input_ids'], sample[1], sample[2]])
        else:
            data_sampled.append([sample[0], sample[1], sample[2]])

    return data_sampled


def encode_category_values(data: dict,
                           existing_category_value2id: Optional[dict] = None,
                           category_undersample=None,
                           alternative_class_names: list[list[str]] = []
                           ) -> tuple:
    """Converts the category values in the data outputted by
    `prepare_from_json` into integer values.

    Args:
        data (dict):
            Output of `prepare_from_json`.
        existing_category_value2id(Optional[dict]):
            Map from category_value to id (old/existing).
        category_undersample:
            Name of class that should be used to undersample the data (for 2
            phase learning)
        alternative_class_names (list[list[str]]):
            A list of lists of strings, where each list contains variations
            of a class name. Usually read from the config at
            `config.general.alternative_class_names`.

    Returns:
        dict:
            New data with integers inplace of strings for category values.
        dict:
            New undersampled data (for 2 phase learning) with integers
            inplace of strings for category values
        dict:
            Map from category value to ID for all categories in the data.

    Raises:
        Exception: If categoryvalue2id is pre-defined and its labels do
            not match the labels found in the data
    """
    data_list = list(data)
    if existing_category_value2id is not None:
        category_value2id = existing_category_value2id
    else:
        category_value2id = {}

    category_values = set([x[2] for x in data_list])

    if (len(category_value2id) != 0 and
            set(category_value2id.keys()) != category_values):
        # if categoryvalue2id doesn't match the labels in the data,
        # then 'alternative_class_names' has to be defined to check
        # for variations
        if len(alternative_class_names) == 0:
            # Raise an exception since the labels don't match
            raise Exception(
                "The classes set in the config are not the same as the one "
                "found in the data. The classes present in the config vs the "
                "ones found in the data - {set(category_value2id.keys())}, "
                f"{category_values}. Additionally, ensure the populate the "
                "'alternative_class_names' attribute to accommodate for "
                "variations.")
        updated_category_value2id = {}
        for _class in category_value2id.keys():
            if _class in category_values:
                updated_category_value2id[_class] = category_value2id[_class]
            else:
                found_in = [sub_map for sub_map in alternative_class_names
                            if _class in sub_map]
                failed_to_find = False
                if len(found_in) != 0:
                    class_name_matched = [label for label in found_in[0]
                                          if label in category_values]
                    if len(class_name_matched) != 0:
                        updated_category_value2id[class_name_matched[0]
                                                  ] = category_value2id[_class]
                        logger.info(
                            "Class name '%s' does not exist in the data; "
                            "however a variation of it '%s' is present; "
                            "updating it...", _class, class_name_matched[0])
                    else:
                        failed_to_find = True
                else:
                    failed_to_find = True
                if failed_to_find:
                    raise Exception(
                        "The classes set in the config are not the same as "
                        "the one found in the data. The classes present in "
                        "the config vs the ones found in the data - "
                        f"{set(category_value2id.keys())}, {category_values}. "
                        "Additionally, ensure the populate the "
                        "'alternative_class_names' attribute to accommodate "
                        "for variations.")
        category_value2id = copy.deepcopy(updated_category_value2id)
        logger.info("Updated categoryvalue2id mapping - %s", category_value2id)
    # Else create the mapping from the labels found in the data
    else:
        for c in category_values:
            if c not in category_value2id:
                category_value2id[c] = len(category_value2id)
        logger.info("Categoryvalue2id mapping created with labels found "
                    "in the data - %s", category_value2id)

    # Map values to numbers
    for i in range(len(data_list)):
        data_list[i][2] = category_value2id[data_list[i][2]]

    # Creating dict with labels and its number of samples
    label_data_ = {v: 0 for v in category_value2id.values()}
    for i in range(len(data_list)):
        if data_list[i][2] in category_value2id.values():
            label_data_[data_list[i][2]] = label_data_[data_list[i][2]] + 1

    logger.info("Original number of samples per label: %s", label_data_)
    # Undersampling data
    if category_undersample is None or category_undersample == '':
        min_label = min(label_data_.values())

    else:
        if (category_undersample not in label_data_.keys() and
                category_undersample in category_value2id.keys()):
            min_label = label_data_[category_value2id[category_undersample]]
        else:
            min_label = label_data_[category_undersample]

    data_undersampled = []
    label_data_counter = {v: 0 for v in category_value2id.values()}

    for sample in data_list:
        if label_data_counter[sample[-1]] < min_label:
            data_undersampled.append(sample)
            label_data_counter[sample[-1]] += 1

    label_data = {v: 0 for v in category_value2id.values()}
    for i in range(len(data_undersampled)):
        if data_undersampled[i][2] in category_value2id.values():
            label_data[data_undersampled[i][2]] = label_data[
                data_undersampled[i][2]] + 1
    logger.info("Updated number of samples per label (for 2-phase learning): "
                "%s", label_data)

    return data_list, data_undersampled, category_value2id
