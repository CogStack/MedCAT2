from medcat2.tokenizing.tokenizers import MutableDocument, MutableEntity


# NOTE: the following used (in medcat v1) check tuis
#       but they were never passed to the method so
#       I've omitted it now
def create_main_ann(doc: MutableDocument) -> None:
    """Creates annotation in the spacy ents list
    from all the annotations for this document.

    Args:
        doc (Doc): Spacy document.
    """
    doc.all_ents.sort(key=lambda x: len(x.base.text), reverse=True)
    tkns_in = set()
    main_anns: list[MutableEntity] = []
    for ent in doc.all_ents:
        to_add = True
        for tkn in ent:
            if tkn in tkns_in:
                to_add = False
        if to_add:
            for tkn in ent:
                tkns_in.add(tkn)
            main_anns.append(ent)
    doc.final_ents = list(doc.final_ents) + main_anns  # type: ignore
