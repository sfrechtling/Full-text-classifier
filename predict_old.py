#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# 
# Date: 18.01.2012
#
#

import os
import argparse
import cPickle as pickle
from re import search

from nltk.tokenize import word_tokenize, sent_tokenize
import scipy as sp

from app_libs.learners import learner_names
from app_libs.file_proc import remove_pdf_password
from app_libs.text_proc import html2text, pdf2text, token_filter
from app_libs.file_proc import UnicodeWriter
from app_libs.app_config import *


def main():
    parser = argparse.ArgumentParser(
        description="Make prediction")
    parser.add_argument("model_file",
        metavar="ModelFile",
        type=str,
        nargs=1,
        help="File with prediction model obtained from learn.py")
    parser.add_argument("sample_file",
        metavar="SampleFile",
        type=str,
        nargs=1,
        help="File with sample text (pdf, html or txt).")
    parser.add_argument("--xpath", action="store",
        help="XPath for html file if necessary")
    parser.add_argument("--report", action="store_true",
        help="Save words report against dictionary as report.csv")

    args = parser.parse_args()
    sfile = args.sample_file[0]
    mfile = args.model_file[0]

    try:
        ext = search(r"\A.*\.(.+)\Z", sfile.strip()).group(1).lower()
    except AttributeError:
        print "Can not understand file format - no extension"
        return
    if ext == "pdf":
        if os.path.exists(TMP_PDF):
            os.unlink(TMP_PDF)
        remove_pdf_password(sfile, TMP_PDF)
        text = pdf2text(TMP_PDF)
    elif ext == "htm" or ext == "html":
        text = html2text(sfile, args.xpath)
    elif ext == "txt" or ext == "text":
        f = open(sfile, "r")
        text = f.read()
    else:
        print "%s: unsupported file type" % sfile
        return

    sent_tokens = sent_tokenize(text)
    res = []
    for sent_num, sent in enumerate(sent_tokens):
        tokens = [token_filter(t) for t in word_tokenize(sent) if
            token_filter(t)]
        res.append(tokens)

    learners_object = pickle.load(open(mfile, "rb"))
    trans_file = learners_object["trans_file"]
    learners = learners_object["learners"]
    scaler = learners_object["scaler"]

    trans = pickle.load(open(trans_file, "rb"))
    vec = trans.get_sample_vector(res)
    bag = trans._get_bag(res)

    if scaler:
        vec = scaler.transform(vec)
    agr = 0
    for k, learner in enumerate(learners):
        if k == 5 and vec.__class__ == sp.sparse.csr.csr_matrix:
        # Last learner (boosting) only supports dense data
            vec = vec.todense()
        if learner.predict(vec)[0] > 0:
            pr = "pos"
            agr += 1
        else:
            pr = "neg"
            agr -= 1
        print "{:<60} : [ {:>3} ]".format(learner_names[k], pr)

    if agr > 0:
        agr_res = 'pos' 
    elif agr < 0:
        agr_res = 'neg' 
    else:
        agr_res = 'N/D'
    print "{:<60} : [ {:>3} ]".format(learner_names[k+1], agr_res)

    if agr > 3:
        agr_res = 'pos' 
    elif agr < -3:
        agr_res = 'neg' 
    else:
        agr_res = 'N/D' 
    print "{:<60} : [ {:>3} ]".format(learner_names[k+2], agr_res)

    if agr > 5:
        agr_res = 'pos' 
    elif agr < -5:
        agr_res = 'neg' 
    else:
        agr_res = 'N/D'
    print "{:<60} : [ {:>3} ]".format(learner_names[k+3], agr_res)

    if args.report:
        f = open("report.csv", "wb")
        csv = UnicodeWriter(f, delimiter = CSV_DELIMITER_WRITE)
        csv.writerow(["Dictionary length:", repr(trans.N)])
        csv.writerow(["Number of words from dictionary in document:",
            repr(len(bag))])
        csv.writerow([])
        csv.writerow(["Word", "count", "Argument for", "pos count (dict)",
            "neg count (dict)", "pos idf", "neg idf"])
        for (w, ipos, ineg, cpos, cneg) in trans.dwords:
            if w in [x for (x, c) in bag]:
                c = [c for (x, c) in bag if x==w][0]
                tend = "pos"
                if cpos < cneg:
                    tend = "neg"
                csv.writerow([w, repr(c), tend, repr(cpos), repr(cneg),
                    repr(ipos), repr(ineg)])
        f.close()

if __name__ == "__main__":
    main()
