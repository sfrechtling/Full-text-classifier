#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# 
# Date: 07.02.2013
# Strongly reengeneered
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

class ModelProcessor(object):
    """docstring for ModelProcessor"""
    def __init__(self, mfile):
        
        learners_object = pickle.load(open(mfile, "rb"))
        trans_file = learners_object["trans_file"]

        self.trans = pickle.load(open(trans_file, "rb"))
        self.learners = learners_object["learners"]
        self.scaler = learners_object["scaler"]

    def output_results(self, tokenized_text):
        """ Output prediction results for sample """
        vec = self.trans.get_sample_vector(tokenized_text)
        if self.scaler:
            vec = self.scaler.transform(vec)
        agr = 0
        pos = 0
        for k, learner in enumerate(self.learners):
            if k == 5 and vec.__class__ == sp.sparse.csr.csr_matrix:
            # Last learner (boosting) only supports dense data
                vec = vec.todense()
            if learner.predict(vec)[0] > 0:
                pr = "pos"
                agr += 1
                pos += 1
            else:
                pr = "neg"
                agr -= 1
            print "{:<60} : [ {:>3} ]".format(learner_names[k], pr)
        print "{:<60} : [ {:>1}/{:>1} ]".format("V O T E S", pos, 6-pos)
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


    def mk_report(self, tokenized_text, model_file_name, sample_file_name):
        """ """
        bag = self.trans._get_bag(tokenized_text)
        report_file_name = '.'.join(['report', model_file_name, sample_file_name, 'csv'])
        f = open(report_file_name, "wb")
        csv = UnicodeWriter(f, delimiter = CSV_DELIMITER_WRITE)
        # Probably need to write some info about transformation !!!!
        csv.writerow(["Model file:", repr(model_file_name)])
        csv.writerow(["Sample file:", repr(sample_file_name)])

        csv.writerow(["Dictionary length:", repr(self.trans.N)])
        csv.writerow(["Number of words from dictionary in document:",
            repr(len(bag))])
        csv.writerow([])
        csv.writerow(["Word", "count", "Argument for", "pos count (dict)",
            "neg count (dict)", "pos idf", "neg idf"])
        for (w, ipos, ineg, cpos, cneg) in self.trans.dwords:
            if w in [x for (x, c) in bag]:
                c = [c for (x, c) in bag if x==w][0]
                tend = "pos"
                if cpos < cneg:
                    tend = "neg"
                csv.writerow([w, repr(c), tend, repr(cpos), repr(cneg),
                    repr(ipos), repr(ineg)])
        f.close()



        

def main():
    parser = argparse.ArgumentParser(
        description="Make prediction")
    parser.add_argument("files",
        metavar="File",
        type=str,
        nargs='+',
        help="File with prediction model obtained from learn.py (.pickle) or file\
            with sample text (.pdf, .txt or .html). There should be at least one\
            model and at least one sample file.")
    parser.add_argument("--xpath", action="store",
        help="XPath for html file if necessary")
    parser.add_argument("--report", action="store_true",
        help="Save words report against dictionary as report.csv")

    args = parser.parse_args()
    file_names = args.files

    models = []
    samples = []

    for fn in file_names:
        try:
            ext = search(r"\A.*\.(.+)\Z", fn.strip()).group(1).lower()
        except AttributeError:
            print "Can not understand file format (%s) - no extension" % fn
            return

        if ext in ["pdf", "htm", "html", "text", "txt"]:
            samples.append(fn)
        elif ext == "pickle":
            models.append(fn)
        else:
            print '%s: unsupported file type. Sample files should have one of\
                the following extentions: "pdf", "htm", "html", "text", "txt";\
                model files should have "pickle" extension!' % fn
            return

    print "Loading models"
    model_processors = [ModelProcessor(model) for model in models]
    print "Done"

    for sfile in samples:
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
        tt = []
        for sent_num, sent in enumerate(sent_tokens):
            tokens = [token_filter(t) for t in word_tokenize(sent) if
                token_filter(t)]
            tt.append(tokens)

        for mp, model in zip(model_processors, models):
            print "  PREDICTION  ::  Sample: %s x Model: %s" % (sfile, model)
            mp.output_results(tt)

            if args.report:
                mp.mk_report(tt, model, sfile)


if __name__ == "__main__":
    main()
