#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# 
# Date: 30.12.2011
#
#

import os
import argparse
import ConfigParser
import cPickle as pickle
from cStringIO import StringIO
from datetime import datetime

from sqlobject import AND, OR, IN
from scipy.sparse import *
from scipy import *
import numpy as np

from app_libs.app_config import *
from app_libs.db_proc import start_db, stop_db, AnT
from app_libs.sample_transformation import Transformation, TransError
from app_libs.sample_transformation import tr_save, tr_load
from app_libs.file_proc import UnicodeWriter


def get_query(start_date, end_date, symbols, min_sentences):
    if symbols != []:
        sq = AnT.select(
            AND(AnT.q.associated_date<=end_date,
                AnT.q.associated_date>=start_date,
                AnT.q.sentences>=min_sentences,
                OR(AnT.q.class_label=="pos",
                AnT.q.class_label=="neg"),
                IN(AnT.q.symbol, symbols),
                ))
    else:
        sq = AnT.select(
            AND(AnT.q.associated_date<=end_date,
                AnT.q.associated_date>=start_date,
                AnT.q.sentences>=min_sentences,
                OR(AnT.q.class_label=="pos",
                AnT.q.class_label=="neg"),
                ))
    return sq


def get_trans_opts(config, output=True):
    opts = {
        "feature_selection_subset": ["ratio", config.getfloat, 1.0],
        "consider_only_presence": ["boolean", config.getboolean, False],
        "consider_negations": ["neg", config.getboolean, False],
        "filter_stopwords": ["stop", config.getboolean, False],
        "min_sentences": ["min_sent", config.getint, 1],
        "min_token_len": ["min_tlen", config.getint, 3],
        "rare_tokens_threshold": ["min_tokens", config.getint, 3],
        "tfidf": ["tfidf", config.getboolean, True],
        "reduction": ["reduction", config.getboolean, False],
        "components": ["components", config.getint, 100],
        "normalize": ["normalize", config.getboolean, False],
        "top_words": ["dict_cut", config.getint, 0],
    }
    trans_opts = {}
    for k, v in opts.iteritems():
        opts_key = v[0]
        call = v[1]
        default_value = v[2]
        try:
            trans_opts[opts_key] = call("Transformations and reductions", k)
        except ConfigParser.NoSectionError:
            print "Error: there should be a section 'Transformations and \
reductions' in configuration file"
            return None
        except (ConfigParser.NoOptionError, ValueError):
            print k
            trans_opts[opts_key] = default_value

    return trans_opts


def main():
    parser = argparse.ArgumentParser(
        description="Prepare samples for training and testing")
    parser.add_argument("config_file",
        metavar="ConfigFile",
        type=str,
        nargs=1,
        help="File with sample preparation options.\
        See documentation for details")
    parser.add_argument("--replace", action="store_true",
        help="Replace output files with newly generated if there are any\
            name colission")
    parser.add_argument("--samples-file", action="store",
        help="Override config samples file name to save samples")
    parser.add_argument("--transformation-file-save", action="store",
        help="Override config transformaton file name to save transformation")
    parser.add_argument("--transformation-file-load", action="store",
        help="Override config transformaton file name to load transformation")
    parser.add_argument("--csv-report", action="store",
        help="Save transformation report in CSV-file (name\
            should be provided)")

    args = parser.parse_args()

    config = ConfigParser.SafeConfigParser()
    try:
        config.readfp(open(args.config_file[0]))
    except (IOError, TypeError):
        print "Error: No such file %s" % args.config_file[0]
        return

    #
    # Parsing 'General' section
    #
    try:
        start_date_str = config.get("General", "start_date")
        start_date = datetime.strptime(start_date_str, CONFIG_DATE_PATTERN)
        end_date_str = config.get("General", "end_date")
        end_date = datetime.strptime(end_date_str, CONFIG_DATE_PATTERN)
    except ValueError:
        print "Error: something wrong with start or/and end format"
        print start_date_str
        return
    except ConfigParser.NoSectionError:
        print "Error: there should be a section 'General' in \
configuration file"
        return
    except ConfigParser.NoOptionError:
        print "Error: there should be both start_date and end_date in \
configuration file"
        return
    try:
        symbols = config.get("General", "symbols").split()
    except ConfigParser.NoOptionError:
        symbols = []

    # Transformation file to load
    if args.transformation_file_load:
        load_trans = args.transformation_file_load
    else:
        try:
            load_trans = config.get("General",
                "use_transformation_from_file")
        except ConfigParser.NoOptionError:
            load_trans = None
        if load_trans and load_trans.lower() == 'no':
            load_trans = None
    if load_trans and not os.path.exists(load_trans):
        print "Error: no transformation file: %s" % load_trans
        return

    # Transformation file to save
    if args.transformation_file_save:
        save_trans = args.transformation_file_save
    else:
        try:
            save_trans = config.get("General",
                "save_transformation_file_as")
        except ConfigParser.NoOptionError:
            save_trans = None
        if save_trans and save_trans.lower() == 'no':
            save_trans = None
    if not save_trans and not load_trans:
        print "WARNING: No transformation file to save specified"
    elif save_trans and os.path.exists(save_trans) and not args.replace:
        print("Error: file %s already exists. Use --replace option to \
replace or change file name" % save_trans)
        return
    else:
        pass

    # Samples file to save
    if args.samples_file:
        save_samples = args.samples_file
    else:
        try:
            save_samples = config.get("General", "save_samples_file_as")
        except ConfigParser.NoOptionError:
            print "Error: file name to save samples should be specified \
ether in config or by --samples_file option"
            return
    if os.path.exists(save_samples) and not args.replace:
        print "Error: file %s already exists. Use --replace option to \
replace or change file name" % save_samples
        return

    start_db()

    if load_trans:
        # Using transformation from file
        transformation = tr_load(load_trans)
        min_sentences = transformation.opts["min_sent"]
        sq = get_query(start_date, end_date, symbols, min_sentences)
    else:
        # Creating new transformation
        print "Creating transformation..."
        trans_opts = get_trans_opts(config)
        min_sentences = trans_opts["min_sent"]
        sq = get_query(start_date, end_date, symbols, min_sentences)
        transformation = Transformation(sq, trans_opts=trans_opts)
        if save_trans:
            print "Saving transformation..."
            tr_save(save_trans, transformation)
            print "Done"

    if args.csv_report:
        if os.path.exists(args.csv_report) and not args.replace:
            print "%s exists, use --replace to force replacement" %\
                args.csv_report
        else:
            # Saving CSV
            f = open(args.csv_report, "wb")
            csv = UnicodeWriter(f, delimiter=CSV_DELIMITER_WRITE)
            csv.writerow(["Dictionary length:", repr(transformation.N)])
            csv.writerow(["Documents processed:", repr(transformation.n_docs)])
            csv.writerow([])
            csv.writerow(["Word", "Argument for", "pos count", "neg count",
                "pos idf", "neg idf"])
            for (w, ipos, ineg, cpos, cneg) in transformation.dwords:
                tend = "pos"
                if cpos < cneg:
                    tend = "neg"
                csv.writerow([w, tend, repr(cpos), repr(cneg),
                    repr(ipos), repr(ineg)])
            f.close()

    # Making samples
    transformation_file = save_trans or load_trans
    if not transformation_file:
        print "No transformation file. Exit"
        return

    print "Preparing samples..."
    samples_object = {}
    samples_object["matrix"] = transformation.get_term_document_matrix(sq)
    samples_object["sample_ids"] = [an.id for an in sq]
    samples_object["trans_file"] = transformation_file
    samples_object["trans_sample_ids"] = transformation.ids

    print "Saving samples..."
    pickle.dump(samples_object, open(save_samples, "wb"))
    print "Done"
    return

if __name__ == "__main__":
    main()
