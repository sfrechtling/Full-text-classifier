#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# 
# Date: 26.12.2011
#
# Preliminary version of setup

import os
import re
import datetime as d

from nltk.corpus import movie_reviews
from nltk import download

from app_libs.app_config import *
from app_libs.file_proc import UnicodeWriter
from app_libs.db_proc import mk_tables
from app_libs.app_config import *


def mk_csv():
    headers = ["Code", "Date", "Class Label", "Error Cost", "Success Gain",
        "Announcement"]

    l = movie_reviews.abspaths()
    pos_date = d.date(1990, 1, 1)
    neg_date = d.date(1990, 1, 2)
    neg_cnt = 0
    pos_cnt = 0

    f = open("MOVIE.csv", "wb")
    csv = UnicodeWriter(f, delimiter=CSV_DELIMITER_WRITE)
    csv.writerow(headers)

    for p in l:
        symbol = "MOVIE"
        if re.search(r"neg", p):
            if neg_cnt < 100:
                symbol = "MOVIE_SHORT"

            label = "neg"
            neg_cnt += 1
            pos_date = pos_date + d.timedelta(2)
            od = pos_date.strftime(CSV_DATE_PATTERN)
        else:
            if pos_cnt < 100:
                symbol = "MOVIE_SHORT"

            label = "pos"
            pos_cnt += 1
            neg_date = neg_date + d.timedelta(2)
            od = neg_date.strftime(CSV_DATE_PATTERN)

        csv.writerow([symbol, od, label, "1", "1", p.path])
    f.close()


def main():
    for i in range(10):
        for j in range(10):
            try:
                os.makedirs(os.path.join(FILES_PATH, str(i), str(j)))
            except os.error:
                pass

    mk_tables()
    download("punkt")
    download("wordnet")
    download("stopwords")
    download("movie_reviews")
    mk_csv()

if __name__ == "__main__":
    main()
