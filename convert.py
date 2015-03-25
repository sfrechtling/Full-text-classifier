#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# 
# Date: 26.01.2012
#
#

import sys
import os
import argparse
import re

from pdfminer.pdfinterp import PDFTextExtractionNotAllowed

from app_libs.text_proc import html2text, pdf2text
from app_libs.file_proc import remove_pdf_password
from app_libs.app_config import *


def process_file(fdir, f):
    try:
        ext = re.search(r".+\.(.+)\Z", f.strip()).group(1).lower()
    except (IndexError, AttributeError):
        ext = ""
    if ext == "pdf":
        try:
            try:
                r = pdf2text(os.path.join(fdir, f))
            except PDFTextExtractionNotAllowed:
                remove_pdf_password(os.path.join(fdir, f), TMP_PDF)
                r = pdf2text(TMP_PDF)
                os.unlink(TMP_PDF)
            fd = open(os.path.join(fdir, f + ".txt"), "w")
            fd.write(r)
            fd.close()

            print "File: %s saved" % (os.path.join(fdir, f + ".txt"))
        except:
            print "File: %s error" % (os.path.join(fdir, f + ".txt"))

def main():
    parser = argparse.ArgumentParser(
        description="Convert pdf files to text. Places\
            converted files to the same directory")
    parser.add_argument("src_path",
        metavar="SourcePath",
        type=str,
        nargs=1,
        help="Directory or file path to be converted")
    args = parser.parse_args()
    path = args.src_path[0]

    if not os.path.exists(path):
        print "No such file or directory: %s" % path
        return
    fdir = os.path.dirname(path)
    if os.path.isdir(path):
        for f in os.listdir(path):
            if not os.path.isdir(os.path.join(fdir, f)):
                process_file(fdir, f)
    else:
        process_file(fdir, os.path.split(path)[1])

if __name__ == "__main__":
    main()
