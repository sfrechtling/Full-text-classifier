#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# Author: Alexice
# Date: 23.12.2011
#
# Pdf to text conversion

import sys
import cStringIO
import re

from lxml.html import parse
from lxml import etree
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfinterp import process_pdf
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfparser import PDFDocument, PDFParser
from pdfminer.pdfdevice import PDFDevice
from pdfminer.cmapdb import CMapDB
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus.reader.wordnet import POS_LIST


lmtzr = WordNetLemmatizer()


def token_filter(token):

    def lemmatize(s):
        if not s:
            return None
        for pos in POS_LIST:
            ns = lmtzr.lemmatize(s, pos=pos)
            if s != ns:
                return ns
        return s

    def compose(func_list):
        if len(func_list) > 0:
            f = func_list[0]
            tail = func_list[1:]
            if tail != []:
                return lambda x: f(compose(tail)(x))
            else:
                return f
        else:
            return lambda x: x

    def _str(token):
        """
        Only ASCI chars
        """
        if token is None:
            return None
        try:
            return str(token).lower()
        except UnicodeEncodeError:
            return None

    def _nt_to_not(token):
        if token and token == "n't":
            return "not"
        return token

    def _not_punctuation(token):
        punc_expr = r"[!?:;,._@\'\^\\\"/\[\]\(\)\{\}]"
        if token and not re.search(punc_expr, token):
            return token
        else:
            return None

    def _not_digits(token):
        if (token and re.match(r"\A\w+-?\w+\Z", token) and
            not re.search(r"[\d]", token)):
            return token
        return None

    #return compose([_str])(token)
    return compose([lemmatize, _not_digits, _nt_to_not, _str])(token)


def html2text(path, xpath=None):
    """
    Converts html to text. Removes script text as it is code.
    <!!!> Maybe should add <br>, </p> line brakes
    """

    def _remove_script(el):
        for sub_el in el.getchildren():
            if sub_el.tag == "script":
                sub_el.text = ""
            else:
                _remove_script(sub_el)
        return
    try:
        e = parse(path).getroot()
    except IOError:
        raise IOError
    _remove_script(e)
    if xpath is not None and xpath != "":
        try:
            e = e.xpath(xpath)[0]
        except (etree.XPathEvalError, IndexError, TypeError):
            return ""
    return re.sub("\u2019", "'", e.text_content()) # Fix '


def pdf2text(path, debug=0):
    # Set debug level
    CMapDB.debug = debug
    PDFResourceManager.debug = debug
    PDFDocument.debug = debug
    PDFParser.debug = debug
    PDFPageInterpreter.debug = debug
    PDFDevice.debug = debug

    # input option
    password = ""
    pagenos = set()
    maxpages = 0
    # output option
    codec = "utf-8"
    laparams = LAParams()

    rsrcmgr = PDFResourceManager()
    device = TextConverter(rsrcmgr,
        outfp=cStringIO.StringIO(), codec=codec, laparams=laparams)

    fp = file(path, "rb")
    process_pdf(rsrcmgr, device, fp, pagenos,
        maxpages=maxpages, password=password)
    fp.close()
    return re.sub(u"\xca\xbc", "'", device.outfp.getvalue()) # Fix '
