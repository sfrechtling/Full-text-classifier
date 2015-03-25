#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# Author: Alexice
# Date: 21.12.2011
#
# csv/xls file reading functionality
# pdf password removing

import csv
import codecs
import cStringIO
import xlrd
import re
import datetime
import subprocess
import hashlib
import math
from app_config import *


class UTF8Recoder:
    """
    Iterator that reads an encoded stream and reencodes the input to UTF-8
    """

    def __init__(self, f, encoding):
        self.reader = codecs.getreader(encoding)(f)

    def __iter__(self):
        return self

    def next(self):
        return self.reader.next().encode("utf-8")


class UnicodeReader:
    """
    A CSV reader which will iterate over lines in the CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        f = UTF8Recoder(f, encoding)
        self.reader = csv.reader(f, dialect=dialect, **kwds)

    def next(self):
        row = self.reader.next()
        return [unicode(s, "utf-8") for s in row]

    def __iter__(self):
        return self


class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)
# End

def csv_normalize(x):
    """
    Helper for csv cells type conversion
    """

    try:
        return datetime.datetime.strptime(x, CSV_DATE_PATTERN)
    except ValueError:
        pass
    try:
        res = float(x)
        if math.isnan(res):
            raise ValueError
        return res
    except ValueError:
        return x.strip()


def xls_normalize(x):
    """
    Helper for xls cells date conversion
    """

    if x.ctype == xlrd.XL_CELL_DATE:
        dtt = xlrd.xldate_as_tuple(x.value, datemode=XLS_DATE_MODE)
        return datetime.datetime(*dtt)
    else:
        try:
            return x.value.strip()
        except AttributeError:
            return x.value


class FileByRow(object):
    """
    Class for reading by row csv and xls files.
    If header=True it is considered that first row contains colum names.
    """

    def __init__(self, file_name, header=True):
        self.table_header = None
        if re.search(r"\.xls\Z", file_name):
            self.file_type = "XLS"
            reader = xlrd.open_workbook(file_name)
            self.first_sheet = reader.sheets()[0]

            if header:
                self.table_header = [s.value for s in self.first_sheet.row(0)]
                self.row_num = 0
            else:
                self.row_num = -1
        elif re.search(r"\.csv\Z", file_name):
            self.file_type = "CSV"
            csvfile = open(file_name, "rb")
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
            csvfile.seek(0)
            self.reader = UnicodeReader(csvfile, dialect,
                delimiter=CSV_DELIMITER_READ)
            if header:
                try:
                    self.table_header = self.reader.next()
                except StopIteration:
                    raise Exception("Wrong csv structure: %s" % file_name)

        else:
            raise Exception("Unsupported file type")

    def next(self):
        if self.file_type == "XLS":
            try:
                self.row_num += 1
                return [xls_normalize(s)
                    for s in self.first_sheet.row(self.row_num)]
            except IndexError:
                raise StopIteration

        elif self.file_type == "CSV":
            row = self.reader.next()
            return [csv_normalize(cell) for cell in row]

        else:
            # Ups, have no idea how we could get here
            pass

    def __iter__(self):
        return self


def remove_pdf_password(file_orig_path, file_dest_path):
    """
    Runs shell command:
    gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile=unencrypted.pdf
    -c .setpdfwrite -f encrypted.pdf
    """

    cmd = "%s -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite\
        -sOutputFile=%s -c .setpdfwrite -f %s" %\
        (GS_PATH, file_dest_path, file_orig_path)
    subprocess.check_call(cmd, shell=True)


def md5_for_file(f, block_size=2**16):
    """
    Calculates md5 for file
    """

    md5 = hashlib.md5()
    while True:
        data = f.read(block_size)
        if not data:
            break
        md5.update(data)
    return md5.hexdigest()
