#!/usr/bin/env python
# -*- coding: utf8 -*-
#
#
# Date: 24.12.2011
#
#

import sys
import os
import argparse
import datetime
from random import randint
from re import search
import shutil
import logging

from nltk.tokenize import word_tokenize, sent_tokenize
from sqlobject import AND, OR, IN
from sqlobject.main import SQLObjectNotFound, SQLObjectIntegrityError

from app_libs.file_proc import FileByRow, remove_pdf_password, md5_for_file
from app_libs.db_proc import start_db, stop_db, Announcement, AnT
from app_libs.app_config import *
from app_libs.text_proc import html2text, pdf2text, token_filter


# Setup logging
LOGGER_NAME = "PREPARE"
hdlr = logging.FileHandler("./prepare.log")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
hdlr.setFormatter(formatter)
logging.getLogger(LOGGER_NAME).addHandler(hdlr)
logging.getLogger(LOGGER_NAME).setLevel(logging.DEBUG)

log = logging.getLogger("PREPARE")

headers = ["Code", "Date", "Class Label", "Error Cost", "Success Gain",
    "Announcement"]


def process_web_announcement_file(data, purl, xpath,
        save_local=False, modify=False):
    """
    Obtain file by web
    Remove password and copy file to local storage
    <Not Implemented>
    """
    return " WARNING: Unsupported schema. Skiped"


def process_local_announcement_file(data, path, xpath,
        save_local=False, modify=False):
    """
    Remove password (if pdf) and copy file to local storage (if needed)
    Save info about fie in DB
    """
    ret = ""
    try:
        f = open(path, "rb")
    except IOError:
        return " WARNING: No such file. Skipped"
    file_md5 = md5_for_file(f)
    try:
        ext = search(r".+\.(.+)\Z", path.strip()).group(1).lower()
    except (IndexError, AttributeError):
        ext = ""
    #print ext
    if ext == "txt" or ext == "text":
        ftype = "txt"
    elif ext == "htm" or ext =="html":
        ftype = "html"
    elif ext == "pdf":
        ftype = "pdf"
    else:
        ftype = "unknown"
    #print ext
    sq = Announcement.select(Announcement.q.orig_md5==file_md5)
    if list(sq) != []:
        ann = sq[0]
        if not modify:
            return " WARNING: Record already exists. Skiped."
        else:
            ann.set(
                announcement_original_url=path,
                symbol=data["Code"],
                associated_date=data["Date"],
                class_label=data["Class Label"],
                error_cost=data["Error Cost"],
                success_gain=data["Success Gain"],
                #close=data["Close"],
                #f1_day=data["1day"],
                #f1_day_abnormal=data["1 Day Abnormal"],
                update_time=datetime.datetime.now(),
            )
            local_path = ann.announcement_path_saved
            ret = " INFO: Record already exists. Modified."
    else:
        if save_local:
            local_path = os.path.join(FILES_PATH,
                str(randint(0, 9)), str(randint(0, 9)), file_md5 + "." + ftype)
        else:
            local_path = path

        ann = Announcement(announcement_original_url=path,
            announcement_path_saved=local_path,
            announcement_file_type=ftype,
            announcement_use_saved=save_local,
            orig_md5=file_md5,
            insert_time=datetime.datetime.now(),

            symbol=data["Code"],
            associated_date=data["Date"],
            class_label=data["Class Label"],
            error_cost=data["Error Cost"],
            success_gain=data["Success Gain"],
            #close=data["Close"],
            #f1_day=data["1day"],
            #f1_day_abnormal=data["1 Day Abnormal"],
            update_time=datetime.datetime.now(),
        )
    if save_local:
        if ftype == "pdf":
            remove_pdf_password(path, local_path)
        else:
            shutil.copyfile(path, local_path)
        ret += " File (re)saved localy."
    return ret


def label_record(data, path, xpath):
    sq = Announcement.select(Announcement.q.announcement_original_url==path)
    if list(sq) != []:
        ann = sq[0]
        ann.set(
            class_label=data["Class Label"],
            error_cost=data["Error Cost"],
            success_gain=data["Success Gain"],
            update_time=datetime.datetime.now(),
        )
        if ann.tokenization:
            ann.tokenization.set(
                class_label=data["Class Label"],
                error_cost=data["Error Cost"],
                success_gain=data["Success Gain"],
                update_time=datetime.datetime.now(),
            )
        if xpath and ann.announcement_file_type == "html":
            ann.xpath = xpath
            if ann.tokenization:
                AnT.delete(ann.tokenization.id)
            # Remove tokenizations
        return " Modified"
    else:
        return " WARNING: No such record"


def tokenize_record(path, force=False, rec_id=None):
    if rec_id:
        ann = Announcement.get(rec_id)
    else:
        sq = Announcement.select(
            Announcement.q.announcement_original_url==path)
        if list(sq) != []:
            ann = sq[0]
        else:
            return " WARNING: No such record"
    if not force and ann.tokenization:
        return " WARNING: Already tokenized. Skipped"
    try:
        #print ann.announcement_path_saved
        if ann.announcement_file_type == "pdf":
            text = pdf2text(ann.announcement_path_saved)
        elif ann.announcement_file_type == "html":
            text = html2text(ann.announcement_path_saved, ann.xpath)
        elif ann.announcement_file_type == "txt":
            f = open(ann.announcement_path_saved, "r")
            text = f.read()
        else:
            return " WARNING: Unsupported file type"
    except IOError:
        return " WARNING: Missed file: %s" % ann.announcement_path_saved

    sent_tokens = sent_tokenize(text)
    res = []
    tokens_num = 0
    for sent_num, sent in enumerate(sent_tokens):
        tokens = [token_filter(t) for t in word_tokenize(sent) if
            token_filter(t)]
        tokens_num += len(tokens)
        res.append(tokens)
    sent_num += 1
    sqt = AnT.select(AND(AnT.q.symbol==ann.symbol,
        AnT.q.associated_date==ann.associated_date))
    if list(sqt) != []:
        annT=sqt[0]
        res.extend(annT.text_tokenized)
        sent_num += annT.sentences
        tokens_num += annT.tokens
        annT.set(symbol=ann.symbol,
            class_label=ann.class_label,
            error_cost=ann.error_cost,
            success_gain=ann.success_gain,
            update_time=datetime.datetime.now(),
            text_tokenized=res,
            sentences=sent_num,
            tokens=tokens_num,
        )
    else:
        annT=AnT(symbol=ann.symbol,
            associated_date=ann.associated_date,
            class_label=ann.class_label,
            error_cost=ann.error_cost,
            success_gain=ann.success_gain,
            update_time=datetime.datetime.now(),
            text_tokenized=res,
            sentences=sent_num,
            tokens=tokens_num,
        )
    ann.set(tokenization=annT)
    return " Tokenized"


def process_input_file(fname, total=0, check=False, tokenize_it=False,
        just_label=False, save_local=False, modify=False, xpath=""):
    """
    Process input csv/xls file with anouncement records.
    """
    try:
        rows = FileByRow(fname)
    except IOError:
        print "No such file: %s" % fname
        return (None, None, None)
    for i, h in enumerate(headers):
        if (not hasattr(rows, "table_header") or
            len(rows.table_header)<len(headers) or
            h != rows.table_header[i].strip()):
            print "Wrong data structure, use data with column names: %s" %\
                str(headers)
            return (None, None, None)
    web_count = 0
    local_count = 0
    out = ""
    for rn, row in enumerate(rows):
        out = ""
        data = {}
        data["Announcement"] = []
        for i, c in enumerate(row):
            if i<len(headers)-1:
                data[headers[i]] = c
            else:
                if c != "":
                    data["Announcement"].append(c)
        if data["Announcement"] == []:
            pass # Nothing to do
        else:
            for an in data["Announcement"]:
                processor = process_local_announcement_file
                local_count += 1
                if total:
                    print "Rec: %4d/%d" % (web_count+local_count, total)
                if not check and not just_label:
                    out += "Rec: %4d/%d Path:%s" %\
                        (web_count+local_count, total, an)
                    out += processor(data, an, xpath, save_local, modify)
                if just_label:
                    out += "Record %4d/%d Path:%s" %\
                        (web_count+local_count, total, an)
                    out += label_record(data, an, xpath)
                if tokenize_it:
                    out += tokenize_record(an, force=modify)
        if out != "":
            log.info(out)

    total_count = web_count + local_count
    return (total_count, web_count, local_count)


def import_command(args):
    for f in args.table_files:
        # Calculating number of records
        (total, web, local) = process_input_file(f, check=True)
        if total is None:
            print "Error processing file %s" % f
        elif not args.check:
            process_input_file(f, total=total, check=False,
                save_local=args.no_copy, modify=args.modify,
                xpath=args.xpath, tokenize_it=args.tokenize_it)
        else:
            print "Records count web:%d local:%d" % (web, local)


def label_command(args):
    for f in args.table_files:
        # Calculating number of records
        (total, web, local) = process_input_file(f, check=True)
        if total is None:
            print "Error processing file %s" % f
        elif not args.check:
            process_input_file(f, total=total, check=False,
                save_local=False, modify=False, just_label=True,
                xpath=args.xpath, tokenize_it=False)
        else:
            print "Records count web:%d local:%d" % (web, local)


def tokenize_command(args):
    if args.force:
        if args.symbols is None or args.symbols == []:
            al = Announcement.select()
        else:
            al = Announcement.select(IN(Announcement.q.symbol, args.symbols))
        # Remove tokenizations
        for rec in al:
            if rec.tokenization is not None:
                AnT.delete(rec.tokenization.id)
    else:
        if args.symbols is None or args.symbols == []:
            al = Announcement.select(Announcement.q.tokenization==None)
        else:
            al = Announcement.select(AND(Announcement.q.tokenization==None,
                IN(Announcement.q.symbol, args.symbols)))
    total = al.count()
    for i, rec in enumerate(al):
        res = tokenize_record(None, force=args.force, rec_id=rec.id)
        out = "Rec: %4d/%d sentences: %d tokens: %d. %s" % (i+1, total,
            rec.tokenization.sentences, rec.tokenization.tokens, res)
        print out
        log.info(out)
    print "%d records are tokenized" % total


def main():
    parser = argparse.ArgumentParser(prog="prepare.py",
        description="Input data processor.")
    subparsers = parser.add_subparsers(help="sub-command help")

    parser_import = subparsers.add_parser("import",
        help="Import data from CSV/XLS",
        description="""
            This command stores information about data in DB for
            further processing. Removes pdf passwords. Moves files to the
            application local folder.
            """)
    parser_import.add_argument("table_files",
        metavar="TABLEfile",
        type=str,
        nargs="+",
        help="CSV or XLS file with information, row format: %s, use \
            these column names as first row in a file" % headers)
    parser_import.add_argument("--check",
        action="store_true",
        help="Just make check of table file structure not actualy save\
            somthing in db or obtain any files by web")
    parser_import.add_argument("--no-copy",
        action="store_false",
        help="""
            Do not copy files with announcements to application data folder
            (ignored if --check option is selected). You should
            not use this option if your announcement files are password
            protected pdfs""")
    parser_import.add_argument("--modify",
        action="store_true",
        help="If record with the same announcement already exists modify \
            record with new data.")
    parser_import.add_argument("--xpath",
        action="store",
        default="",
        help="""
            Specify xpath of text block in html files if necessary.
            Otherwise all text in html (except tags and scripts) will be
            used. Have no effect on pdf or text files.""")
    parser_import.add_argument("--tokenize-it",
        action="store_true",
        help="Tokenize each text just after importing. You may run \
            tokenization later using 'tokenize' command")
    parser_import.set_defaults(func=import_command)

    parser_label = subparsers.add_parser("label",
        help="Label imported data",
        description="""
            Label data and/or change Error Cost and Success Gain. Use same file
            structure as for 'import' command. Announcement URIs are used as
            keys. This command runs much faster than import with --modify flag
            as it doesn't check md5 for source files and actually doesn't even
            need source files to be at their original places -- only URIs
            should correspond to those saved in DB.""")
    parser_label.add_argument("table_files",
        metavar="TABLEfile",
        type=str,
        nargs="+",
        help="CSV or XLS file with information, row format: %s, use \
            these column names as first row in a file" % headers)
    parser_label.add_argument("--check",
        action="store_true",
        help="Just make check of table file structure not actualy change\
            somthing in db")
    parser_label.add_argument("--xpath",
        action="store",
        help="Set new XPath for html files")
    parser_label.set_defaults(func=label_command)

    parser_tokenize = subparsers.add_parser("tokenize",
        help="Tokenize texts from imported files",
        description="""
            Split texts to list of sentences which are list of tokens and save
            to DB.""")
    parser_tokenize.add_argument("--force",
        action="store_true",
        help="Retokenize records that were already tokenized")
    parser_tokenize.add_argument("--symbols",
        metavar="symbol",
        type=str,
        nargs="*",
        help="Symbols to process")

    parser_tokenize.set_defaults(func=tokenize_command)

    args = parser.parse_args()
    start_db()
    args.func(args)
    stop_db()

if __name__ == "__main__":
    main()
