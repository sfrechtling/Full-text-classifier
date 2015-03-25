#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# Author: Alexice
# Date: 23.12.2011
#
#

import os
from logging import getLogger

from sqlobject import *

from app_config import *
import my_logging

log = getLogger(__name__)


class AnT(SQLObject):
    symbol = StringCol(length=20)
    class_label = StringCol() # may be changed
    error_cost = FloatCol()
    success_gain = FloatCol()
    associated_date = DateCol()

    text_tokenized = PickleCol(default=None) # List of lists
    sentences = IntCol(default=0)
    tokens = IntCol(default=0)

    update_time = DateTimeCol()


class Announcement(SQLObject):
    symbol = StringCol(length=20)
    class_label = StringCol() # may be changed
    error_cost = FloatCol()
    success_gain = FloatCol()
    associated_date = DateCol()

    announcement_original_url = StringCol()
    announcement_path_saved = StringCol()
    announcement_file_type = StringCol(length=20)
    announcement_use_saved = BoolCol()
    orig_md5 = StringCol(length=32)

    insert_time = DateTimeCol()
    update_time = DateTimeCol()

    xpath = StringCol(default=None) # Only for html files

    tokenization = ForeignKey('AnT', default=None, cascade='null')


def mk_tables():
    start_db()
    if not Announcement.tableExists():
        log.debug("Creating table for announcements")
        Announcement.createTable()
    if not AnT.tableExists():
        log.debug("Creating table for tokenized announcements")
        AnT.createTable()


def start_db():
    log.debug("Starting DB session")
    if DB_TYPE == "sqlite":
        db_filename = os.path.abspath(DB_PATH)
        connection_string = "sqlite:" + db_filename
    elif DB_TYPE == "mysql":
        raise Exception("mysql: Unsupported DB type")
        connection_string = "mysql://root:@localhost/gf"
    else:
        raise Exception("Unsupported DB type")
    connection = connectionForURI(connection_string)
    sqlhub.processConnection = connection


def stop_db():
    log.debug("Closing DB session")
    sqlhub.processConnection.close()
