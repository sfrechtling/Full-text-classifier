#
# Just imports and logging initialization for tests
#
import logging

# Setup logging
LOGGER_NAME = "app_libs"
hdlr = logging.FileHandler("./log_test.log")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
hdlr.setFormatter(formatter)
logging.getLogger(LOGGER_NAME).addHandler(hdlr)
logging.getLogger(LOGGER_NAME).setLevel(logging.DEBUG)
