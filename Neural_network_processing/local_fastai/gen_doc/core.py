from ..core import *
import re

def strip_local_fastai(s):  return re.sub(r'^local_fastai\.', '', s)

