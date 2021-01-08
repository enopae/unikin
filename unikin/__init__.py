# This file is part of unikin.
#
# Copyright (C) 2021 ETH Zurich, Eno Paenurk
#
# unikin is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# unikin is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with unikin. If not, see <https://www.gnu.org/licenses/>.

# Initialize an empty list of methods to export
__all__ = []

def export(defn):
    """
    Define a decorator to export definitions from modules
    """
    globals()[defn.__name__] = defn
    __all__.append(defn.__name__)
    return defn

from functools import wraps
from time import time

# https://stackoverflow.com/questions/1622943
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r  took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap

# Import methods
from .models import *
from .rate import *
from .constants import *
from .opt import *
from .statecount import *
from .misc import *
from .main import *
from .sigmasim import *
