import atexit
import numpy as np 
import random
import utils
from PIL import Image

import nintaco
import addresses as a

import eval as e
import controller as c

nintaco.initRemoteAPI("localhost", 9999)
api = nintaco.getAPI()

api.run()