#python imports
import sys
import os
import math
import random as r
import numpy
from collections import OrderedDict
import cv2
import numpy as np
from abc import ABC, abstractmethod

#project imports
import addresses as a
import nintaco

class Backend(ABC):
    def __init__(self):
        #######
        ##
        ##  The following properties are specific to the Nintaco API.
        ##  These are pretty static.
        ##
        #######
        nintaco.initRemoteAPI("localhost", 9999)
        self.api = nintaco.getAPI()
        self.api.addFrameListener(self.renderFinished)
        self.api.addStatusListener(self.statusChanged)
        self.api.addActivateListener(self.apiEnabled)
        self.api.addDeactivateListener(self.apiDisabled)
        self.api.addStopListener(self.dispose)

    ################################################################################
    ##################### THESE ARE FOR DEBUGGING RAM ADDRESSES.
    ################################################################################
    def get_address_list(self):
        add_list = {}
        for i in range(27557):
            add_list.append

    def initialize_ram(self):
        temp = []
        for i in range(0, 1919):
            temp.append(self.get_address_value(i))
        self.ram = temp

    def ram_mapper(self):
        print("New action:")
        for i in range(0,1919):
            new_val = self.get_address_value(i)
            old_val = self.ram[i]
            if (new_val != old_val):
                print(hex(i) + ": " + str(old_val) + " -> " + str(new_val) + ".")
                self.ram[i] = new_val
        print("\n\n")

    def specific_array(self):
        temp = {}
        addresses = [0x034a, 0x034b, 0x034c, 0x034d, 0x034e, 0x034f]
        for address in addresses:
            temp[address] = self.get_address_value(address)
        self.ram = temp

    def specific_mapper(self):
        print("New action:")
        addresses = list(self.ram.keys())
        for address in addresses:
            new_val = self.get_address_value(address)
            old_val = self.ram[address]
            #if (new_val != old_val):
            print(hex(address) + ": " + str(old_val) + " -> " + str(new_val) + ".")
            self.ram[address] = new_val
        print("\n\n")

    def try_write(self):
        self.api.writeCPU(a.map_current_room,r.randint(0,400))
        print("Current room: " + str(self.get_address_value(a.map_current_room)))

    def get_address_value(self, address):
        return self.api.peekCPU(address)

    ################################################################################
    ##################### THESE ARE FOR NON-NINTACO METHODS.
    ################################################################################
    def image_work(self, i):
        path = './screenshots/'
        attempts = 0
        #debug logic for screenshots
        #TODO: find nintaco source and recompile with custom screenshot names...
        screenName = None
        i = i - 1
        if (i < 10):
            screenName = 'zelda-00' + str(i)
        elif (i < 100):
            screenName = 'zelda-0' + str(i)
        else:
            screenName = 'zelda-' + str(i)

        while (attempts < 3):
            try:
                image = cv2.imread(path + screenName + '.png',0)

                if image is None:
                    exit()
                #gray = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
                view = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                view = view[56:224, 0:255]
                cv2.putText(
                        view,                     #numpy array image
                        str(i),                    #text
                        (10,50),                   #position
                        cv2.FONT_HERSHEY_SIMPLEX,  #font
                        1,                         #size
                        (66, 227, 245),         #color
                        3)                         #font stroke

                #DEBUGGING PURPOSES - remove?
                cv2.imwrite(path + 'view' + str(i) + '.png', view)
                return view
            except Exception as e:
                attempts += 1
        return None
    
    def getScreen(self):
        screen = np.ndarray((61440,), int)
        self.api.getPixels(screen)
        screen = screen.reshape(240,256)
        screen = screen[56:240, 0:255]
        screen = screen.astype(np.uint8)
        return screen
    
    def pressButton(self, action):
        if action != None:
            self.api.writeGamepad(0, action, True)

    def releaseButton(self, action):
        if action != None:
            self.api.writeGamepad(0, action, False)

    ################################################################################
    ##################### NINTACO SPECIFIC METHODS
    ################################################################################
    def apiEnabled(self):
        print("API enabled")

    def apiDisabled(self):
        print("API disabled")
  
    def dispose(self):
        print("API stopped")
  
    def statusChanged(self, message):
        print("Status message: %s" % message)
