import cv2
import numpy
import os
from PIL import Image
import addresses as a
#import methods as m
import random as r
import math
from collections import defaultdict
from backend import Backend as backend
import nintaco

class ZeldaInterface(backend):
    def __init___(self):
        print("Initializing...")
        super().__init__()
        self.debug = True
        self.current_state = None
        self.api.run()

    def get_address_value(self, address):
        return self.api.peekCPU(address)

    def load_inventory(self):
        if self.debug: print("load_inventory")
        inventory = {
            "Bow":              self.get_address_value(a.inv_has_bow),
            "Clock":            self.get_address_value(a.inv_has_clock),
            "Compass":          self.get_address_value(a.inv_compass),
            "Map":              self.get_address_value(a.inv_map),
            "Keys":             self.get_address_value(a.link_keys),
            "Bombs":            self.get_address_value(a.link_bombs),
            "Rupees":           self.get_address_value(a.link_rupees),
            "Boomerang":        self.get_address_value(a.inv_has_boom)
            }
        return inventory

    def get_actual_frame(self):
        if self.debug: print("get_actual_frame")
        frame = self.get_address_value(a.backend_frame_count) - self.frame_offset
        frame = abs(frame)
        return frame
    
    def get_tile_x_y(x, y):
        return [(round(x/21.8282828282828282828) + 1), (round(y/10.125) - 6)]

    def get_values(self):
        if self.debug: print("get_values")
        state = {}
        state["x"] =            self.get_address_value(a.link_x)
        state["y"] =            self.get_address_value(a.link_y)
        state["xy"] =           self.get_tile_x_y(state["x"],state["y"])
        state["mode"] =         self.get_address_value(a.backend_mode)
        state["level"] =        self.get_address_value(a.map_current_level)
        state["current_room"] = self.get_address_value(a.map_current_room)
        state["frame"] =        self.get_actual_frame()
        state["inventory"] =    self.load_inventory()

    def renderFinished(self):
        #if self.debug: print("Render finished.")
        #if self.iteration == 0:
        #    self.frame_offset = self.get_address_value(a.backend_frame_count) #0 - 225
        #self.iteration += 1
        #self.get_values()
        #self.do_stuff()
        print(self.get_screen())
    
    def checkPause(self, api):
        if self.getValue(0x00E0, api) == 1:
            return True
        return False
    
    def get_life(self, hp):
        hp = self.getValue(hp)
        try:
            partials = {
                256:4,
                128:3.5,
                255:3,
                127:2.5,
                254:2,
                126:1.5,
                253:1,
                125:0.5,
                0:0,
            }
            return partials[hp]
        except:
            return 0

    def make_move(self, move):
        if self.debug: print("make_move")
        if move.strip() != "":
            if move:
                self.api.writeGamepad(0, move, True)

    def release_move(self, move):
        if self.debug: print("release_move")
        if move.strip() != "":
            if move != None:
                self.api.writeGamepad(0, move, False)

    def make_random_move(self):
        if self.debug: print("make_random_move")
        moves = ["Up","Down","Left","Right","A_button","B_button"]
        if (not self.last_move.strip() or self.last_move != "A_button" or self.last_move != "B_button"):
            prob_rand = r.random()
            if (prob_rand > 0.05):
                if (self.last_move == "Up"):
                    moves.remove("Down")
                elif (self.last_move == "Down"):
                    moves.remove("Up")
                elif (self.last_move == "Left"):
                    moves.remove("Right")
                elif (self.last_move == "Right"):
                    moves.remove("Left")
                elif (self.last_move == "A_button"):
                    moves.remove("A_button")
        move = (moves[r.randint(0,len(moves) - 1)])
        self.gamepad.press_button(move)
        self.last_move = move

    def get_screen(self):
        #if self.debug: print("get_screen")
        screen = numpy.ndarray((61440,), int)
        self.api.getPixels(screen)
        screen = screen.reshape(240,256)
        screen = screen[56:240, 0:255]
        screen = screen.astype(numpy.uint8)
        screen = cv2.resize(screen, (0,0), fx=0.5, fy=0.5) 
        #240-56=184/2=92, 256/2=128
        return screen
    
    def show_screen(self, screen):
        if self.debug: print("show_screen")
        cv2.imshow('Screenshot', screen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def left_dungeon(level):
        return level == 00

    def is_dead(mode):
        return mode == 17
    
    def grabbed_triforce(mode):
        return mode == 18

    def print_debug_stats(self, i):
        if self.debug: print("print_debug_stats")
        #os.system('cls')
        print("Stats:")
        print("\niteration: " + str(i) + ", x: " + str(self.x) + ", y: " + str(self.y))
        print("Current room: " + str(self.current_room))
        print("Current level: " + str(self.level))
        print("[x,y]: " + str(self.xy))
        print("Dead?: " + str(m.is_dead(self.mode)))
        print("In dungeon?: " + str(m.left_dungeon(self.level)))