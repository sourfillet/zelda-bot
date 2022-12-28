import numpy as np

#                            #
#   Nintaco Server Methods   #
#                            #
def apiEnabled():
    print("API enabled")

def apiDisabled():
    print("API disabled")

def dispose():
    print("API stopped")

def statusChanged(message):
    print("Status message: %s" % message)

#                                  #
#   Nintaco Manipulation Methods   #
#                                  #
def getScreen(api):
    screen = np.ndarray((61440,), int)
    api.getPixels(screen)
    screen = screen.reshape(240,256)
    screen = screen[56:240, 0:255]
    screen = screen.astype(np.uint8)
    return screen

def getValue(address, api):
    return api.peekCPU(address)

def get_life(hp):
    hp = getValue(hp)
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

def checkPause(api):
    if getValue(0x00E0, api) == 1:
        return True
    return False

def pressButton(action, api):
    if action:
        api.writeGamepad(0, action, True)

def releaseButton(action, api):
    if action != None:
        api.writeGamepad(0, action, False)