import pytvc
from pytvc.physics import *
from pytvc.rocket import rocketBody

rocket1: rocketBody = rocketBody(dry_mass=1.0, time_step=100)

@rocket1.setup
def lol():
    print("setup")

@rocket1.update
def lol1():
    print("update")

rocket1.run()