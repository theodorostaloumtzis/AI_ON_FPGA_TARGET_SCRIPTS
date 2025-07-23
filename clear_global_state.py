from pynq.pl_server.global_state import clear_global_state
from pynq import Device
import glob, os, pathlib

# 1. Make sure no PL design is currently running
Device.active_device.free_bitstream()   # puts the PL into reset

# 2. Remove the global_pl_state.json file
clear_global_state()                    # same as deleting it manually

# 3. Delete *.pynqmetadata files that belong to earlier builds
for fname in glob.glob("/home/ubuntu/Documents/ai_on_fpga/*.pynqmetadata"):
    os.remove(fname)
