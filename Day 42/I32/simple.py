import os
import ctypes
 
module = ctypes.CDLL("./libsimple.so")
 
module.main.argtypes = []
module.main.restype = ctypes.c_int
 
print(module.main())