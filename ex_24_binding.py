#----------------------------------------------------------------------------------------------------------------------
from ctypes import *
import os
#----------------------------------------------------------------------------------------------------------------------
class wrapper(object):

    lib = cdll.LoadLibrary(os.getcwd()+'/foo.dll')

    lib.new_vector.restype = c_void_p
    lib.new_vector.argtypes = []

    lib.delete_vector.restype = None
    lib.delete_vector.argtypes = [c_void_p]

    lib.vector_size.restype = c_int
    lib.vector_size.argtypes = [c_void_p]

    lib.vector_get.restype = c_int
    lib.vector_get.argtypes = [c_void_p, c_int]

    lib.vector_push_back.restype = None
    lib.vector_push_back.argtypes = [c_void_p, c_int]

    lib.foo.restype = None
    lib.foo.argtypes = [c_void_p]

# ----------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        self.vector = wrapper.lib.new_vector()
# ----------------------------------------------------------------------------------------------------------------------
    def __del__(self):
        wrapper.lib.delete_vector(self.vector)
# ----------------------------------------------------------------------------------------------------------------------
    def __len__(self):
        return wrapper.lib.vector_size(self.vector)
# ----------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, i):
        if 0 <= i < len(self):
            return wrapper.lib.vector_get(self.vector, c_int(i))
        raise IndexError('Vector index out of range')
# ----------------------------------------------------------------------------------------------------------------------
    def __repr__(self):
        return '[{}]'.format(', '.join(str(self[i]) for i in range(len(self))))
# ----------------------------------------------------------------------------------------------------------------------
    def push(self, i):
        wrapper.lib.vector_push_back(self.vector, c_int(i))
# ----------------------------------------------------------------------------------------------------------------------
    def foo(self, filename):
        wrapper.lib.foo(self.vector, c_char_p(filename.encode('utf-8')))
#----------------------------------------------------------------------------------------------------------------------
filename_in = './data/ex_ctypes/data.txt'
#----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    W = wrapper()
    W.foo(filename_in)

    print(W.vector)

    u=0


