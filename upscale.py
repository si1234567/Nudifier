import megapper
import sys

print(sys.argv)
argone = sys.argv[1]
argtwo = sys.argv[2]

megapper.upscale()(argone, argtwo)
