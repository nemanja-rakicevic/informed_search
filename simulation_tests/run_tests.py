import itertools
import os

covs = [2, 5, 10, 20]
sqlen = [0.001, 0.01, 0.1, 0.5]
# sqlen = [0.001, 0.1, 0.5]

tests = [x for x in itertools.product(covs, sqlen)]
# tests = tests[1:]

for i in tests:
	print("\nrunning ",i)
	os.system("python simulation_complete.py -o "+str(i[0])+" "+str(i[1])+" 1")
