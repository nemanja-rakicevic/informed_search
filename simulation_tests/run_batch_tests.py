import itertools
import os

model = ['informed', 'random']

covs = [5, 10, 20]
# sqlen = [0.001, 0.01, 0.1, 0.5]
sqlen = [0.001, 0.01, 0.1]

tests = [x for x in itertools.product(model, covs, sqlen)]
# tests = tests[1:]

for i in tests:
	print("\nrunning ",i)
	os.system("python simulation_experiments.py -m {} -o {} {} 1".format(i[0], str(i[1]), str(i[2]) ))
