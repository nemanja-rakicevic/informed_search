
import itertools
import os

# TODO Rewrite this as a config file ?

hypersweep_model = ['uidf', 'entropy']

hypersweep_covs = [5, 10, 20]
# sqlen = [0.001, 0.01, 0.1, 0.5]
hypersweep_sqlen = [0.001, 0.01, 0.1]

test_hypersweep = [x for x in itertools.product(hypersweep_model, hypersweep_covs, hypersweep_sqlen)]
# tests = tests[1:]

for i in test_hypersweep:
	print("\nrunning ", i)
	os.system("python simulation_experiments.py -m {} -o {} {} 1".format(i[0], str(i[1]), str(i[2]) ))
