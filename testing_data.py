

# DISTANCE
model_err = -1*np.ones((4,7))
rand_err = -1*np.ones((5,4,7))

# MODEL
model_err[0,:] = np.array([56, 6.5, 14, 32, 17.5, 31, 75])
model_err[1,:] = np.array([22.5, 22.5, 58, 41, 18, 4.5, 41])
model_err[2,:] = np.array([18, 8, 37, 30, 37.5, 21.5, 22])
model_err[3,:] = np.array([19, 26.5, 45, 9, 34.5, 42, 36])

#RANDOM
#seed1
rand_err[0,0,:] = np.array([23.5, 59, 176, 26, 49, 166, 64])
rand_err[0,1,:] = np.array([9, 19, 120, 60, 58, 32.5, 43])
rand_err[0,2,:] = np.array([12, 29.5, 58, 113, 150, 51, 18])
rand_err[0,3,:] = np.array([67,65,83,104,112,15.5,169])
#seed2
rand_err[1,0,:] = np.array([ 9, 42, 36, 58, 39, 234, 90])
rand_err[1,1,:] = np.array([ 19.5, 57, 15, 43, 95, 56, 46])
rand_err[1,2,:] = np.array([ 61, 20.5, 52.5, 16, 37, 94, 14])
rand_err[1,3,:] = np.array([ 31, 37.5, 108, 16, 121, 52, 78.5])
#seed3
rand_err[2,0,:] = np.array([69, 66, 20, 80, 36, 94, 67])
rand_err[2,1,:] = np.array([12, 5, 70, 33, 28, 101.5, 20])
rand_err[2,2,:] = np.array([10, 40, 30, 116, 62, 184, 70])
rand_err[2,3,:] = np.array([23, 43, 81, 23, 40.5, 95, 39.5])
#seed4
rand_err[3,0,:] = np.array([14, 41, 83, 38, 23, 50, 99])
rand_err[3,1,:] = np.array([58, 49, 45, 64, 61, 51, 89])
rand_err[3,2,:] = np.array([20, 37, 11.5, 76, 93, 177, 78])
rand_err[3,3,:] = np.array([62, 14, 76, 57, 89, 59, 165])
#seed5
rand_err[4,0,:] = np.array([23, 26, 50, 28.5, 71, 49, 62])
rand_err[4,1,:] = np.array([45, 75, 194, 74, 99.5, 160, 45])
rand_err[4,2,:] = np.array([34.5, 19.5, 21.5, 80, 15, 76, 188])
rand_err[4,3,:] = np.array([45, 43, 55, 134, 97.5, 140, 165])



model_err.mean()
rand_err.mean()

model_err.std()
rand_err.std()



