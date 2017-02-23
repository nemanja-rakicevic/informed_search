

# DISTANCE
model_err = -1*np.ones((4,7))
rand_err = -1*np.ones((5,4,7))

human_err = -1*np.ones((10,4,7))
pro_err = -1*np.ones((5,4,7))

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


#HUMANS
#1 tom
human_err[0,0,:] = np.array([7, 30, 14, 28, 31, 115, 89])
human_err[0,1,:] = np.array([68, 9.5, 74, 12, 30, 60, 40])
human_err[0,2,:] = np.array([51, 72, 35, 38, 53, 32, 60])
human_err[0,3,:] = np.array([10, 97, 60, 72, 47, 50, 46])
#2 roni
human_err[1,0,:] = np.array([8, 23, 25, 25, 25, 56, 18])
human_err[1,1,:] = np.array([34, 13, 20, 13, 63, 46, 50])
human_err[1,2,:] = np.array([5, 10, 32, 140, 8, 26, 106])
human_err[1,3,:] = np.array([15, 4, 38, 18, 43, 152, 54])
#3 arash
human_err[2,0,:] = np.array([40, 15, 13, 4, 17, 53, 45])
human_err[2,1,:] = np.array([8, 15, 5.5, 59, 15, 18, 112])
human_err[2,2,:] = np.array([30, 15, 43, 2.5, 19, 10, 6])
human_err[2,3,:] = np.array([ 12, 66, 17, 34, 14, 6, 35])
#4 fabio
human_err[3,0,:] = np.array([25, 7, 10, 30, 45, 22.5, 33])
human_err[3,1,:] = np.array([10, 3, 14, 7.5, 27, 49, 41])
human_err[3,2,:] = np.array([10, 7, 2, 20, 54, 12, 10])
human_err[3,3,:] = np.array([1, 6, 17, 39, 41.5, 6, 20])
#5 sangeetha
human_err[4,0,:] = np.array([47, 29, 103, 23, 112, 25, 25])
human_err[4,1,:] = np.array([10, 23, 86, 31, 48, 119, 27])
human_err[4,2,:] = np.array([7.5, 26, 12, 29, 16, 0, 35])
human_err[4,3,:] = np.array([4.5, 48, 16, 22, 44, 55, 30])
#6 belen
human_err[5,0,:] = np.array([20, 27, 6, 42, 26, 62, 68])
human_err[5,1,:] = np.array([52, 2, 9, 13, 30, 50, 40])
human_err[5,2,:] = np.array([8.5, 16, 33, 27.5, 27, 25, 44])
human_err[5,3,:] = np.array([8, 20, 6.5, 19, 14, 20, 31])
#7 ben
human_err[6,0,:] = np.array([25, 7.5, 8, 19, 18, 32, 65])
human_err[6,1,:] = np.array([5, 4, 18.5, 17, 23, 47, 9])
human_err[6,2,:] = np.array([3, 19, 8.5, 44, 13.5, 9, 56])
human_err[6,3,:] = np.array([1, 19, 17, 30, 12, 60, 19])
#8 tom2
human_err[7,0,:] = np.array([9, 14, 13, 130, 54, 8, 53])
human_err[7,1,:] = np.array([17, 1, 66, 45, 12, 119, 50])
human_err[7,2,:] = np.array([22, 18, 23, 56, 68, 34, 17])
human_err[7,3,:] = np.array([1, 28, 14, 25.5, 50, 52, 77])
#9 nathan
human_err[8,0,:] = np.array([15, 18, 14, 24, 12, 30, 9])
human_err[8,1,:] = np.array([18, 4.5, 20, 43, 29, 64, 2])
human_err[8,2,:] = np.array([38, 21, 18, 24.5, 2, 8, 8.5])
human_err[8,3,:] = np.array([30, 13, 46, 60, 24, 74, 62])
#10 lucy
human_err[9,0,:] = np.array([23, 28, 62, 13, 75, 55, 26])
human_err[9,1,:] = np.array([5, 10, 32, 17, 29, 44, 40])
human_err[9,2,:] = np.array([20, 27, 21, 81.5, 115, 9, 3.5])
human_err[9,3,:] = np.array([14, 51.5, 28, 20, 99, 2, 140])


#EXPERTS
#1 jukka
pro_err[0,0,:] = np.array([2, 18, 42, 28, 13, 86, 60])
pro_err[0,1,:] = np.array([12, 14, 27, 15, 40, 36, 40])
pro_err[0,2,:] = np.array([20, 0, 18, 4, 5, 73, 40])
pro_err[0,3,:] = np.array([4, 29.5, 6, 19, 25, 3, 19])
#2 matthias
pro_err[1,0,:] = np.array([14, 8, 4, 11, 42, 38, 57])
pro_err[1,1,:] = np.array([27, 27, 14, 22, 24, 24, 6])
pro_err[1,2,:] = np.array([6, 29, 11, 49, 46, 46, 18])
pro_err[1,3,:] = np.array([16, 25, 60, 22, 9.5, 41, 44])
#3 gus
pro_err[2,0,:] = np.array([7, 5, 16, 10, 13.5, 35, 16])
pro_err[2,1,:] = np.array([12, 4.5, 11, 23, 28, 15, 15])
pro_err[2,2,:] = np.array([2, 5, 10, 10, 32, 35, 6])
pro_err[2,3,:] = np.array([14, 4, 15, 35, 8, 21, 74])
#4 
pro_err[3,0,:] = np.array([])
pro_err[3,1,:] = np.array([])
pro_err[3,2,:] = np.array([])
pro_err[3,3,:] = np.array([])
#5 
pro_err[4,0,:] = np.array([])
pro_err[4,1,:] = np.array([])
pro_err[4,2,:] = np.array([])
pro_err[4,3,:] = np.array([])




human_err.mean()

model_err.mean()
rand_err.mean()



human_err.std()

model_err.std()
rand_err.std()



