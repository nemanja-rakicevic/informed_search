
import numpy as np
import cv2
import colorsys
import PIL

DWN_RATE = 4
# Load background image
img_background = PIL.Image.open('pic_background.png', 'r')
img_bd = img_background.resize((img_background.size[0]/DWN_RATE, img_background.size[1]/DWN_RATE), Image.ANTIALIAS)

# Load puck image
img_puck = Image.open('pic_puck.png', 'r')
img_pd = img_puck.resize((img_puck.size[0]/DWN_RATE, img_puck.size[1]/DWN_RATE), Image.ANTIALIAS)

# Load end and end2 image (use either depending on the y coordinate)
img_goal = Image.open('pic_goal1.png', 'r')
img_gd = img_goal.resize((img_goal.size[0]/DWN_RATE, img_goal.size[1]/DWN_RATE), Image.ANTIALIAS)


# Generate image (60k)
img = img_bd.copy()
draw = PIL.ImageDraw.Draw(img)
th = np.linspace(0,2*np.pi,100)
r = img_bd.size[0]/3.8    # width
h = img_bd.size[0]/2.47 # x-coord
k = img_bd.size[1]/3    # y-coord
ratio = 0.55
x = h + r*np.cos(th)
y = k - ratio * r*np.sin(th)
draw.point(zip(x,y),fill=128)

th_r = th #np.random.choice(th)
phi = np.linspace(0,1,100) #np.random.choice(np.linspace(0,1,100))
x_p = np.sqrt(phi) * np.cos(th_r)
y_p = np.sqrt(phi) * np.sin(th_r)
x_p = h + x_p * r
y_p = k + y_p * ratio*r

draw.point(zip(x_p,y_p),fill=255)

plt.imshow(img)
plt.show()

# puck around starting point
img.paste(img_pd, (x_p, y_p), img_pd)

plt.imshow(img)
plt.show()

# goal within a range (whole hat or patch???)
# size = 
# img_goal.resize(size, Image.ANTIALIAS)
# img_background.paste(img_gd, (x_g, y_g), img_gd)

# img.save()



# ELIPSE of positions (downsampled 4x)
# img = img_bd.copy()
# draw = PIL.ImageDraw.Draw(img)
# th = np.linspace(-np.pi,np.pi,100)
# r = 40
# h = 65
# k = 40
# x = h + r*np.cos(th)
# y = k - 0.55 * r*np.sin(th)
# draw.point(zip(x,y),fill=128)
# plt.imshow(img)
# plt.show()