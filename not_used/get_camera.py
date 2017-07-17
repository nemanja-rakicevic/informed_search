
import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)

    while True:
        ret_val, img = cam.read()

        if mirror: 
            img = cv2.flip(img, 1)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_filt = cv2.medianBlur(img_gray, 15)
        img_th = cv2.adaptiveThreshold(img_filt,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        img_disp = img
        cv2.drawContours(img_disp, contours, -1, (0,255,0), 3)

        cv2.imshow('my webcam', img_disp)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    cam.release()
    cv2.destroyAllWindows()

def main():
    show_webcam(mirror=False)

if __name__ == '__main__':
    main()