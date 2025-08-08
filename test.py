import numpy as np
import cv2
import pickle

width = 640
height = 480

cap = cv2.VideoCapture(1)
cap.set(3, width)
cap.set(4, height)

pickle_in = open