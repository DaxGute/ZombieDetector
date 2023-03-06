import cv2 
import argparse

def getVideo():
    while True:
        videoPath = input("Path: ")
        try:
            if input == "quit":
                return "quit"
            vidcap = cv2.VideoCapture(videoPath)
            return vidcap
        except:
            print("Error reading video")
    
def main():
    # Gets the video
    vidcap = getVideo()
    if vidcap == "quit":
        print("Quitting Program")
        return

    success, image = vidcap.read()                          # Gets the initial frame
    count = 0

    # Loops through each frame in the video 
    while success:
        print("yay")
        cv2.imwrite("./data/zombie/vid1_%d.jpg" % count, image)    # Writes the current frame
        for i in range(30):
            success, image = vidcap.read()                      # Gets the subsequent frame
        if success == False:
            print("Break")
            break
        count += 1

    vidcap.release()
main()