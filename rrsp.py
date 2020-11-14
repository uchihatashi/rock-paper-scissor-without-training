# Imports
import numpy as np
import cv2
import math
from gameLogic import *
import time
import threading

#defining some flags 
#global str_time, count_str_game, time_flag, start_game, count_12345, starts
str_time,count_str_game = 0,0
time_flag, start_game = False,False
count_12345 = "0"
starts = False
after_count = False
j = 0

robot,human,count_score=0,0,1
# Open Camera

capture = cv2.VideoCapture(0) 



while capture.isOpened():

    # Capture frames from the camera
    ret, frame = capture.read()

    # Get hand data from the rectangle sub window
    cv2.rectangle(frame, (50, 100), (300, 400), (0, 255, 0), 0)
    crop_image = frame[100:400, 50:300]

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # Change color-space from BGR -> HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([0, 48, 80], dtype = 'uint8'), np.array([20, 255, 255], dtype = 'uint8'))

    # Kernel for morphological transformation
    #kernel = np.ones((5, 5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # Apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Apply Gaussian Blur and Threshold
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)


    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.imread('./assets/RvsH.jpg')
    try:
        k = cv2.waitKey(10)
        # Find contour with maximum area
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        # Create bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # Find convex hull
        hull = cv2.convexHull(contour)

        # Draw contour
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        # Find convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger tips) for all defects
        count_defects = 0
        
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            # apply cosine rule 
            angle = (math.acos((b**2 + c**2 - a**2) / (2*b*c)) * 180) / 3.14
    
            
            # if angle > 90 draw a circle at the far point
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

            cv2.line(crop_image, start, end, [0, 255, 0], 2)
        
    except:
        pass

    def game_result(winner, robot, human):
        global count_score
        if winner == "you won":
            human += 1
        elif winner == "you lost":
            robot += 1

        cv2.putText(result, str(robot), (50, 80), cv2.FONT_HERSHEY_TRIPLEX, 3,(0,0,0), 2)
        cv2.putText(result, str(human), (270, 80), cv2.FONT_HERSHEY_TRIPLEX, 3,(0,0,0), 2)
        
        count_score += 1
        if winner == "0":
            if count_score >= 7 and count_score <= 11:
                if robot > human:
                    final_result = "Robot WON"
                elif robot < human:
                    final_result = "Human WON"
                else: 
                    final_result = "DRAW"
                cv2.putText(result, "GAME OVER", (100, 180), cv2.FONT_HERSHEY_TRIPLEX, 1,(50,255,50), 2)
            else:
                final_result = "New Game"
                count_score = 1
                cv2.putText(result, "Lets Start", (100, 180), cv2.FONT_HERSHEY_TRIPLEX, 1,(50,255,50), 2)
            cv2.putText(result, final_result, (30,250), cv2.FONT_HERSHEY_TRIPLEX, 1.5,(0,0,255), 2)
            
        else:
            cv2.putText(result, winner, (30,250), cv2.FONT_HERSHEY_TRIPLEX, 2,(0,0,255), 2)
            cv2.putText(result, "Round "+str(count_score), (60, 180), cv2.FONT_HERSHEY_TRIPLEX, 2,(0,255,0), 4)
        
        cv2.imshow("Robot vs Human", result)
        global after_count
        after_count = False
        return robot, human

    if (starts == False):
            info = "Press (S/s) to start the game"
            info1= "Please put only the fingers in the rectangel box (ie: no face or body part)"
            cv2.putText(frame, info, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)
            cv2.putText(frame, info1, (20, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)

    if after_count == True:
        robot, human = callgame(robot, human)
        count_12345, count_str_game = "0", 0
    if count_score == 1:
        human, robot = 0, 0
        game_result("0", robot, human)
    if count_score >= 6 and count_score <= 10:
        game_result("0", robot, human)
        

    # Print number of fingers to make a countdown for the game
    if (k == ord('s') or k == ord('S')):
        starts = True
        start_game = True
        
    if start_game == True:
        count_12345, count_str_game, j = countdown_game(count_12345,count_str_game,j)
        


    def countdown_game(count_12345, count_str_game, j):
        if count_str_game == 5:
            if j<=30:
                cv2.putText(frame, "ROCK", (125, 270), cv2.FONT_HERSHEY_TRIPLEX, 4.3,(0,255,255), 10)
            elif j <= 60:
                cv2.putText(frame, "PAPER", (90, 270), cv2.FONT_HERSHEY_TRIPLEX, 4.3,(0,255,255), 10)
            elif j <= 90:
                cv2.putText(frame, "SCISSOR", (20, 270), cv2.FONT_HERSHEY_TRIPLEX, 4.3,(0,255,255), 10)
            elif j < 105:
                cv2.putText(frame, "START", (100, 250), cv2.FONT_HERSHEY_TRIPLEX, 5,(0,255,0), 10)
            else:
                t1 = threading.Thread(target = game_thread)
                t1.run()
            j += 1

        elif count_defects == 0:
            cv2.putText(frame, "1", (205, 350), cv2.FONT_HERSHEY_TRIPLEX, 12,(0,0,255), 12)
            if (count_12345 == "0" and count_str_game == 0):
                count_12345 = "1"
                count_str_game += 1
            else:
                cv2.putText(frame, "Current count is " + count_12345, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)
                cv2.putText(frame, "Please show " + str(count_str_game + 1 ), (20, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)

        elif count_defects == 1:
            cv2.putText(frame, "2", (205, 350), cv2.FONT_HERSHEY_TRIPLEX, 12,(0,0,255), 12)
            if count_12345 == "1" and count_str_game == 1:
                count_12345 = "1_2"
                count_str_game += 1
            else:
                cv2.putText(frame, "Current count is " + count_12345, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)
                cv2.putText(frame, "Please show " + str(count_str_game + 1 ), (20, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)
        
        elif count_defects == 2:
            cv2.putText(frame, "3", (205, 350), cv2.FONT_HERSHEY_TRIPLEX, 12,(0,0,255), 12)
            if count_12345 == "1_2" and count_str_game == 2:
                count_12345 = "1_2_3"
                count_str_game += 1
            else:
                cv2.putText(frame, "Current count is " + count_12345, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)
                cv2.putText(frame, "Please show " + str(count_str_game + 1 ), (20, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)
        
        elif count_defects == 3:
            cv2.putText(frame, "4", (205, 350), cv2.FONT_HERSHEY_TRIPLEX, 12,(0,0,255), 12)
            if count_12345 == "1_2_3" and count_str_game == 3:
                count_12345 = "1_2_3_4"
                count_str_game += 1
            else:
                cv2.putText(frame, "Current count is " + count_12345, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)
                cv2.putText(frame, "Please show " + str(count_str_game + 1 ), (20, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)
        
        elif count_defects == 4:    #total of five fingers
            cv2.putText(frame, "5", (205, 350), cv2.FONT_HERSHEY_TRIPLEX, 12,(0,0,255), 12)
            if count_12345 == "1_2_3_4" and count_str_game == 4:
                count_str_game += 1
                j = 0
                if count_score >= 10:
                    human, robot = 0, 0
                    game_result("0", robot, human)
            else:
                cv2.putText(frame, "Current count is " + count_12345, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)
                cv2.putText(frame, "Please show " + str(count_str_game ), (20, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)
                
        return count_12345,count_str_game,j
       
    def game_thread():
        global after_count
        time.sleep(1)
        after_count = True
        

    def callgame(robot,human):
        
        rps=""
        if count_defects == 0:
            cv2.putText(crop_image, "Rock", (20, 40), cv2.FONT_HERSHEY_TRIPLEX, 1.3,(255,51,51), 2)
            rps="rock"
        elif count_defects == 1:
            cv2.putText(crop_image, "Scissor", (20, 40), cv2.FONT_HERSHEY_TRIPLEX, 1.3,(255,51,51), 2)
            rps="scissor"
        elif count_defects == 4:
            cv2.putText(crop_image, "Paper", (20, 40), cv2.FONT_HERSHEY_TRIPLEX, 1.3,(255,51,51), 2)
            rps="paper"
        else:
            cv2.putText(crop_image, "Unknown", (20, 40), cv2.FONT_HERSHEY_TRIPLEX, 1.3,(255,51,51), 2)
            rps="unknown"

        if(not rps=="unknown"):
            computerChoice = runGame()
            print("Player chooses: " + rps)
            print("Computer chooses: " + computerChoice)
            
            winner,showme = logic(rps, computerChoice)

            cv2.imshow('computer choice',showme)
            cv2.imshow('human choice',crop_image)

            print(winner + "\n")
            robot, human = game_result(winner, robot, human)

        else:       
            robot,human = game_result("Unknown",robot,human)
            cv2.imshow('human choice',crop_image)
        return robot,human
        
            
    

    
            
    if k == ord('q') or k == ord('Q'):
        break

    
    # Show threshold image
    cv2.imshow("Threshold", thresh)
    
    # Show required images
    cv2.imshow("GAME FRAME", frame)
    #all_image = np.hstack((drawing, crop_image))
    #cv2.imshow('Contours', all_image)

    # Close the camera if 'q' is pressed
    


capture.release()
cv2.destroyAllWindows()




