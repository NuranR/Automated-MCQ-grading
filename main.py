import cv2 as cv
import numpy as np
import openpyxl

import utils
##########################################
# parameters
path = 'test_imgs/s1.jpg'
heightImg = 850
widthImg = 600
questions = 50
choices = 5

##########################################
# load answers from marking_scheme
workbook = openpyxl.load_workbook('marking_scheme.xlsx')
worksheet = workbook['MCQ']
cell_range = worksheet['E2:E51']
ans = []
for cell in cell_range:
    ans.append(cell[0].value)
print("Marking Scheme Answers: ",ans)

img = cv.imread(path)

# img preprocessing
img = cv.resize(img,(widthImg,heightImg))

imgContours = img.copy()
imgBiggestContours = img.copy()
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray,(5,5),1)
imgCanny = cv.Canny(imgBlur,10,50)

# Finding all contours
contours, hierarchy = cv.findContours(imgCanny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
cv.drawContours(imgContours,contours,-1,(0,255,0),1)

# finding rectangles
rectCon = utils.rectContour(contours)
biggestContour1 = utils.getCornerPoints(rectCon[0])
biggestContour2= utils.getCornerPoints(rectCon[1])

# get the 2 biggest contours and determine the left and right
new_contours = [biggestContour1, biggestContour2]

#print("new contours: ")
point_of_interest = (0,0)
min_distance = float('inf')
for i in range(2):
     contour = new_contours[i]
     #print(contour)
     #print(i)
     x,y,w,h = cv.boundingRect(contour)
     distance = np.sqrt(x**2 + y**2)
     #print("distance: " + str(distance))
     if distance < min_distance:
         min_distance = distance
         left_contour = contour
         left_contour_index = i
if left_contour_index==0:
    right_contour = new_contours[1]
else:
    right_contour = new_contours[0]


if biggestContour1.size != 0 and biggestContour2.size != 0:
    cv.drawContours(imgBiggestContours,left_contour,-1,(0,0,255),20)
    cv.drawContours(imgBiggestContours, right_contour, -1, (255, 0, 0), 20)

    left_contour = utils.reorder(left_contour)
    right_contour = utils.reorder(right_contour)

    # get birds eye view on left contour
    pt1 = np.float32(left_contour)
    pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix = cv.getPerspectiveTransform(pt1,pt2)
    imgWarpColored = cv.warpPerspective(img,matrix,(widthImg,heightImg))

    # get birds eye view on right contour
    ptR1 = np.float32(right_contour)
    ptR2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrixR = cv.getPerspectiveTransform(ptR1, ptR2)
    imgWarpColoredR = cv.warpPerspective(img,matrixR,(widthImg,heightImg))

    # apply threshold to left
    imgWarpGrayL = cv.cvtColor(imgWarpColored,cv.COLOR_BGR2GRAY)
    imgThreshL = cv.threshold(imgWarpGrayL,170,255, cv.THRESH_BINARY_INV)[1]

    # apply threshold to right
    imgWarpGrayR = cv.cvtColor(imgWarpColoredR, cv.COLOR_BGR2GRAY)
    imgThreshR = cv.threshold(imgWarpGrayR, 170, 255, cv.THRESH_BINARY_INV)[1]

    boxes = utils.splitBoxes(imgThreshL)
    # cv.imshow("Boxes",boxes[24])
    # cv.waitKey(0)
    # print(cv.countNonZero(boxes[1]),cv.countNonZero(boxes[2]))

    # get non zero pixel values of each box
    myPixelVal = np.zeros((questions, choices))
    countR = 0
    countC = 0
    for image in boxes:
        # cv2.imshow(str(countR)+str(countC),image)
        totalPixels = cv.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if (countC == choices): countC = 0;countR += 1
    print(myPixelVal)

    ### split boxes of right side
    boxes = utils.splitBoxes(imgThreshR)

    # get non zero pixel values of each box
    #myPixelVal = np.zeros((questions, choices))
    countR = 0
    countC = 0
    for image in boxes:
        # cv2.imshow(str(countR)+str(countC),image)
        totalPixels = cv.countNonZero(image)
        myPixelVal[25+countR][countC] = totalPixels
        countC += 1
        if (countC == choices):
            countC = 0
            countR += 1
    print(myPixelVal)

    # find student answer and enter to a list
    myIndex = []
    for x in range(0, questions):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr == np.amax(arr))
        myIndex.append(myIndexVal[0][0])
    print("Student Answers:",myIndex)

    # compare the answers and obtain the score
    grading = []
    for x in range(0, questions):
        if ans[x] == myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)
    score = (sum(grading) / questions) * 100  # final score
    print("SCORE",round(score))

imgBlank = np.zeros_like(img)
imgArray = ([img,imgGray,imgBlur,imgCanny],
#[imgContours,imgBiggestContours,img,img])
            [imgContours,imgBiggestContours,imgWarpColored,imgThreshR])
imgStack = utils.stackImages(imgArray,0.4)

cv.imshow("Image Stack",imgStack)
#cv.imshow("Image Stack",imgBiggestContours)

cv.waitKey(0)