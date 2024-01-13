import cv2
import cv2 as cv
import numpy
import numpy as np
import openpyxl

import utils
##########################################
# PARAMETERS
path = 'imgs/1.png'
heightImg = 700
widthImg = 700
questions = 5
choices = 5
# ans = [1,2,0,1,4]

##########################################
# load answers from marking_scheme
workbook = openpyxl.load_workbook('marking_scheme.xlsx')
worksheet = workbook['MCQ']
cell_range = worksheet['B2:B6']
ans = []
for cell in cell_range:
    ans.append(cell[0].value)
print("Ans: ",ans)



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
cv.drawContours(imgContours,contours,-1,(0,255,0),10)

# finding rectangles
rectCon = utils.rectContour(contours)
biggestContour = utils.getCornerPoints(rectCon[0])
gradePoints = utils.getCornerPoints(rectCon[1])
#print(biggestContour.shape)

if biggestContour.size != 0 and gradePoints.size != 0:
    cv.drawContours(imgBiggestContours,biggestContour,-1,(0,255,0),20)
    cv.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 20)

    biggestContour = utils.reorder(biggestContour)
    gradePoints = utils.reorder(gradePoints)

    # get birds eye view on answers box
    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix = cv.getPerspectiveTransform(pt1,pt2)
    imgWarpColored = cv.warpPerspective(img,matrix,(widthImg,heightImg))

    # get birds eye view on grade display box
    ptG1 = np.float32(gradePoints)
    ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
    matrixG = cv.getPerspectiveTransform(ptG1, ptG2)
    imgGradeDisplay = cv.warpPerspective(img, matrixG, (325, 150))
    # cv.imshow("Grade Display",imgGradeDisplay)

    # apply threshold
    imgWarpGray = cv.cvtColor(imgWarpColored,cv.COLOR_BGR2GRAY)
    imgThresh = cv.threshold(imgWarpGray,170,255, cv.THRESH_BINARY_INV)[1]

    #
    boxes = utils.splitBoxes(imgThresh)
    #cv.imshow("Boxes",boxes[24])
    # print(cv.countNonZero(boxes[1]),cv.countNonZero(boxes[2]))

    # get non zero pixel values of each box
    myPixelVal = np.zeros((questions, choices))
    countR = 0
    countC = 0
    for image in boxes:
        # cv2.imshow(str(countR)+str(countC),image)
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if (countC == choices): countC = 0;countR += 1
    # print(myPixelVal)

    # FIND THE USER ANSWERS AND PUT THEM IN A LIST
    myIndex = []
    for x in range(0, questions):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr == np.amax(arr))
        myIndex.append(myIndexVal[0][0])
    # print("USER ANSWERS",myIndex)

    # COMPARE THE VALUES TO FIND THE CORRECT ANSWERS
    grading = []
    for x in range(0, questions):
        if ans[x] == myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)
    # print("GRADING",grading)
    score = (sum(grading) / questions) * 100  # FINAL GRADE
    print("SCORE",score)


imgBlank = np.zeros_like(img)
imgArray = ([img,imgGray,imgBlur,imgCanny],
            [imgContours,imgBiggestContours,imgWarpColored,imgThresh])
imgStack = utils.stackImages(imgArray,0.4)

cv.imshow("Image Stack",imgStack)

cv.waitKey(0)