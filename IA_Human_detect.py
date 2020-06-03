import cv2
import imutils
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
image = cv2.imread('road2.jpg')
image = imutils.resize(image, width=min(1500, image.shape[1]))
(regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.10)
for (x, y, w, h) in regions:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

width = 1920
height = 1080

dsize = (width, height)
output = cv2.resize(image, dsize)
cv2.imwrite('img.png', output)
cv2.imshow("Image", image)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
cv2.destroyAllWindows()