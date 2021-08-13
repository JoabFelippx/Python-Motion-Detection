import cv2 as cv


# video de teste
videopath = './video-test/video-test.mp4'

# pegar frame base para comparação


def getFirstFrame():

    capture = cv.VideoCapture(1)

    ret, frame = capture.read()

    if ret:

        gray_baseimage = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_baseimage = cv.GaussianBlur(gray_baseimage, (25, 25), 0)
        cv.imwrite('baseframe.jpg', gray_baseimage)
        return gray_baseimage


capture = cv.VideoCapture(1)
baseimage = getFirstFrame()

while(1):

    ret, frame = capture.read()

    # converter frames
    gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_image = cv.GaussianBlur(gray_image, (25, 25), 0)

    # comparar imagens
    delta_img = cv.absdiff(baseimage, gray_image)
    threshold = cv.threshold(delta_img, 37, 255, cv.THRESH_BINARY)[1]

    # achar todos os contornos da imagem
    (contours, _) = cv.findContours(threshold,
                                    cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv.contourArea(contour) < 10000:
            continue
        (x, y, w, h) = cv.boundingRect(contour)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    cv.imshow('VIDEO', frame)

    if cv.waitKey(1) == ord('q'):
        break

cv.imwrite('view.jpg', threshold)
cv.destroyAllWindows()
