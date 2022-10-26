import os
import cv2
from PIL import Image
import numpy as np
import img_math 
import img_recognition 

SZ = 20  # Training picture length and width
MAX_WIDTH = 1000  # Maximum width of original image
Min_Area = 2000  # Maximum area allowed for license plate area
PROVINCE_START = 1000


class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    # # training svm
    # def train(self, samples, responses):
    #     self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # Character Recognition
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()


class CardPredictor:
    def __init__(self):
        pass

    def train_svm(self):
        # Identify letters and numbers
        self.model = SVM(C=1, gamma=0.5)
        # Recognize Chinese
        self.modelchinese = SVM(C=1, gamma=0.5)
        if os.path.exists("svm.dat"):
            self.model.load("svm.dat")
        if os.path.exists("svmchinese.dat"):
            self.modelchinese.load("svmchinese.dat")

    def img_first_pre(self, car_pic_file):
        """
        :param car_pic_file: 图像文件
        :return:已经处理好的图像文件 原图像文件
        """
        if type(car_pic_file) == type(""):
            img = img_math.img_read(car_pic_file)   #read file
        else:
            img = car_pic_file

        pic_hight, pic_width = img.shape[:2]  #Take the height and width of the color picture
        if pic_width > MAX_WIDTH:
            resize_rate = MAX_WIDTH / pic_width
            # Zoom in on the picture
            img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
        # about ‘interpolation’ There are several parameters to choose from:
        # cv2.INTER_AREA - Partial pixel resampling, suitable for shrinking images.
        # cv2.INTER_CUBIC和 cv2.INTER_LINEAR Better for enlarging images，INTER_LINEAR  Default Method。

        img = cv2.GaussianBlur(img, (5, 5), 0)
        # Gaussian filtering is a linear smoothing filter, which is very effective in removing Gaussian noise
        # 0 is the size of the window according to（ 5,5 ）accumulate Gaussian function S方
        

        oldimg = img
        # Conversion to grayscale images
        # Convert color space cv2.cvtColor
        # BGR ---> Gray  cv2.COLOR_BGR2GRAY
        # BGR ---> HSV  cv2.COLOR_BGR2HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        cv2.imwrite("tmp/img_gray.jpg", img)
        
        #ones()Returns an all-1 n-dimensional array 
        Matrix = np.ones((20, 20), np.uint8)  

        # Open operation:Advanced erosion followed by expansion is called open operation. It is used to remove noise。 cv2.MORPH_OPEN        
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, Matrix)

        # Image overlay and fusion
        # g (x) = (1 − α)f0 (x) + αf1 (x)   a→（0，1）不同的a值可以实现不同的效果
        img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)
        # cv2.imwrite("tmp/img_opening.jpg", img_opening)
        # create 20*20 elements 1 Matrix Open Operation，and img  overlay


       
        ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Canny Edge detection
        #Larger threshold 2 is used to detect obvious edges in the image Generally the detection is not so perfect, and the edges are detected intermittently
        # 较大的阈值2用于检测图像中明显的边缘  一般情况下检测的效果不会那么完美，边缘检测出来是断断续续的
        # 较小的阈值1用于将这些间断的边缘连接起来
        #A smaller threshold of 1 is used to connect these interrupted edges
        img_edge = cv2.Canny(img_thresh, 100, 200)
        cv2.imwrite("tmp/img_edge.jpg", img_edge)

        Matrix = np.ones((4, 19), np.uint8)
        # Closed operation:Expansion before corrosion
        img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, Matrix)
        #open operation
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)
        cv2.imwrite("tmp/img_xingtai.jpg", img_edge2)
        return img_edge2, oldimg
    
    def img_only_color(self, filename, oldimg, img_contours):
        """
        :param filename: 图像文件
        :param oldimg: 原图像文件
        :return: 识别到的字符、定位的车牌图像、车牌颜色
        """
        pic_hight, pic_width = img_contours.shape[:2] # #Take the height and width of the color picture

        lower_blue = np.array([100, 110, 110])
        upper_blue = np.array([130, 255, 255])
        lower_yellow = np.array([15, 55, 55])
        upper_yellow = np.array([50, 255, 255])
        lower_green = np.array([50, 50, 50])
        upper_green = np.array([100, 255, 255])

        # BGR ---> HSV
        hsv = cv2.cvtColor(filename, cv2.COLOR_BGR2HSV)
        # Use the cv2.inRange function to set the threshold value and remove the background part
        # Parameter 1: the original image
        # Parameter 2: below the value in the image, the image value becomes 0
        # Parameter 3: Above the value in the image, the image value becomes 0
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_yellow, upper_green)

        # Image arithmetic operations Bitwise operations Bitwise operations are： AND， OR， NOT， XOR 等
        output = cv2.bitwise_and(hsv, hsv, mask=mask_blue + mask_yellow + mask_green)
        # Find the corresponding color according to the threshold value

        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        Matrix = np.ones((20, 20), np.uint8)
        #use 20x20  
        img_edge1 = cv2.morphologyEx(output, cv2.MORPH_CLOSE, Matrix)  #close operation
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix) #open operation 

        card_contours = img_math.img_findContours(img_edge2)
        card_imgs = img_math.img_Transform(card_contours, oldimg, pic_width, pic_hight)
        colors, car_imgs = img_math.img_color(card_imgs)

        predict_result = []
        predict_str = ""
        roi = None
        card_color = None

        for i, color in enumerate(colors):

            if color in ("blue", "yello", "green"):
                card_img = card_imgs[i]

                try:
                    gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
                except:
                    print("gray转换失败")

                # Yellow and green license plate characters are darker than the background, and the blue plate is just the opposite,
                #so the yellow and green plates need to be reversed
                if color == "green" or color == "yello":
                    gray_img = cv2.bitwise_not(gray_img)
                ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                x_histogram = np.sum(gray_img, axis=1)

                x_min = np.min(x_histogram)
                x_average = np.sum(x_histogram) / x_histogram.shape[0]
                x_threshold = (x_min + x_average) / 2
                wave_peaks = img_math.find_waves(x_threshold, x_histogram)
                
                if len(wave_peaks) == 0:
                    # print("peak less 0:")
                    continue
                # Considering the horizontal direction, the largest wave crest is the license plate area
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])

                
                gray_img = gray_img[wave[0]:wave[1]]
                # Find vertical histogram crests
                row_num, col_num = gray_img.shape[:2]
                # Remove 1 pixel from the top and bottom edges of the license plate to avoid white edges affecting the threshold judgment
                gray_img = gray_img[1:row_num - 1]
                y_histogram = np.sum(gray_img, axis=0)
                y_min = np.min(y_histogram)
                y_average = np.sum(y_histogram) / y_histogram.shape[0]
                y_threshold = (y_min + y_average) / 5  # U and 0  Requirement threshold is small，else U and 0 wil Splited into two halves
                wave_peaks = img_math.find_waves(y_threshold, y_histogram)
                if len(wave_peaks) < 6:
                    # print("peak less 1:", len(wave_peaks))
                    continue

                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                max_wave_dis = wave[1] - wave[0]
                # judement it is the edge of the left license plate
                if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                    wave_peaks.pop(0)

                # Combination of separated Chinese characters
                cur_dis = 0
                for i, wave in enumerate(wave_peaks):
                    if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                        break
                    else:
                        cur_dis += wave[1] - wave[0]
                if i > 0:
                    wave = (wave_peaks[0][0], wave_peaks[i][1])
                    wave_peaks = wave_peaks[i + 1:]
                    wave_peaks.insert(0, wave)

                point = wave_peaks[2]
                point_img = gray_img[:, point[0]:point[1]]
                if np.mean(point_img) < 255 / 5:
                    wave_peaks.pop(2)

                if len(wave_peaks) <= 6:
                    # print("peak less 2:", len(wave_peaks))
                    continue
                # print(wave_peaks)
                
                # wave_peaks  License plate characters Type list Contains 7 (start of horizontal coordinate, end of horizontal coordinate)



                part_cards = img_math.seperate_card(gray_img, wave_peaks)

                for i, part_card in enumerate(part_cards):
                    # May be the rivets that fix the license plate

                    if np.mean(part_card) < 255 / 5:
                        # print("a point")
                        continue
                    part_card_old = part_card

                    w = abs(part_card.shape[1] - SZ) // 2

                    part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
                    
                    part_card = img_recognition.preprocess_hog([part_card])
                    if i == 0:
                        resp = self.modelchinese.predict(part_card)
                        charactor = img_recognition.provinces[int(resp[0]) - PROVINCE_START]
                    else:
                        resp = self.model.predict(part_card)
                        charactor = chr(resp[0])
                    # judgment the last number is the edge of the license plate, assuming that the edge of the license plate is considered to be 1
                    if charactor == "1" and i == len(part_cards) - 1:
                        if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1 Too fine, think edge
                            continue
                    predict_result.append(charactor)
                    predict_str = "".join(predict_result)

                roi = card_img
                card_color = color
                break
        cv2.imwrite("tmp/img_caijian.jpg", roi)
        return predict_str, roi, card_color  # Recognized characters, positioned license plate image, license plate color
