import cv2
import numpy as np

def show_labels_img(imgname):

    # img = cv2.imread(DATASET_PATH + imgname + ".jpg")
    img = cv2.imdecode(np.fromfile(DATASET_PATH + imgname + ".jpg", dtype=np.uint8), flags=cv2.IMREAD_COLOR) # 中文路径使用
    h, w = img.shape[:2]
    print(w,h)
    label = []
    with open("G:/大创项目/test_process_data/new_gt/"+imgname+".txt",'r') as flabel:
        for label in flabel:
            label = label.split(' ')
            label = [float(x.strip()) for x in label]
            print(CLASSES[int(label[8])])
            pt1 = (int(label[0]), int(label[1]))
            pt2 = (int(label[4]), int(label[5]))
            print(pt1)
            print(pt2)
            cv2.putText(img,CLASSES[int(label[8])],pt1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
            cv2.rectangle(img,pt1,pt2,(0,0,255,2))

    cv2.imwrite("./show_label_img.jpg",img)
if __name__ == '__main__':
    # CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    #    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    CLASSES=['Unknown', 'Plane', 'Ship']
    DATASET_PATH='G:/大创项目/test_process_data/new_train/'
    show_labels_img('P1500')

