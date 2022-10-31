import torch
import numpy as np
import cv2
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import Annotator, colors

WEIGHTS_PATH = r"yolov5s.pt"


def get_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    half = device.type != 'cpu'
    weights = WEIGHTS_PATH
    model = DetectMultiBackend(weights, device=device, dnn=False, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    model.eval()
    if half:
        model.half()
    return model, device, half, stride, names


def get_img(img0):
    img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img = img / 255.0
    if len(img.shape) == 3:
        img = img[None]
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
    det = pred[0]
    im0 = img0.copy()
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    annotator = Annotator(im0, line_width=1, example=str(names))
    xywh_list = []
    if len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            xywh_list.append(xywh)
            c = int(cls)  # integer class
            label = None if True else (names[c] if True else f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))
    im0 = annotator.result()
    return im0, xywh_list


def getVector(xywh_list):
    global GOAL_POINT
    vectors = []
    distances = []
    for index, xywh in enumerate(xywh_list, 0):
        x, y, _, _ = xywh
        x *= frame_width
        y *= frame_height
        # x *= h
        # y *= w
        vectorx = int(x) - GOAL_POINT[0]
        vectory = int(y) - GOAL_POINT[1]
        vector = (vectorx, vectory)
        distance = int((vectorx ** 2 + vectory ** 2) ** (1 / 2))
        # if distance<200:
        #     detectpoints.append(center)
        #     cv2.circle(imgFinal, center, 10, (0, 255, 0), 2)
        vectors.append(vector)
        distances.append(distance)
        if len(distances) != 0:
            cv2.circle(img, (int(x), int(y)), 10, (0, 255, 0), 1)
            cv2.line(img, GOAL_POINT, (vectors[index][0] + GOAL_POINT[0], vectors[index][1] + GOAL_POINT[1]),
                     (0, 255, 0), 1)
    if len(distances) == 0:
        return "No objects were detected"
    else:
        mini = -1
        distances2 = distances.copy()
        for i in range(len(distances2) - 1):
            if distances2[i] < distances2[i + 1]:
                distances2[i + 1] = distances2[i]
            mini = i + 1
        mindistance = distances2[mini]
        for i in range(len(distances)):
            if mindistance == distances[i]:
                minindex = i
        cv2.line(img, GOAL_POINT, (vectors[minindex][0] + GOAL_POINT[0], vectors[minindex][1] + GOAL_POINT[1]),
                 (0, 0, 255), 3)
        cv2.circle(img, (vectors[minindex][0] + GOAL_POINT[0], vectors[minindex][1] + GOAL_POINT[1]), 10, (0, 0, 255),
                   3)
    return "Displacement vector is {}".format(vectors[minindex])


model, device, half, stride, names = get_model()
imgsz = check_img_size((640, 640), s=stride)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if cap.isOpened() == False:
        print("The camera failed to open")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    GOAL_POINT = (int(frame_width / 2), int(frame_height / 2))
    while cap.isOpened():
        t1, img = cap.read()
        if img is None:
            print("Cannot read")
            break
        img, xywh_list = get_img(img)
        vector = getVector(xywh_list)
        print(vector)
        cv2.circle(img, GOAL_POINT, 10, (0, 0, 255), 3)
        cv2.imshow('Detect', img)
        if cv2.waitKey(1) == 32:
            cap.release()
            cv2.destroyAllWindows()
