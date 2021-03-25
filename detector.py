import cv2
import torch
import numpy as np
from random import randint

try:
    from models.models.experimental import attempt_load
    from models.utils.general import non_max_suppression, scale_coords, letterbox
    from models.utils.torch_utils import select_device
except:
    from models.experimental import attempt_load
    from utils.general import non_max_suppression, scale_coords, letterbox
    from utils.torch_utils import select_device


class Detector(object):
    def __init__(self, img_size=480, threshold=0.4, weights='weights/yolov5s.pt'):
        self.img_size = img_size
        self.threshold = threshold
        self.max_frame = 160
        self.weights = weights
        self._init_model()
        self._run_once()

    def _init_model(self):
        print('Init model ...')
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        if self.device != 'cpu':
            model.half()

        self.model = model
        self.names = model.module.names if hasattr(model, 'module') else model.names
        self.colors = [
            (randint(0, 255), randint(0, 255), randint(0, 255)) for _ in self.names
        ]

    def preprocess(self, img):
        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float()
        if self.device != 'cpu':
            img = img.half()  # 半精度
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def plot_bboxes(self, image, bboxes, line_thickness=None):
        tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
        for (x1, y1, x2, y2, cls_id, conf) in bboxes:
            color = self.colors[self.names.index(cls_id)]
            c1, c2 = (x1, y1), (x2, y2)
            cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, '{} ID-{:.2f}'.format(cls_id, conf), (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return image

    def _run_once(self):
        im = torch.randn((1, 3, self.img_size, self.img_size)).to(self.device)
        if self.device != 'cpu':
            im = im.half()
        self.model(im, augment=False)

    def detect(self, im):
        im0, img = self.preprocess(im)

        pred = self.model(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.3)

        pred_boxes = []
        image_info = {}
        count = 0
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    label = self.names[int(cls_id)]
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append((x1, y1, x2, y2, label, conf))

                    count += 1
                    key = '{}-{:02}'.format(label, count)
                    image_info[key] = {
                        'xmin': x1,
                        'ymin': y1,
                        'xmax': x2,
                        'ymax': y2,
                        'label': label,
                        'confidence': np.round(float(conf), 3),
                    }
        if not pred_boxes:
            im = None
            status = 2
        else:
            im = self.plot_bboxes(im, pred_boxes)
            status = 1
        return status, im, image_info

