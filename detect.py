# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
import argparse
import collections
import json

import common
import cv2
import numpy as np
import os
from PIL import Image
import re
import time
import tflite_runtime.interpreter as tflite

import rclpy
from rclpy.node import Node

from std_msgs.msg import String

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])


def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
        lines = (p.match(line).groups() for line in f.readlines())
        return {int(num): text.strip() for num, text in lines}


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()


def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]


def draw_label(img, text, pos, bg_color, scale):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


sensor_data = None

class rvrNode(Node):
    
    def __init__(self):
        super().__init__('sphero_node')

        self.subscription = self.create_subscription(
            String,
            'sensors',  # listen on drive channel
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    # listener that updates sensor_data
    def listener_callback(self, msg):
        global sensor_data
        sensor_data = json.loads(msg.data)


def main():
    default_model_dir = 'all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default=0)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = load_labels(args.labels)

    print('Initializing RVR node')
    rclpy.init(args=None)
    ros = rvrNode()

    width = 1280
    height = 720

    print("starting video capture")
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    gst_str = 'appsrc ! videoconvert ! v4l2sink sync=false'

    # output destination selection
    out = cv2.VideoWriter(gst_str, 0, 30, (width, height), True)  # /dev/video1 stream
    # out = cv2.VideoWriter('output.avi', fourcc, 30, (width, height))  # output to file

    print("loop starting")
    try:
        prev_frame_time = None
        curr_frame_time = time.time()
        while cap.isOpened():
            prev_frame_time = curr_frame_time
            curr_frame_time = time.time()
            fps = int(1 / (curr_frame_time - prev_frame_time))

            ret, frame = cap.read()
            if not ret:
                break

            cv2_im = frame
            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im_rgb)


            common.set_input(interpreter, pil_im)
            interpreter.invoke()
            objs = get_output(interpreter, score_threshold=args.threshold, top_k=args.top_k)
            cv2_im = append_objs_to_img(cv2_im, objs, labels)

            # draw sensor data
            rclpy.spin_once(ros, timeout_sec=0.001)
            draw_label(cv2_im, str(fps) + 'fps', (30, 30), (255, 255, 255), 0.6)
            if sensor_data is not None:
                i = 50
                for key in sensor_data:
                    sensor_label = "%s:" % key
                    for sub_key in sensor_data[key]:
                        if sub_key != "is_valid":
                            sensor_label += " %s: %s" % (sub_key, str(sensor_data[key][sub_key]))
                    draw_label(cv2_im, sensor_label, (30, i), (255, 255, 255), 0.6)
                    i += 20

            out.write(cv2_im)

            # pi dies for some reason if I dont limit the fps - fix this?
            time.sleep(0.1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("keyboard Interrupt")

    cap.release()
    cv2.destroyAllWindows()


def append_objs_to_img(cv2_im, objs, labels):
    height, width, channels = cv2_im.shape
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y1),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return cv2_im


if __name__ == '__main__':
    main()
