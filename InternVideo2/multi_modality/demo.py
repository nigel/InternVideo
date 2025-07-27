import numpy as np
import os
import io
import cv2

import torch

from utils.config import (Config,
                    eval_dict_leaf)

from demo.utils import (retrieve_text,
                  retrieve_video,
                  _frame_from_video,
                  setup_internvideo2)
from IPython import embed

from tasks.retrieval_utils import (extract_text_feats, extract_vision_feats)
import os

video_frames = []
directory = "/Users/nchen/analyze/sample/"

text = "person walking out of a door"

for f in os.listdir(directory):
    if not os.path.isfile(os.path.join(directory, f)):
        continue
    video = cv2.VideoCapture(directory + f)
    video_frames.append([x for x in _frame_from_video(video)])
    print(f)
    if len(video_frames) == 5:
        break

config = Config.from_file('demo/internvideo2_stage2_config.py')
config = eval_dict_leaf(config)

intern_model, tokenizer = setup_internvideo2(config)
retrieve_video(video_frames, text, model=intern_model, topk=5, config=config)
