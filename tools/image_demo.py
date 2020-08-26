import cv2
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import os

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    save_pth = os.path.join(args.img, args.config.split('/')[-1].split('.')[0])
    os.mkdir(save_pth)
    img_list = os.listdir(args.img)
    for im_str in img_list:
        if im_str.endswith('.jpg'):
            _im = os.path.join(args.img, im_str)
            img = cv2.imread(_im, cv2.IMREAD_UNCHANGED)
            # test a single image
            try:
                result = inference_detector(model, img)
                show_result_pyplot(model, _im, result, im_str, save_pth, score_thr=args.score_thr)
            except Exception as e:
                print('[error]', _im)
                pass
            # show the results

if __name__ == '__main__':
    main()
