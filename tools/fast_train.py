import argparse
import logging
import mmcv
import os
import platform
import subprocess
import sys
import time
import yaml

from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--max-keep-ckpts', help='max keep checkpoints file number', type=int, default=-1)
    parser.add_argument(
        '--checkpoint-interval', help='interval epochs of saving checkpoint', type=int, default=5)
    parser.add_argument(
        '--record-pid',
        action='store_true',
        help='whether to record the process id'
    )

    args = parser.parse_args()
    return args


def get_config_file_depend_model_type(model_type):
    if model_type == 'mask_rcnn':
        config_filepath = '../configs/mechmind/deploy/mask_rcnn_r50_fpn_train_example.py'
    elif model_type == 'point_rend':
        config_filepath = '../configs/mechmind/deploy/point_rend_r50_caffe_fpn_train_example.py'
    elif model_type == 'keypoint_mask':
        config_filepath = '../configs/mechmind/deploy/kp_mask_dla34_rend_mask.py'

    return config_filepath


def get_gpu_availability():
    nvidia_output = bytes.decode(subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv']))
    if len([str(i) for i, j in enumerate(nvidia_output.splitlines()[1:]) if int(j.split(' ')[0])
            <= 1000]) == len([str(i) for i, j in enumerate(nvidia_output.splitlines()[1:])]):
        return len([str(i) for i, j in enumerate(nvidia_output.splitlines()[1:])])
    else:
        return 0


def update_key_value(config, key, new_value):
    for attr, value in config.items():
        if attr == key:
            config[attr] = new_value
        else:
            if isinstance(value, dict):
                update_key_value(config[attr], key, new_value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        update_key_value(item, key, new_value)


def get_dataset_class_number(dataset_root):

    class_name_set = set()
    for dataset in ['train2014', 'val2014']:
        curr_train_coco = COCO(
            os.path.join(
                dataset_root,
                'annotations',
                'instances_{}.json'.format(dataset)))
        class_name_set = class_name_set.union(set([curr_train_coco.cats[cat]['name']
                                                   for cat in sorted(curr_train_coco.cats)]))

    return len(class_name_set)


def generate_python_config(yaml_filepath, if_do_eval, if_record_pid, max_keep_ckpts, checkpoint_interval):
    available_gpu = get_gpu_availability()
    if not available_gpu:
        logging.warning(
            "PAY ATTENTION!!!! Not all GPU(s) is(are) available. Please check your machine.")
        sys.exit(0)

    with open(yaml_filepath, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config['gpu_count'] = available_gpu
    config['no_validate'] = if_do_eval
    config['record_pid'] = if_record_pid

    if platform.system() == 'Windows':
        config['data_root'] = config['data_root'].replace('\\', '/')
    if not config['data_root'].endswith('/'):
        config['data_root'] += '/'

    num_class = get_dataset_class_number(config['data_root'])

    root_dir, yaml_filename = os.path.split(yaml_filepath)
    base_config = mmcv.Config.fromfile(get_config_file_depend_model_type(config['model_type']))

    if "file_path" in config:
        base_config["file_path"] = config["file_path"]
    if "total_epochs" in config:
        total_epochs = config["total_epochs"]
        base_config["total_epochs"] = total_epochs
        if total_epochs < 60:
            base_config['lr_config'] = dict(
                policy='step',
                step=[int(total_epochs*0.75), int(total_epochs*0.90)])
        else:
            base_config['lr_config'] = dict(
                policy='step',
                warmup='linear',
                warmup_iters=10,
                warmup_ratio=0.001,
                warmup_by_epoch=True,
                step=[int(total_epochs*0.75), int(total_epochs*0.90)])

    origin_data_root = base_config['data_root']
    for _ in ['train', 'test', 'val']:
        base_config['data'][_]['ann_file'] = base_config['data'][_]['ann_file'].replace(
            origin_data_root, config['data_root'])
        base_config['data'][_]['img_prefix'] = base_config['data'][_]['img_prefix'].replace(
            origin_data_root, config['data_root'])

    update_key_value(base_config, 'num_classes', num_class)
    update_key_value(base_config, 'hm', num_class)

    if config['model_type'] == 'mask_rcnn':
        with open(os.path.join(config['data_root'], 'image_rgb_mean.txt'), 'r') as f:
            mean_std = f.readlines()
        mean_value = [
            float(value) for value in mean_std[0].replace(
                ']\n', '').replace(
                '[', '').split(',')]
        update_key_value(base_config, 'mean', mean_value)
        std_value = [
            float(value) for value in mean_std[1].replace(
                ']', '').replace(
                '[', '').split(',')]
        update_key_value(base_config, 'std', std_value)

    base_config['checkpoint_config']['max_keep_ckpts'] = max_keep_ckpts
    base_config['checkpoint_config']['interval'] = checkpoint_interval

    project_name, _ = os.path.splitext(yaml_filename)
    project_name = project_name + "_" + time.strftime('%Y%m%d_%H%M')

    base_config.dump(os.path.join(root_dir, project_name + '.py'))

    return config, os.path.join(root_dir, project_name + '.py')


def start_training(config, py_config_path):
    work_dir = os.path.join(config['data_root'], 'models')

    if platform.system() == 'Windows':
        start_cmd = 'python train.py'
        gpu_count = ''
    else:
        start_cmd = './dist_train.sh'
        gpu_count = config['gpu_count']

    assert not (config['resume_training'] and config['finetune_training']
                ), "resume_training and finetune_training can't be True at the same time"

    if config['resume_training']:
        resume_model = '--resume-from {}'.format(config['resume_from_which_model'])
    elif config['finetune_training']:
        resume_model = '--load-from {}'.format(config['finetune_from_which_model'])
    else:
        resume_model = ''

    if config['no_validate']:
        no_validate = '--no-validate'
    else:
        no_validate = ''

    if config['record_pid']:
        record_pid = '--record-pid'
    else:
        record_pid = ''

    training_cmd = '{} {} {} --work-dir {} {} {} {}'.format(
        start_cmd, py_config_path, gpu_count, work_dir, resume_model, no_validate, record_pid)
    print(training_cmd)
    os.system(training_cmd)


if __name__ == '__main__':
    args = parse_args()
    config, py_config_path = generate_python_config(
        args.config, args.no_validate, args.record_pid, args.max_keep_ckpts, args.checkpoint_interval)
    start_training(config, py_config_path)
