_base_ = [
    '../_base_/models/distiller.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,)