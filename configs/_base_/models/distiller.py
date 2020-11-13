# model settings
discriminator = None
find_unused_parameters=True

distiller = dict(   
    type='DetectionDistiller',
    teacher_pretrained = '/home/nfs/em2/gaojianfei/distill/mmdetection/pretraind_model/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-e75f90c8.pth',
    discriminator = discriminator,
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                        
                         methods=[dict(type='CriterionPixelWiseLossLogits',
                                       tau = 3,
                                        inplanes = 256,
                                       weight = 5.0),
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.2.conv',
                         teacher_module = 'neck.fpn_convs.2.conv',
                         output_hook = True,
                         
                         methods=[dict(type='CriterionPixelWiseLossLogits',
                                       tau = 3,
                                       inplanes = 256,
                                       weight = 5.0),
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.1.conv',
                         teacher_module = 'neck.fpn_convs.1.conv',
                         output_hook = True,
                         
                         methods=[dict(type='CriterionPixelWiseLossLogits',
                                       tau = 3,
                                       inplanes = 256,
                                       weight = 5.0),
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.0.conv',
                         teacher_module = 'neck.fpn_convs.0.conv',
                         output_hook = True,
                         
                         methods=[dict(type='CriterionPixelWiseLossLogits',
                                       tau = 3,
                                       inplanes = 256,
                                       weight = 5.0),
                                ]
                        ),

                    
                   ]
    )

student_cfg = 'configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
teacher_cfg = 'configs/dcn/cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py'
# pretraind_model/           cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco-e75f90c8.pth


