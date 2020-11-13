import torch.nn as nn
import torch.nn.functional as F
import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmcv.runner import  load_checkpoint
from . import builder
from .builder import DISTILLER





@DISTILLER.register_module()
class DetectionDistiller(BaseDetector):
    """Knowledge distillation segmentors.

    It typically consists of teacher_model, student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 discriminator=None,
                 distill_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 teacher_pretrained=None,):

        super(DetectionDistiller, self).__init__()
        # import pdb;pdb.set_trace()
        self.teacher = build_detector(teacher_cfg.model,train_cfg=teacher_cfg.train_cfg,test_cfg=teacher_cfg.test_cfg)
        self.init_weights_teacher(teacher_pretrained)

        
        self.teacher.eval()
        self.student= build_detector(student_cfg.model,train_cfg=student_cfg.train_cfg,test_cfg=student_cfg.test_cfg)

        if discriminator is not None:
            self.discriminator = builder.build_discriminator(discriminator)
        else:
            self.discriminator = None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.distill_losses = nn.ModuleDict()
        self.distill_cfg = distill_cfg
        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())
        def regitster_hooks(student_module,teacher_module):
            def hook_teacher_forward(module, input, output):
                    # import pdb; pdb.set_trace()
                # if isinstance(output,tuple):
                #     for lvl , m in enumerate(output):
                #         self.register_buffer(teacher_module + str(lvl),output )
                # else:
                    self.register_buffer(teacher_module,output)
                
            def hook_student_forward(module, input, output):

                # if isinstance(output,tuple):
                #     for lvl , m in enumerate(output):
                #         self.register_buffer(student_module + str(lvl),output )
                # else:
                    self.register_buffer( student_module,output )
            return hook_teacher_forward,hook_student_forward
        
        for item_loc in distill_cfg:
            
            
            
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')

            self.register_buffer(student_module,None)
            self.register_buffer(teacher_module,None)

            
            # assert type(teacher_modules[item_loc.teacher_module]) == type(student_modules[item_loc.student_module])

            # if isinstance(teacher_modules[item_loc.teacher_module],tuple):
            #     assert len(teacher_modules[item_loc.teacher_module]) == len(student_modules[item_loc.student_module])
            #     for lvl,tm,sm in enumerate( zip(teacher_module,student_module)):
            #         hook_teacher_forward,hook_student_forward = regitster_hooks(student_module + str(lvl),teacher_module + str(lvl),item_loc.output_hook)
            #         tm.register_forward_hook(hook_teacher_forward)
            #         sm.egister_forward_hook(hook_student_forward)
            # else:
            hook_teacher_forward,hook_student_forward = regitster_hooks(student_module ,teacher_module )
            teacher_modules[item_loc.teacher_module].register_forward_hook(hook_teacher_forward)
            student_modules[item_loc.student_module].register_forward_hook(hook_student_forward)

            for item_loss in item_loc.methods:
                loss_name = teacher_module + '_' + student_module + '_' + item_loss.type
                self.distill_losses[loss_name] = builder.build_distill_loss(item_loss)
    def base_parameters(self):
        return nn.ModuleList([self.student,self.distill_losses])
    def discriminator_parameters(self):
        return self.discriminator

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self.student, 'neck') and self.student.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self.student, 'roi_head') and self.student.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_bbox)
                or (hasattr(self.student, 'bbox_head') and self.student.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_mask)
                or (hasattr(self.student, 'mask_head') and self.student.mask_head is not None))

    def init_weights_teacher(self, path=None):
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')



    def forward_train(self, img, img_metas, **kwargs):
       

        with torch.no_grad():
            self.teacher.eval()
            # teacher_loss = self.teacher.backbone(img, img_metas,**kwargs)
            loss = self.teacher.extract_feat(img)
           
        student_loss = self.student.forward_train(img, img_metas, **kwargs)
        # import pdb; pdb.set_trace()

        
        
        buffer_dict = dict(self.named_buffers())
        for item_loc in self.distill_cfg:
            
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')
            # import pdb;pdb.set_trace()
            student_feat = buffer_dict[student_module]
            teacher_feat = buffer_dict[teacher_module]

            for item_loss in item_loc.methods:
                loss_name = teacher_module + '_' + student_module + '_' + item_loss.type
                if self.distill_losses[loss_name].discriminator:
                    teacher_D = self.discriminator(teacher_feat)
                    student_D = self.discriminator(student_feat)
                    student_loss['loss_' + loss_name] = self.distill_losses[loss_name](student_D,teacher_D)
                else:
                    student_loss['loss_' + loss_name] = self.distill_losses[loss_name](student_feat,teacher_feat)
        
        
        return student_loss
    
    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)
    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(img, img_metas, **kwargs)
    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)


