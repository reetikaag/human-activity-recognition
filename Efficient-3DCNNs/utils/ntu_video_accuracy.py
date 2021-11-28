from eval_ucf101 import UCFclassification
from eval_kinetics import KINETICSclassification



ucf_classification = UCFclassification('/home/shared/workspace/Resnet3D/3D-ResNets-PyTorch/data/ntu_01.json',
                                       '/home/shared/workspace/human-activity-recognition/Efficient-3DCNNs/data/results/resnet_101_50_ccrop/val2.json',
                                       subset='validation', top_k=1)
ucf_classification.evaluate()
print(ucf_classification.hit_at_k)


#kinetics_classification = KINETICSclassification('../annotation_Kinetics/kinetics.json',
#                                       '../results/val.json',
#                                       subset='validation',
#                                       top_k=1,
#                                       check_status=False)
#kinetics_classification.evaluate()
#print(kinetics_classification.hit_at_k)
