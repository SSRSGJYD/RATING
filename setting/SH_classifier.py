params = dict()
params['params'] = 'task label in_channels backbones archs frozen_stages heads lr'
params['values'] = ('GSDOPPLER', 'SH', '3,3', 'resnet,resnet', '18,18', 4, '0,1_2', 3e-4)