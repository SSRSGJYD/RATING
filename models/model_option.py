from .networks.cfn_net import CFNNet
from .networks.GSDopplerFeatureFusion_net import GSDopplerFeatureFusionNet

model_dict = {
    'cfn': {'name': CFNNet,
            'params': ['backbones', 'archs', 'heads', 'in_channels', 'frozen_stages', 'layer_norm_type', 'activation_type']
            },
    'GSDopplerFeatureFusion': {'name': GSDopplerFeatureFusionNet,
            'params': ['backbones', 'archs', 'share_weight', 'heads', 'in_channels', 'frozen_stages', 'output_layers', 'layer_norm_type', 'activation_type']
            }
    }
