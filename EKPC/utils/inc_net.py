import copy

from easydict import EasyDict
from network import vision_transformer_adapter
from network.classifier import SimpleContinualLinear
from torch import nn


def get_convnet(args, pretrained=True):
    constructors = {
        "pretrained_vit_b16_224_in21k_adapter": vision_transformer_adapter.vit_base_patch16_224_in21k_adapter,
        "pretrained_vit_b16_224_dino1k_adapter": vision_transformer_adapter.pretrained_vit_b16_224_dino1k_adapter,
    }
    name = args["convnet_type"].lower()
    if name not in constructors:
        raise ValueError(f"Unsupported backbone: {name}")
    tuning_config = EasyDict(
        ffn_adapt=True,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=args["ffn_num"],
        d_model=768,
    )
    model = constructors[name](
        pretrained=pretrained,
        checkpoint_path=args.get("pretrained_path"),
        num_classes=0,
        global_pool=False,
        drop_path_rate=0.0,
        tuning_config=tuning_config,
    )
    model.out_dim = 768
    return model.eval()


class SimpleVitNet(nn.Module):
    def __init__(self, args, pretrained=True):
        super().__init__()
        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def update_fc(self, nb_classes):
        if self.fc is None:
            self.fc = SimpleContinualLinear(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=False)

    def extract_vector(self, x):
        return self.convnet(x)

    def forward(self, x):
        features = self.extract_vector(x)
        outputs = self.fc(features)
        outputs["features"] = features
        return outputs

    def ca_forward(self, x):
        return self.fc(x)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        self.requires_grad_(False)
        return self.eval()
