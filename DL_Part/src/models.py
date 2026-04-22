import torch.nn as nn
import torchvision.models as models

def get_classification_model(model_name, num_classes=3, fine_tune_mode="full"):
    """Get a pretrained model for classification"""
    model = None

    # ResNet Family
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnext50":
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnext101":
        model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    # EfficientNet Family
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_v2_m":
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # DenseNet Family
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "densenet161":
        model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "densenet169":
        model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # MobileNet Family
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    # ConvNeXt Family
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_name == "convnext_small":
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_name == "convnext_base":
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    # Vision Transformers
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == "vit_b_32":
        model = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == "vit_l_16":
        model = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == "swin_t":
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif model_name == "swin_s":
        model = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif model_name == "swin_b":
        model = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif model_name == "maxvit_t":
        model = models.maxvit_t(weights=models.MaxVit_T_Weights.DEFAULT)
        model.classifier[5] = nn.Linear(model.classifier[5].in_features, num_classes)

    # RegNet Family
    elif model_name == "regnet_y_400mf":
        model = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "regnet_y_800mf":
        model = models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "regnet_y_1_6gf":
        model = models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # Apply fine-tuning strategy
    if fine_tune_mode == "freeze":
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
        elif hasattr(model, 'heads'):
            for param in model.heads.parameters():
                param.requires_grad = True
    elif fine_tune_mode == "partial":
        total_layers = len(list(model.parameters()))
        layers_to_freeze = total_layers // 2
        for i, param in enumerate(model.parameters()):
            param.requires_grad = i >= layers_to_freeze

    return model

def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
