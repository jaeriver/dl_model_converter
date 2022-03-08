import torchvision.models as models
import torch
# models_detail = {
#     'resnet50':models.resnet50(pretrained=True),
#     'inception_v3':models.inception_v3(pretrained=True),
#     'mobilenet_v2':models.mobilenet_v2(pretrained=True),
#     'vgg16' : models.vgg16(pretrained=True)
# }

models.alexnet(pretrained=True)
torch.save(models,'alexnet.pt')
models.resnet50(pretrained=True)
torch.save(models,'resnet50.pt')
models.inception_v3(pretrained=True)
torch.save(models,'inception_v3.pt')
models.mobilenet(pretrained=True)
torch.save(models,'mobilenet.pt')
models.mobilenet_v2(pretrained=True)
torch.save(models,'mobilenet_v2.pt')
models.vgg16(pretrained=True)
torch.save(models,'vgg16.pt')
models.vgg19(pretrained=True)
torch.save(models,'vgg19.pt')
