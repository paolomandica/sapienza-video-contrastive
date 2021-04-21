import resnet


encoder = resnet.resnet_3d_18(pretrained=True)
print(encoder)
