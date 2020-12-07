'''
@Descripttion: 
@version: 
@Author: 周耀海 u201811260@hust.edu.cn
@LastEditTime: 2020-07-23 10:04:18
'''
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import torch
import numpy as np
import cv2
import json

# from models.imagenet.resnet_sge import sge_resnet50
# from models.imagenet.resnet_old import old_resnet50
# from resnet import resnet18

LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json' # 下载label
# 使用本地的图片和下载到本地的labels.json文件
LABELS_PATH = "labels1.json"
# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 2
# 选择使用的网络
if model_id == 1:
 net = models.squeezenet1_1(pretrained=True)
 finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
 net=models.resnet50()
#  net.cuda()
 net=torch.nn.DataParallel(net).cuda()


 checkpoint=torch.load("checkpoint_cls_11.pth.tar")
 c = checkpoint['state_dict']
#  c=checkpoint['state_dict']
#  print(checkpoint)
 net.module.fc = torch.nn.Linear(2048,2,True)
 net.cuda()
 net.load_state_dict(c)

 finalconv_name = 'layer4'
#  print(net._modules)

# elif model_id == 3:
#  net = models.densenet161(pretrained=True)
#  finalconv_name = 'features'
net.eval()
# print(net)
# 获取特定层的feature map
# hook the feature extractor
features_blobs = []

def hook_feature(module, input, output): # input是注册层的输入 output是注册层的输出
    print("hook input",input[0].shape)
    print("hook output",output[0].shape)

    features_blobs.append(output.data.cpu().numpy())
# 对layer4层注册，把layer4层的输出append到features里面
net._modules["module"]._modules.get(finalconv_name).register_forward_hook(hook_feature) # 注册到finalconv_name,如果执行net()的时候，
                                                            # 会对注册的钩子也执行，这里就是执行了 layer4()
# print(net._modules)
# 得到softmax weight,
params = list(net.parameters()) # 将参数变换为列表 按照weights bias 排列 池化无参数
# print(param.shape for param in params)

weight_softmax = np.squeeze(params[-2].data.cpu().numpy()) # fc 层的参数 （weights，-1是bias）
print("param[-2] shape=========",params[-2].shape)
print("weight_softmax shape=========",weight_softmax.shape)

# 生成CAM图的函数，完成权重和feature相乘操作，最后resize成上采样
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape # 获取feature_conv特征的尺寸
    output_cam = []
    # class_idx为预测分值较大的类别的数字表示的数组，一张图片中有N类物体则数组中N个元素
    for idx in class_idx:
    # weight_softmax中预测为第idx类的参数w乘以feature_map(为了相乘，故reshape了map的形状)

        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w))) # 把原来的相乘再相加转化为矩阵
                                                                    # w1*c1 + w2*c2+ .. -> (w1,w2..) * (c1,c2..)^T -> (w1,w2...)*((c11,c12,c13..),(c21,c22,c23..))
        print("weight_softmax[idx] shape===========",weight_softmax[idx].shape)
        print("feature map shape=========",feature_conv.reshape((nc, h*w)).shape)
        print("cam shape===========",cam.shape)
        # 将feature_map的形状reshape回去
        cam = cam.reshape(h, w)
        print("cam reshape shape===========",cam.shape)

        # 归一化操作（最小的值为0，最大的为1）
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        # 转换为图片的255的数据
        cam_img = np.uint8(255 * cam_img)
        # resize 图片尺寸与输入图片一致
        output_cam.append(cv2.resize(cam_img, size_upsample))
        # print("output_cam shape=========",np(output_cam).shape)
    return output_cam
# 数据处理，先缩放尺寸到（224*224），再变换数据类型为tensor,最后normalize
normalize = transforms.Normalize(
 mean=[0.485, 0.456, 0.406],
 std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
 transforms.Resize((224,224)),
 transforms.ToTensor(),
 normalize
])

# img_pil = Image.open('motorcycle2.jpg')
# img = cv2.imread('motorcycle2.jpg')

# img_pil = Image.open('motorcycle.jpg')
# img = cv2.imread('motorcycle.jpg')

# img_pil = Image.open('car.jpg')
# img = cv2.imread('car.jpg')

img_pil = Image.open('car2.jpg')
img = cv2.imread('car2.jpg')



img_tensor = preprocess(img_pil)
# 处理图片为Variable数据
img_variable = Variable(img_tensor.unsqueeze(0))
# 将图片输入网络得到预测类别分值
logit = net(img_variable)

# 使用本地的 LABELS_PATH
with open(LABELS_PATH) as f:
    data = json.load(f).items()
classes = {int(key):value for (key, value) in data}
# 使用softmax打分
h_x = F.softmax(logit, dim=1).data.squeeze() # 分类分值
# 对分类的预测类别分值排序，输出预测值和在列表中的位置
probs, idx = h_x.sort(0, True)
# 转换数据类型
probs = probs.cpu().numpy()
idx = idx.cpu().numpy()


print("features_blobs[0] shape===========",features_blobs[0].shape)

# 输出与图片尺寸一致的CAM图片
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])


print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
# 将图片和CAM拼接在一起展示定位结果结果
img = cv2.resize(img,(224,224))
height, width, _ = img.shape
# 生成热度图
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
# cv2.imwrite('heatmap.jpg', heatmap)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM.jpg', result)