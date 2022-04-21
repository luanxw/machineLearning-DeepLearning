from anyio import get_all_backends
from flask import Flask, jsonify, request,send_file,Response
from urllib3 import Retry
import uuid
import torchvision.transforms as transforms
from PIL import Image
import os
import io
import base64
import json

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor


app = Flask(__name__)
path = 'HW06/checkpoints'

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
class Generator(nn.Module):
    def __init__(self, in_dim,out_dim=64):
        """
        Input shape: (N, in_dim)
        Output shape: (N, 3, 64, 64)
        """
        super(Generator, self).__init__()
        def dconv_bn_relu(in_dim, out_dim):
          return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=5, stride =2, padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, out_dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(out_dim * 8 * 4 * 4),
            nn.ReLU()
        )
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(out_dim * 8, out_dim * 4),
            dconv_bn_relu(out_dim * 4, out_dim * 2),
            dconv_bn_relu(out_dim * 2, out_dim),
            nn.ConvTranspose2d(in_channels=out_dim, out_channels =3, kernel_size=5, stride =2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self,x):
        y = self.l1(x)
        # 把数据变成指定纬度
        # import torch
        # a = torch.arange(0,20)	#此时a的shape是(1,20)
        # a.view(4,5).shape		    #输出为(4,5)
        # a.view(-1,5).shape		#输出为(4,5)
        # a.view(4,-1).shape		#输出为(4,5)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y

class Discriminator(nn.Module):
    def __init__(self, in_dim,out_dim=64):
        super(Discriminator,self).__init__()
        def dconv_bn_relu(in_dim, out_dim):
          return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=5, stride =2, padding=2,),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2),
            )
        """  WGAN需要一处最后一层的Sigmoid """
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, out_dim,  kernel_size=5, stride =2, padding=2,), 
            nn.LeakyReLU(0.2),
            dconv_bn_relu(out_dim, out_dim * 2),
            dconv_bn_relu(out_dim * 2, out_dim * 4),
            dconv_bn_relu(out_dim * 4, out_dim * 8),
            nn.Conv2d(out_dim * 8, 1, 4),
            ## 比WGAN多了一个激活函数
            nn.Sigmoid(), 
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.ls(x)
        # 把数据变成一维的
        y = y.view(-1)
        return y

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

D = Discriminator(3)
D.load_state_dict(torch.load(os.path.join(path,'D.pth')))
D.eval()
D.to('cpu')
G = Generator(100)
G.load_state_dict(torch.load(os.path.join(path, 'G.pth')))
G.eval()
G.to('cpu')

def preprocess(img):
    formatImage = transforms.Compose([
            # transforms.ToPILImage(),
            # 输入图片128X128、 模型是64X64的输出、所以这里先变成64X64
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
    return formatImage(img)

# @app.route("/disc",methods=["POST",'GET'])
# def show():
#     result = {'success': False,
#             "msg":"success"
#             }
#     if request.method == 'POST':
#         if request.files.get('image'):
#             image = request.files['image'].read()
#             image = Image.open(io.BytesIO(image))  # 将字节对象转为Byte字节流数据
#             # 转换图片格式
#             image = preprocess(image)
#             #判断图片
#             result['predictions'] = D(image).numpy()
#     return result

@app.route("/ping",methods=["GET"])
def ping():
    data = {'success': False,
            "data": "",
            "msg":"服务没问题👌 !"
            }

    return jsonify(data)


@app.route("/gan",methods = ["GET","POST"])
def run():
    z_sample = torch.randn(1, 100).to('cpu')
    imgs_sample = (G(z_sample).data + 1) / 2.0
    imgs_sample = imgs_sample.squeeze(0)
    filename = os.path.join('HW06/logs', 'gan.jpg')
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)
    return send_file('logs/gan.jpg',mimetype='image/png')
if __name__ == "__main__":
    # app.run(host="127.0.0.1",port=5555,debug=False)
    from gevent import pywsgi
    server = pywsgi.WSGIServer(('127.0.0.1',5555),app)
    server.serve_forever()