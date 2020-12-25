import os
import torch
from skimage import io, transform, color
from torchvision import transforms
from data_loader import RescaleT
from data_loader import ToTensorLab
from torch.autograd import Variable
from PIL import Image
import numpy as np
import uuid

model = torch.load('u-2-net.pt', map_location=torch.device('cpu'))
model.eval()


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)

    return dn

def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if (3 == len(label_3.shape)):
        label = label_3[:, :, 0]
    elif (2 == len(label_3.shape)):
        label = label_3

    if (3 == len(image.shape) and 2 == len(label.shape)):
        label = label[:, :, np.newaxis]
    elif (2 == len(image.shape) and 2 == len(label.shape)):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    sample = transform({'imidx': np.array([0,0]), 'image': image, 'label': label})

    return sample

def get_mask(original_img):
    torch.cuda.empty_cache()

    sample = preprocess(original_img)
    inputs_test = sample['image'].unsqueeze(0)
    inputs_test = inputs_test.type(torch.FloatTensor)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    d1,d2,d3,d4,d5,d6,d7= model(inputs_test)

    # normalization
    predict = d1[:,0,:,:]
    predict = normPRED(predict)
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    image = original_img # io.imread(original_img)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    # imo.save('output.png')

    del d1,d2,d3,d4,d5,d6,d7
    return imo


def get_cut(img):
    ref = Image.open("uploads/"+img)
    original_img = io.imread("uploads/"+img)
    mask = get_mask(original_img).convert("L")
    
    empty = Image.new("RGBA", ref.size, 0)
    img_u2 = Image.composite(ref, empty, mask)
    location = str(uuid.uuid4())+".png"
    img_u2.save("uploads/"+location)
    return location
