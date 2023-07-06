import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class SACNN(nn.Module):
    def __init__(self):
        super(SACNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.l1 = nn.Linear(27*27*32, 2)
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        x = self.bn1(self.pool1(self.conv1(x)))
        x = F.relu(x)
        x = self.bn2(self.pool2(self.conv2(x)))
        x = F.relu(x)
        x = self.bn3(self.pool3(self.conv3(x)))
        x = F.relu(x)
        x = x.view(-1,27*27*32)
        x = self.l1(x)
        x = F.relu(x)
        out = self.dropout(x)
        return out

class SACNNContainer(object):
    def __init__(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args
        self.setup_model()

    def setup_model(self):
        self.model = SACNN().to(self.device)
        self.model.load_state_dict(torch.load(self.args['sacnn_path']))
        self.model.eval()

    def predict(self, batch):
        batch = self.preprocess(batch)
        out = self.model(batch)
        predicted = torch.argmax(out, dim=1)
        return predicted.item()
    
    def preprocess(self, img):
        pre = transforms.Compose([
            transforms.Resize((216, 384)),
            transforms.CenterCrop((216, 216)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = pre(img).to(self.device)
        batch = torch.unsqueeze(img, 0)
        return batch