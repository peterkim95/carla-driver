import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class PilotNet(nn.Module):
    """
        Nvidia's steering predictor (https://arxiv.org/abs/1604.07316, https://arxiv.org/abs/1704.07911)
        with VisualBackProp (https://arxiv.org/abs/1611.05418)
    """
    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1)

        self.fc1 = nn.Linear(18 * 64, 1164)
        self.fc2 = nn.Linear(18 * 64, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 1)

        self.dconv1 = nn.ConvTranspose2d(1, 1, 3, stride=1, bias=False)
        self.dconv1.weight.data = torch.ones((1,1,3,3)) # TODO: make sure they are not learned!
        self.dconv1.weight.requires_grad = False
        self.dconv2 = nn.ConvTranspose2d(1, 1, 5, stride=2, bias=False)
        self.dconv2.weight.data = torch.ones((1,1,5,5))
        self.dconv2.weight.requires_grad = False
        self.visual_mask = None

    def forward(self, x):
        x = F.relu(self.conv1(x))
        avg_fm1 = torch.mean(x, dim=1, keepdim=True)
        x = F.relu(self.conv2(x))
        avg_fm2 = torch.mean(x, dim=1, keepdim=True)
        x = F.relu(self.conv3(x))
        avg_fm3 = torch.mean(x, dim=1, keepdim=True)
        x = F.relu(self.conv4(x))
        avg_fm4 = torch.mean(x, dim=1, keepdim=True)
        x = F.relu(self.conv5(x))

        avg_fm5 = torch.mean(x, dim=1, keepdim=True)
        deconvolved_fm5 = self.dconv1(avg_fm5) # TODO: set weight to 1 and bias to 0

        pw_fm = deconvolved_fm5 * avg_fm4 # pointwise multiplication
        deconvolved_fm4 = self.dconv1(pw_fm)

        pw_fm = deconvolved_fm4 * avg_fm3
        deconvolved_fm3 = self.dconv2(pw_fm)

        # avg_fm2 is (14,47) but deconvolved_fm3 is (13,47) so pad deconvolved
        deconvolved_fm3 = nn.ZeroPad2d((0,0,1,0))(deconvolved_fm3)
        pw_fm = deconvolved_fm3 * avg_fm2
        deconvolved_fm2 = self.dconv2(pw_fm)

        deconvolved_fm2 = nn.ZeroPad2d((1,0,0,0))(deconvolved_fm2)
        pw_fm = deconvolved_fm2 * avg_fm1
        deconvolved_fm1 = self.dconv2(pw_fm)

        deconvolved_fm1 = nn.ZeroPad2d((1,0,1,0))(deconvolved_fm1)
        self.visual_mask = deconvolved_fm1

        x = x.view(-1, 18 * 64) # flatten

        # TODO: add dropout to fc layers
        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        return x

def get_truncated_transform():
    transform = transforms.Compose([
        transforms.CenterCrop((160, 320)),
        transforms.Resize((66, 200)),
    ])
    return transform

def get_transform():
    transform = transforms.Compose([
        transforms.CenterCrop((160, 320)),
        transforms.Resize((66, 200)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform
