from model import *

a = torch.randn(92)

net = TomoModel(92, 1)

out = net(a)

print(out.shape)