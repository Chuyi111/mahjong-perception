import torch, torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

img=128
tf = transforms.Compose([transforms.Resize((img,img)),
                         transforms.ToTensor(),
                         transforms.Normalize([0.5]*3,[0.25]*3)])
tr = datasets.ImageFolder("data/tiles/train", transform=tf)
subset_idx = list(range(min(64, len(tr))))
ds = Subset(tr, subset_idx)
loader = DataLoader(ds, batch_size=16, shuffle=True)
m = models.mobilenet_v3_small(weights=None)
m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(tr.classes))
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
loss = nn.CrossEntropyLoss()
m.train()
for step in range(200):
    x,y = next(iter(loader))
    opt.zero_grad()
    out = m(x); l = loss(out,y); l.backward(); opt.step()
acc = (m(next(iter(loader))[0]).argmax(1) == next(iter(loader))[1]).float().mean().item()
print("overfit-acc ~ should be >0.95:", acc)
