import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset
from PIL import Image

# 设置训练参数
dataroot = "/content/drive/MyDrive/VisDrone2019-Patch/awning-tricycle"
output_folder = "/content/drive/MyDrive/VisDrone2019-Generate/Generator/awning-tricycle"
batch_size = 64
image_size = 64
nc = 3  # 图像通道数，彩色是3
nz = 100  # 噪声维度
ngf = 64  # 生成器特征通道数
ndf = 64  # 判别器特征通道数
num_epochs = 25  # 训练轮数
lr = 0.0002
beta1 = 0.5  # Adam优化器的参数

# 设置随机种子（保证每次训练一样）
manualSeed = 999
print("Random Seed:", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [os.path.join(image_dir, f)
                            for f in os.listdir(image_dir)
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # 第二个值随便写，因为 DCGAN 不需要 label


# 1. 定义图像预处理流程
transform = transforms.Compose([
    transforms.Resize(image_size),       # 缩放图像为64x64
    transforms.CenterCrop(image_size),   # 中心裁剪保证大小
    transforms.ToTensor(),                   # 转换为Tensor，值归一化到[0,1]
    transforms.Normalize([0.5, 0.5, 0.5],  # 归一化，使每个通道的像素范围变为[-1,1]，与Tanh输出对应
                         [0.5, 0.5, 0.5])
])

# 2. 加载数据集（ImageFolder默认会自动按文件夹分类，但你这里每个文件夹是一个类别）
dataset = CustomImageDataset(dataroot, transform=transform)

# 3. 构造DataLoader，batch大小和线程数由超参数控制
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=2)

print(f"数据集大小：{len(dataset)} 张图片")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入噪声向量大小 nz x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # 输出像素值范围[-1,1]
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            # nn.utils.spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid()  # 输出真假概率
        )

    def forward(self, input):
        return self.main(input).view(-1)

netG = Generator(nz, ngf, nc).to(device)
netG.apply(weights_init)

netD = Discriminator(nc, ndf).to(device)
netD.apply(weights_init)

# 损失函数：二元交叉熵（BCE）
criterion = nn.BCELoss()

# 创建噪声向量（用于生成图像）——固定噪声可视化用
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# 优化器
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
# 判别器和生成器使用不同学习率，让判别器不要太快压制生成器，提升稳定性
# optimizerD = optim.Adam(netD.parameters(), lr=0.0004, betas=(0.5, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.999))

# 训练循环
real_label = 1
fake_label = 0

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # (1) 训练判别器D：最大化 log(D(x)) + log(1 - D(G(z)))
        netD.zero_grad()

        # 训练判别器用真实图片
        real_cpu = data[0].to(device)  # 取一批真实图片，放到设备上（GPU/CPU）
        batch_size = real_cpu.size(0)  # 这批次的大小
        label = torch.full((batch_size,), real_label, device=device)  # 真实标签全1

        output = netD(real_cpu)  # 判别器判别真实图像
        errD_real = criterion(output, label)  # 计算判别器对真实图的损失
        errD_real.backward()  # 反向传播梯度
        D_x = output.mean().item()  # 记录判别器对真实图的平均预测

        # 训练判别器用生成的假图片
        noise = torch.randn(batch_size, nz, 1, 1, device=device)  # 生成噪声
        fake = netG(noise)  # 生成假图
        label.fill_(fake_label)  # 假图标签设为0
        output = netD(fake.detach())  # 判别器判别假图，detach防止生成器梯度更新
        errD_fake = criterion(output, label)  # 判别器假图损失
        errD_fake.backward()  # 反向传播
        D_G_z1 = output.mean().item()  # 判别器对假图的平均输出
        errD = errD_real + errD_fake  # 判别器总损失
        optimizerD.step()  # 优化判别器参数

        ############################
        # (2) 训练生成器G：最大化 log(D(G(z)))，骗过判别器
        ############################
        netG.zero_grad()
        label.fill_(real_label)  # 生成器目标是让假图被判别器认为是真图，标签设1
        output = netD(fake)  # 判别器对假图的判断（这次不detach）
        errG = criterion(output, label)  # 生成器损失
        errG.backward()  # 反向传播生成器梯度
        D_G_z2 = output.mean().item()  # 判别器对生成器图像输出
        optimizerG.step()  # 优化生成器参数

        # 打印训练状态
        print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
              f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
              f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

        # 每隔100批次保存图片，方便观察训练效果
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                              f'{output_folder}/real_samples.jpg',
                              normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                              f'{output_folder}/fake_samples_epoch_{epoch:03d}.jpg',
                              normalize=True)

    # 每个epoch结束后保存模型权重，方便中断后续训练或评估
    torch.save(netG.state_dict(), f'{output_folder}/netG_epoch_{epoch}.pth')
    torch.save(netD.state_dict(), f'{output_folder}/netD_epoch_{epoch}.pth')
