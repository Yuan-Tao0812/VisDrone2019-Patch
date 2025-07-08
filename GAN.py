import os
import torch
import torchvision.utils as vutils
import os
from DCGAN import Generator  # 如果你是写在main.py里，就直接复制Generator类进来
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from cleanfid import fid

# 参数设置（根据训练时保持一致）
nz = 100           # 噪声维度（latent vector size）
ngf = 64           # 生成器的 feature map 基数
nc = 3             # 图像通道数（RGB=3）
n_generate = 1000  # 要生成的图像总数量
batch_size = 64    # 每批生成多少张图
image_size = 64    # 图像尺寸（通常为64×64）
ngpu = 1           # GPU 数量（单卡）
real_images_path = "/content/drive/MyDrive/VisDrone2019-Patch/awning-tricycle"
output_path = "/content/drive/MyDrive/VisDrone2019-Generate/awning-tricycle"
model_path = '/content/drive/MyDrive/VisDrone2019-Generate/Generator/awning-tricycle/netG_epoch_24.pth'
os.makedirs(output_path, exist_ok=True)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator(nz=nz, ngf=ngf, nc=nc).to(device)
netG.load_state_dict(torch.load(model_path, map_location=device))  # 你保存的路径
netG.eval()

# ==== 图像生成并保存 ====
transform = transforms.ToPILImage()
idx = 0
with torch.no_grad():
    for _ in tqdm(range((n_generate + batch_size - 1) // batch_size), desc="Generating"):
        current_batch = min(batch_size, n_generate - idx)
        noise = torch.randn(current_batch, nz, 1, 1, device=device)
        fake_images = netG(noise).cpu()

        for i in range(current_batch):
            img = transform((fake_images[i] + 1) / 2)  # [-1,1] → [0,1]
            img.save(os.path.join(output_path, f"{idx + i:05d}.jpg"))
        idx += current_batch

print(f"\n✅ 生成完成，图像保存在：{output_path}")

# ==== 计算 FID 分数 ====
fid_score = fid.compute_fid(real_images_path, output_path, mode="clean", dataset_name=None)
print(f"\n🎯 FID 得分（越低越好）: {fid_score:.4f}")