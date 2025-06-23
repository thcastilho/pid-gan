import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import subprocess

# hiperparâmetros
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
image_size = 64    
nc = 3            
z_dim = 100       
ngf = 64          
df = 64           
num_epochs = 300   
lr = 0.0002
beta1 = 0.5       

# diretórios
label = 4
print(f'Label: {label}')
train_dir = f'/path/to/train/{label}/train_OBJ'
val_dir = f'/path/to/test_separados/{label}'
os.makedirs(f"./output_{label}/images/grids", exist_ok=True)
os.makedirs(f"./output_{label}/images/samples", exist_ok=True)

# transformação nas imagens
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*nc, [0.5]*nc)
])

# carrega as imagens
dataset_train = datasets.ImageFolder(root=train_dir, transform=transform)
dataset_val = datasets.ImageFolder(root=val_dir, transform=transform)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=2)

# aplica distribuição normal aos pesos das camadas convolucionais e batch-norm, prática padrão em DCGAN para ajudar na estabilidade do treinamento
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# gerador: recebe um vetor de ruído (z_dim) e, via camadas de convolução, “aprende” a produzir uma imagem RGB
generator = nn.Sequential(
    nn.ConvTranspose2d(z_dim, ngf*8, 4, 1, 0, bias=False),
    nn.BatchNorm2d(ngf*8), nn.ReLU(True),
    nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf*4), nn.ReLU(True),
    nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf*2), nn.ReLU(True),
    nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
    nn.BatchNorm2d(ngf), nn.ReLU(True),
    nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
    nn.Tanh() # função de ativação: é o "cérebro" do neurônio que decide quanto daquela informação vai ser passada adiante pra próxima camada
).to(device)

# discriminador: recebe uma imagem (real ou gerada) e emite um escalar entre 0 e 1, indicando se acha que é “real” (perto de 1) ou “falsa” (perto de 0)
discriminator = nn.Sequential(
    nn.Conv2d(nc, df, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(df, df*2, 4, 2, 1, bias=False), nn.BatchNorm2d(df*2), nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(df*2, df*4, 4, 2, 1, bias=False), nn.BatchNorm2d(df*4), nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(df*4, df*8, 4, 2, 1, bias=False), nn.BatchNorm2d(df*8), nn.LeakyReLU(0.2, inplace=True),
    nn.Conv2d(df*8, 1, 4, 1, 0, bias=False), nn.Sigmoid() # produz vetor de probabilidade
).to(device)

# garante que as redes comecem com uma distribuição normal nos parâmetros
generator.apply(weights_init)
discriminator.apply(weights_init)

# binary cross entropy: compara a probabilidade dada pela sigmoide do discriminador e retorna um valor que quantifica o erro
criterion = nn.BCELoss()

# Adam é um método de otimização, atualizam os pesos depois da perda
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

best_fid = float('inf')

print("Starting Training...")
for epoch in range(1, num_epochs+1):
    for i, (real_images, _) in enumerate(train_loader, 1):
        b_size = real_images.size(0)
        real_images = real_images.to(device)

        real_labels = torch.full((b_size,), 1., device=device)
        fake_labels = torch.full((b_size,), 0., device=device)

        # passa as imagens reais pelo discriminador
        discriminator.zero_grad()
        output_real = discriminator(real_images).view(-1)
        lossD_real = criterion(output_real, real_labels)

        # gera imagens falsas
        noise = torch.randn(b_size, z_dim, 1, 1, device=device)
        fake_images = generator(noise)

        # discriminador determina se as imagens são fake ou não
        output_fake = discriminator(fake_images.detach()).view(-1)
        lossD_fake = criterion(output_fake, fake_labels)

        # discriminador melhora quando consegue separar real de fake (diminui lossD)
        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # gerador melhora quando consegue criar imagens que o discriminador classifica como real (diminui lossG)
        generator.zero_grad()
        output = discriminator(fake_images).view(-1)
        lossG = criterion(output, real_labels)
        lossG.backward()
        optimizerG.step()

    if epoch == 1 or epoch % 10 == 0:
        with torch.no_grad():
            fake_grid = generator(fixed_noise).cpu()
            utils.save_image(fake_grid, f"./output_{label}/images/grids/fake_epoch_{epoch}.png", normalize=True)

        with torch.no_grad():
            sample_noise = torch.randn(len(dataset_val), z_dim, 1, 1, device=device)
            fake_samples = generator(sample_noise).cpu()
            sample_dir = f"./output_{label}/images/samples/epoch_{epoch}"
            os.makedirs(sample_dir, exist_ok=True)
            for i, img in enumerate(fake_samples):
                utils.save_image(img, f"{sample_dir}/fake_{i:04d}.png", normalize=True)

        cmd = ['pytorch-fid', sample_dir, f'{val_dir}/img', '--device', 'cuda']
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            fid_value = float(result.stdout.strip().split()[-1])
        except:
            fid_value = None

        print(f"Epoch {epoch} - FID: {fid_value}")

        if fid_value is not None and fid_value < best_fid:  
            best_fid = fid_value    
            torch.save(generator.state_dict(), f"./output_{label}/best_generator.pth")  

torch.save(generator.state_dict(), f"./output_{label}/final_generator.pth")
torch.save(discriminator.state_dict(), f"./output_{label}/final_discriminator.pth")
print("Treinamento encerrado. Best FID:", best_fid)