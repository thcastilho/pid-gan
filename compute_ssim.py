import os
import argparse
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread

def calcular_ssim_entre_imagens(dir1, dir2):
    imagens_dir1 = set(os.listdir(dir1))
    imagens_dir2 = set(os.listdir(dir2))
    nomes_comuns = imagens_dir1.intersection(imagens_dir2)

    avg_ssim = 0

    for nome in sorted(nomes_comuns):
        caminho1 = os.path.join(dir1, nome)
        caminho2 = os.path.join(dir2, nome)

        try:
            img1 = imread(caminho1)
            img2 = imread(caminho2)

            valor_ssim = ssim(img1, img2, channel_axis=-1)
            print(f"{nome}: SSIM = {valor_ssim:.4f}")
            avg_ssim += valor_ssim

        except Exception as e:
            print(f"Erro ao processar '{nome}': {e}")

    print(len(nomes_comuns))

    print(f'Média: {avg_ssim / len(nomes_comuns)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1")
    parser.add_argument("--dir2")
    args = parser.parse_args()

    calcular_ssim_entre_imagens(args.dir1, args.dir2)
