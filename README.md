## Projeto PID-GAN

Este repositório contém a implementação de métodos de **Generative Adversarial Networks (GANs)** para geração de imagens palinológicas sintéticas e super-resolução de imagens degradadas.

### Instalação

1. **Clone o repositório**

   ```bash
   git clone https://github.com/thcastilho/pid-gan.git
   ```
2. **Entre na pasta do projeto**

   ```bash
   cd pid-gan
   ```
3. **Crie e ative um ambiente virtual**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate     # Windows
   ```
4. **Instale as dependências**

   ```bash
   pip install -r requirements.txt
   ```

---

### Download do Dataset

Baixe os arquivos do dataset de grãos de pólen (Pollengrain Dataset) neste link:

[https://iplab.dmi.unict.it/pollengraindataset/dataset](https://iplab.dmi.unict.it/pollengraindataset/dataset)

---

### Pré-processamento de Dados

Após baixar o dataset, execute o script `split.py` para separar o conjunto de teste em subpastas organizadas por classe:

```bash
python split.py
```

Isso criará uma estrutura de pastas onde cada classe de grão de pólen terá sua própria pasta contendo as imagens de teste correspondentes.

---

### Geração de dados sintéticos com DCGAN

Antes de rodar a DCGAN, instale o *pytorch-fid* ([https://github.com/mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid)) para cálculo do FID:

   ```bash
   pip install pytorch-fid
   ```

Em seguida, para cada classe de grão de pólen, edite o parâmetro `label` em `DCGAN.py` e execute:

```bash
python DCGAN.py
```

---

### Super-Resolução com SRGAN

1. **Clone o repositório SRGAN-PyTorch**

   ```bash
   git clone https://github.com/Lornatang/SRGAN-PyTorch.git
   cd SRGAN-PyTorch
   ```

2. **Baixe o modelo pré-treinado**

   Siga as instruções do repositório para obter o checkpoint `SRGAN_x4-SRGAN_ImageNet`.

3. **Prepare as imagens degradadas**

   No projeto principal, para cada classe de grão de pólen, edite o parâmetro `label` em `degrade.py` e gere as imagens em baixa resolução:

   ```bash
   python degrade.py
   ```

4. **Integre e execute o script de super-resolução**

   * Copie ou mova o script `apply_srgan.py` para dentro da pasta `SRGAN-PyTorch`.
   * Execute o script para aplicar a SRGAN sobre as imagens degradadas:

     ```bash
     python apply_srgan.py --input_dir path/para/degraded --output_dir path/para/srgan_output
     ```

---

### Cálculo de Similaridade Estrutural (SSIM)

Por fim, volte à raiz do projeto e execute o script `compute_ssim.py` para calcular a similaridade estrutural entre as imagens geradas SRGAN e as originais:

```bash
python compute_ssim.py --dir1 path/para/imagens_originais --dir2 path/para/srgan_output
```

O script utiliza os seguintes parâmetros:

* `--dir1`: diretório contendo as imagens originais.
* `--dir2`: diretório contendo as imagens geradas pela SRGAN.

Isso produzirá um relatório ou métricas que indicam a qualidade das imagens processadas em relação às originais. Você pode redirecionar a saída para um arquivo de texto para salvar resultados individuais, por exemplo:
```bash
python compute_ssim.py --dir1 path/para/imagens_originais --dir2 path/para/imagens_processadas >> output.txt
```
Isso gerará (ou acrescentará) os resultados em output.txt.

---


## Contato

- Eduardo Roldão Nonato Perondini: `eduardo.perondini@unesp.br`
- Rafael Fabri Chimidt: `rafael.fabri@unesp.br`
- Thiago César Castilho Almeida: `tc.almeida@unesp.br`  
