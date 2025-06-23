import os
import json
import shutil

# caminhos
json_path = '/path/to/test_labels.json'         
imagens_dir = '/path/to/test/images'          
saida_dir = '/path/to/test_separados'  


with open(json_path, 'r') as f:
    dados = json.load(f)


for item in dados:
    filename = item['Filename']
    classe = item['Class']
      
    destino_classe = os.path.join(saida_dir, classe)
    os.makedirs(destino_classe, exist_ok=True)

    origem = os.path.join(imagens_dir, filename)
    destino = os.path.join(destino_classe, filename)

    if os.path.exists(origem):
        shutil.move(origem, destino)
    else:
        print(f"Arquivo não encontrado: {origem}")
