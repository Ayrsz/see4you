# 👁️ See4You

O **See4You** é um projeto de *Image Captioning* (Legendagem Automática de Imagens) desenvolvido com o propósito central de **assistir pessoas com deficiência visual**. O sistema processa as imagens do ambiente e descreve o cenário em linguagem natural, promovendo maior autonomia e inclusão digital.

---

## ⚙️ Arquitetura e Performance

Para garantir que o projeto possa ser executado em dispositivos com recursos limitados (como smartphones ou sistemas embarcados de assistência), a eficiência computacional foi a prioridade máxima.

O modelo final utiliza a seguinte arquitetura:
* **Encoder (Visão):** **MobileNetV3** — Rede convolucional pré-treinada, responsável por extrair a representação vetorial da imagem.
* **Decoder (Linguagem):** **GRU** (Gated Recurrent Unit) — Rede recorrente responsável pela geração de texto.

### Por que esta escolha?

Realizamos testes rigorosos comparando diferentes redes recorrentes e redes convolucionais pré-treinadas. A combinação **MobileNetV3 + GRU** obteve métricas próximas às das outras arquiteturas, porém com uma redução significativa no tempo de execução

| Comparativo de Arquitetura | Ganho de Velocidade |
| :--- | :--- |
| **vs. MobileNetV3 + LSTM** | ⚡ **2.0x mais rápida** |
| **vs. ResNet50 + GRU** | ⚡⚡ **2.5x mais rápida** |

Isso significa menos latência entre a captura da imagem e a descrição auditiva para o usuário, algo crítico para aplicações de acessibilidade.

---
## 🛠️ Instalação e Execução

O projeto foi estruturado para ser reprodutível e simples de configurar. Siga os passos abaixo para preparar o ambiente e treinar o modelo.

### 1. Clonar e Instalar Dependências

Clone este repositório e instale as bibliotecas necessárias:

```bash
git clone [https://github.com/seu-usuario/see4you.git](https://github.com/seu-usuario/see4you.git)
cd see4you
pip install -r requirements.txt
```
### 📥 2. Preparação dos Dados

Antes de iniciar o treinamento, é necessário configurar o ambiente e baixar os dados necessários. Execute o notebook **`setup.ipynb`** para realizar este processo.

**O que este notebook faz:**
* **Dataset:** Baixa e descompacta o dataset de imagens e legendas.
* **Embeddings:** Realiza o download dos embeddings pré-treinados **FastText**.
* **Estrutura:** Cria automaticamente as pastas `/data` e `/embeddings` no diretório raiz do projeto.

### 📊 3. Treinamento e Avaliação

Com os dados organizados, execute o notebook **`training.ipynb`** para iniciar o pipeline de Deep Learning.

**O fluxo de execução inclui:**
1.  **Pré-processamento:** Carregamento dos DataLoaders e tokenização.
2.  **Modelagem:** Instanciação da arquitetura **MobileNetV3 + GRU**.
3.  **Treino:** Execução das épocas de treinamento com monitoramento da *Loss*.
4.  **Teste:** Avaliação automática utilizando métricas de similaridade no conjunto de teste.
