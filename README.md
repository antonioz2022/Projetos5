# 🚌 Dashboard de Mobilidade Urbana - RMR 2016

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antonioz2022/Projetos5/blob/main/projetos5_v3.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://g13-multimodais-rmr.streamlit.app)
[![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker)](https://www.docker.com/)

---

## 📋 Sobre o Projeto

Análise completa da **Pesquisa Origem-Destino 2016 da Região Metropolitana do Recife (RMR)**, com foco em padrões de mobilidade urbana e uso de transporte público.

### 🔗 Links 
- [Fonte dos Dados](http://dados.recife.pe.gov.br/dataset/pesquisa-origem-destino/resource/2452573b-c07c-442e-a2e2-92af3190d8b4)

- [Plano de Análise e Preparação dos Dados](https://docs.google.com/document/d/1O-OcNtFVkVN8_pLJhpScSWqNUMG3TVe4WHvTLtoHWRg/edit?tab=t.0)
- [Colab](https://colab.research.google.com/github/antonioz2022/Projetos5/blob/main/projetos5_v3.ipynb)
- [Dashboard](https://g13-multimodais-rmr.streamlit.app)


### ✨ Destaques:
- 🎯 **Dashboard Interativo** com 10 páginas de análise
- 🤖 **3 Modelos de Machine Learning** para classificação
- 📊 **20+ Visualizações Interativas** (Plotly + Matplotlib)
- 📓 **Notebook Jupyter** com análise exploratória completa
-  **58.644 registros** analisados

---

## 🚀 Como Rodar Localmente

### Opção 1: 🐳 Com Docker (Recomendado)

```bash
cd streamlit_app
docker-compose up -d
```

**Acesse:** http://localhost:8501

**Parar:** `docker-compose down`

### Opção 2: 💻 Sem Docker

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

**Acesse:** http://localhost:8501

---

## 📊 Estrutura do Dashboard

### 10 Páginas de Análise Completa:

#### 🏠 **Visão Geral**
- KPIs principais (total de viagens, uso de integração)
- Distribuição de tipos de trajeto (monomodal vs multimodal)
- Preview do dataset

#### 📊 **Estatísticas Descritivas**
- Distribuição demográfica (idade, sexo, renda)
- Estatísticas de trabalho e estudo
- Uso de terminais de integração

#### 🚇 **Tipo de Trajeto**
- Análise monomodal vs multimodal
- Distribuição por contexto (trabalho, aula, filhos)
- Gráficos de pizza interativos

#### 🚌 **Modal Share**
- **Diferenciação clara:** Analisa TODOS os modais (população geral)
- Top 8 modais mais utilizados
- Comparação entre trabalho, aula e filhos
- Percentuais de participação de cada modal

#### 🗺️ **Análise por Localização**
- Top 10 bairros com mais viagens
- Top 10 municípios
- Análise de terminais de integração mais usados

#### 🔄 **Integração Multimodal**
- **Diferenciação clara:** Foca APENAS em viagens multimodais
- Top combinações de modais (ex: Ônibus + Metrô)
- Análise de integração formal vs informal
- Perfil demográfico de usuários multimodais

#### 👥 **Perfil Demográfico**
- Distribuição por gênero, faixa etária e renda
- Cruzamento de variáveis
- Análise de escolaridade

#### 📈 **Modelos de Regressão**
- Regressão Linear Simples (renda vs num_modais)
- Regressão Linear Múltipla
- Visualização de coeficientes e resíduos

#### 🤖 **Modelos de Classificação**
- **Regressão Logística** (~78% acurácia)
- **Decision Tree** (~80% acurácia)
- **Random Forest** (~80% acurácia)
- Matriz de confusão e métricas comparativas
- Predição de uso de integração formal/terminal

#### 📝 **Conclusões**
- Insights principais da análise
- Recomendações para políticas públicas
- Próximos passos

---

## 🔑 Principais Insights

- � **Ônibus domina:** 43.8% das viagens ao trabalho
- 🚶 **A pé em segundo lugar:** 30.2% das viagens dos filhos à escola
- 🔄 **~29% das viagens** utilizam mais de um modal (multimodal)
- 🤝 **Combinação mais comum:** Ônibus + Metrô
- ⚠️ **Integração formal baixa:** Apenas 15.2% usam terminais
- 🤖 **Random Forest:** Melhor modelo de ML (80% acurácia)

---

## 🛠️ Tecnologias Utilizadas

### Backend & Análise
- **Python 3.11**
- **Pandas** - Manipulação de dados
- **NumPy** - Operações numéricas
- **Scikit-learn** - Machine Learning
- **Docker** - Containerização
- **Docker Compose** - Orquestração

### Visualização
- **Streamlit** - Dashboard interativo
- **Plotly** - Gráficos interativos
- **Matplotlib** - Visualizações estáticas
- **Seaborn** - Gráficos estatísticos

---

## 📁 Estrutura do Projeto

```
Projetos5/
├── streamlit_app/
│   ├── app.py                 # Dashboard Streamlit principal
│   ├── Dockerfile            # Imagem Docker
│   ├── docker-compose.yml    # Orquestração
│   └── requirements.txt      # Dependências Python
├── dados/
│   └── dataset2.csv          # Dataset RMR 2016 (58.644 registros)
├── projetos5_v3.ipynb        # Notebook Colab completo
├── start-docker.ps1          # Script Windows para iniciar
├── stop-docker.ps1           # Script Windows para parar
└── README.md                 # Este arquivo
```

---

## 🔧 Correções e Melhorias Recentes

### ✅ Correção de Bugs
1. **Modelos de Classificação:**
   - Corrigida criação de `num_modais_trabalho` (agora conta modais reais da string)
   - Antes: valores binários (1 ou 2)
   - Depois: valores reais (1 a 6 modais)
   - **Resultado:** Modelos agora produzem predições diferentes e corretas

2. **Modal Share:**
   - Corrigido cálculo de percentuais (texto e gráfico agora consistentes)
   - Ambos usam o mesmo subset (top8) para cálculo

3. **Random Forest:**
   - Melhoria na acurácia e diferenciação dos modelos

### 🎨 Melhorias Visuais
1. **Caixas de Diferenciação:**
   - Fundo amarelo claro (`#fff3cd`)
   - Borda laranja grossa (6px)
   - Sombra para destaque
   - Fonte maior (1.05rem)
   - **Localização:**
     - Modal Share: "Analisa TODOS os modais"
     - Integração Multimodal: "Analisa APENAS viagens multimodais"

2. **Menu de Navegação:**
   - Adicionado emoji 🚌 ao "Modal Share"
   - Todos os itens agora têm emojis consistentes

### 🧹 Limpeza de Código
- Removido código de debug desnecessário
- Removido warnings de predições idênticas
- Código mais limpo e eficiente

---
🎯 **Pronto para começar?** Execute `docker-compose up -d` e explore os dados!
