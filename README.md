# 🚌 Dashboard de Mobilidade Urbana - RMR 2016

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antonioz2022/Projetos5/blob/main/projetos5_v3.ipynb)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://localhost:8501)
[![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker)](https://www.docker.com/)

> **TL;DR**: Execute `docker-compose up -d` na pasta `streamlit_app/` e acesse http://localhost:8501

---

## 📋 Sobre o Projeto

Análise completa da **Pesquisa Origem-Destino 2016 da Região Metropolitana do Recife (RMR)**, com foco em padrões de mobilidade urbana e uso de transporte público.

### ✨ Destaques:
- 🎯 **Dashboard Interativo** com 10 páginas de análise
- 🤖 **3 Modelos de Machine Learning** para classificação
- 📊 **20+ Visualizações Interativas** (Plotly + Matplotlib)
- 📓 **Notebook Jupyter** com análise exploratória completa
- 🐳 **Docker** para deploy facilitado
- 📈 **58.644 registros** analisados

---

## 🚀 Execução Rápida

### 🐳 Docker (Recomendado)

```bash
cd streamlit_app
docker-compose up -d
```

**Acesse:** http://localhost:8501

**Parar:** `docker-compose down`

---

### 💻 Sem Docker

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

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

### 📍 Modal Share
- **Ônibus domina:** 43.8% das viagens ao trabalho
- **A pé em segundo lugar:** 30.2% das viagens dos filhos à escola
- **Metrô:** Modal importante na região metropolitana

### 🔄 Multimodalidade
- **~29% das viagens** utilizam mais de um modal
- **Combinação mais comum:** Ônibus + Metrô
- **Integração formal baixa:** Apenas 15.2% usam terminais

### 👥 Perfil Demográfico
- **Mulheres:** Ligeiramente maioria nas viagens
- **Faixa etária ativa:** 25-59 anos predomina
- **Renda:** 71% ganham até 2 salários mínimos

### 🤖 Machine Learning
- **Random Forest:** Melhor modelo (80% acurácia)
- **Features importantes:** Renda e número de modais
- **Aplicação prática:** Previsão de demanda por integração

---

## 🛠️ Tecnologias Utilizadas

### Backend & Análise
- **Python 3.11**
- **Pandas** - Manipulação de dados
- **NumPy** - Operações numéricas
- **Scikit-learn** - Machine Learning

### Visualização
- **Streamlit** - Dashboard interativo
- **Plotly** - Gráficos interativos
- **Matplotlib** - Visualizações estáticas
- **Seaborn** - Gráficos estatísticos

### Deploy
- **Docker** - Containerização
- **Docker Compose** - Orquestração

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
   - Ajustado `max_depth` de 5 para 10 (igual ao Colab)
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

## 📖 Documentação Adicional

### Dataset
- **Fonte:** Pesquisa Origem-Destino RMR 2016
- **Registros:** 58.644 viagens
- **Variáveis:** 51 colunas
- **Contextos:** Trabalho, Aula, Filhos

### Notebook Colab
O arquivo `projetos5_v3.ipynb` contém:
- Análise exploratória completa
- Tratamento de dados
- Visualizações detalhadas
- Modelos de ML com validação
- Todas as análises que inspiraram o dashboard

**Acesse:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antonioz2022/Projetos5/blob/main/projetos5_v3.ipynb)

---

## 🚀 Deploy

### Streamlit Cloud (Gratuito)
1. Push para GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte o repositório
4. Defina o arquivo principal: `streamlit_app/app.py`
5. Deploy automático ✨

### Docker em Cloud
- **AWS:** ECS/Fargate
- **Azure:** Container Instances
- **Google Cloud:** Cloud Run
- **DigitalOcean:** App Platform
- **Heroku:** Container Registry

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar bugs
- Sugerir melhorias
- Adicionar novas análises
- Melhorar visualizações

---

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos.

---

## 👨‍💻 Autor

**Projeto 5 - Análise de Mobilidade Urbana**  
Desenvolvido como parte de estudos em Ciência de Dados

---

## 💡 Dicas de Uso

1. **Wide Mode:** Use o modo "Wide" do Streamlit para melhor visualização
2. **Filtros:** Explore os filtros disponíveis em cada página
3. **Gráficos Interativos:** Passe o mouse sobre os gráficos para mais detalhes
4. **Notebook:** Consulte o Colab notebook para análises mais profundas
5. **Performance:** O Docker garante ambiente isolado e consistente

---

🎯 **Pronto para começar?** Execute `docker-compose up -d` e explore os dados!
