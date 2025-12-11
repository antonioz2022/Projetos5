import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, roc_curve, auc, mean_squared_error, r2_score,
                            precision_recall_curve, average_precision_score)

# ==================== CONFIGURAÇÃO DA PÁGINA ====================
st.set_page_config(
    page_title="Dashboard - Mobilidade Urbana RMR",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ESTILO CSS ====================
st.markdown("""
<style>
.main-header {font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; font-weight: bold;}
.sub-header {font-size: 1.5rem; color: #2c3e50; margin-top: 2rem; margin-bottom: 1rem; 
              border-left: 5px solid #1f77b4; padding-left: 10px;}
.insight-box {background-color: #fff3cd; padding: 20px; border-radius: 10px; 
               border-left: 6px solid #ff9800; margin: 20px 0; color: #2c3e50; 
               font-size: 1.05rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

# ==================== DICIONÁRIOS DE MAPEAMENTO ====================
SEXO_MAP = {1: 'Masculino', 2: 'Feminino'}

FAIXA_ETARIA_MAP = {
    1: 'Até 6 anos', 2: '6 a 15 anos', 3: '16 a 24 anos',
    4: '25 a 39 anos', 5: '40 a 59 anos', 6: 'Acima de 60 anos'
}

RENDA_MAP = {
    1: 'Até 1 SM', 2: '1 a 2 SM', 3: '2 a 3 SM', 4: '3 a 5 SM',
    5: '5 a 10 SM', 6: '10 a 20 SM', 7: '+ 20 SM',
    8: 'Sem rendimento', 9: 'Sem declaração'
}

MODAL_MAP = {
    0: "Não declarado", 1: "A pé", 2: "Bicicleta", 3: "Ônibus", 4: "Metrô",
    5: "Carro (dirigindo)", 6: "Carona familiar", 7: "Carona amigo/colega",
    8: "Carro com motorista", 9: "Motocicleta", 10: "Transporte escolar",
    11: "Táxi", 12: "Fretado"
}

APP_TAXI_MAP = {0: "Não declarado", 1: "Nunca", 2: "Às vezes", 3: "Sempre"}

TERMINAL_MAP = {
    0: "Não declarado", 1: "TI Aeroporto", 2: "TI Afogados", 3: "TI Barro", 4: "TI Cabo",
    5: "TI Cajueiro Seco", 6: "TI Camaragibe", 7: "TI Cavaleiro", 8: "TI Caxangá",
    9: "TI Igarassu", 10: "TI Jaboatão", 11: "TI Joana Bezerra", 12: "TI Macaxeira",
    13: "TI PE-15", 14: "TI Pelópidas Silveira", 15: "TI Recife", 16: "TI Tancredo Neves",
    17: "TI TIP", 18: "TI Xambá", 19: "TI Rio Doce"
}

NIVEL_ESTUDO_MAP = {
    0: "Não declarado", 1: "Fundamental", 2: "Médio", 
    3: "Graduação", 4: "Pós-Graduação"
}

# ==================== FUNÇÕES DE CARREGAMENTO E PREPARAÇÃO ====================
@st.cache_data
def load_data():
    """Carrega o dataset com cache"""
    import os
    
    # Lista de caminhos possíveis para tentar
    possible_paths = [
        '../dados/dataset2.csv',  # Caminho relativo local
        'dados/dataset2.csv',     # Caminho relativo do Streamlit Cloud
        '/app/dados/dataset2.csv', # Caminho do Docker
        os.path.join(os.path.dirname(__file__), '../dados/dataset2.csv')  # Caminho absoluto relativo
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path, low_memory=False)
                return df
        except (FileNotFoundError, PermissionError):
            continue
    
    # Se nenhum caminho funcionou, levanta erro
    raise FileNotFoundError(
        f"Arquivo dataset2.csv não encontrado. Tentou os seguintes caminhos: {possible_paths}"
    )

def clean_modal(value):
    """Limpa valores de modais e corrige erros (ex: 2005 → 5)"""
    if pd.isna(value):
        return []
    raw = str(value).replace(" ", "").split(",")
    cleaned = []
    for v in raw:
        if v.isdigit():
            num = int(v)
            if num > 12:
                num = int(str(num)[-1])
            if 0 <= num <= 12:
                cleaned.append(num)
    return cleaned

def classifica_modo(valor):
    """Classifica trajeto em monomodal/multimodal/sem_resposta"""
    if pd.isna(valor):
        return "sem_resposta"
    valor = str(valor).strip()
    if valor == "0" or valor == "":
        return "sem_resposta"
    if "," in valor:
        return "multimodal"
    return "monomodal"

def contar_modais(col):
    """Conta distribuição de modais em uma coluna"""
    lista = []
    for linha in col.dropna():
        linha = str(linha).strip()
        if linha == "" or linha == "0":
            continue
        for m in linha.split(","):
            try:
                codigo = int(m.strip())
                lista.append(MODAL_MAP.get(codigo, "Outro"))
            except ValueError:
                continue
    if not lista:
        return pd.Series(dtype=int)
    return pd.Series(lista).value_counts().sort_values(ascending=False)

@st.cache_data
def prepare_data(df):
    """Prepara e enriquece o dataframe com variáveis derivadas"""
    # Flags trabalho e estudo
    df['trabalha_flag'] = df['trabalha'] == 1
    df['estuda_flag'] = df['pesquisado_estuda'] == 1
    
    # Classificação de trajetos
    df['tipo_trajeto_trabalho'] = df['meio_transporte_trab'].map(classifica_modo)
    df['tipo_trajeto_aula'] = df['transporte_aula'].map(classifica_modo)
    df['tipo_trajeto_filhos'] = df['meios_transporte_filhos'].map(classifica_modo)
    
    # Uso de terminais e integração
    df['usa_terminal_trabalho'] = df['terminal_int_trabalho'].astype(str).str.strip() != '0'
    df['usa_integracao_aula'] = df['utiliza_integracao_aula'] == 1
    
    # Número de modais
    def to_num_modais(tipo):
        if tipo == 'multimodal':
            return 2
        if tipo == 'monomodal':
            return 1
        return 0
    
    df['num_modais_trabalho'] = df['tipo_trajeto_trabalho'].map(to_num_modais)
    df['num_modais_aula'] = df['tipo_trajeto_aula'].map(to_num_modais)
    df['num_modais'] = df[['num_modais_trabalho', 'num_modais_aula']].max(axis=1)
    
    # Variável binária de integração
    df['usa_integracao'] = ((df['usa_terminal_trabalho']) | (df['usa_integracao_aula'])).astype(int)
    
    # Listas de modais
    df['modal_trabalho_list'] = df['meio_transporte_trab'].apply(clean_modal)
    df['modal_aula_list'] = df['transporte_aula'].apply(clean_modal)
    df['modal_filhos_list'] = df['meios_transporte_filhos'].apply(clean_modal)
    
    # Mapeamentos descritivos
    df['sexo_desc'] = df['sexo'].map(SEXO_MAP)
    df['faixa_etaria_desc'] = df['faixa_etaria'].map(FAIXA_ETARIA_MAP)
    df['renda_desc'] = df['renda'].map(RENDA_MAP)
    
    return df

# ==================== FUNÇÕES DE VISUALIZAÇÃO ====================
def plot_modal_share_pie(series, titulo):
    """Gráfico de pizza para distribuição de modais"""
    if series.empty:
        st.info(f"📊 Sem dados suficientes para: {titulo}")
        return
    
    fig = px.pie(
        values=series.values,
        names=series.index,
        title=titulo,
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

# ==================== MAIN ====================
def main():
    st.markdown('<h1 class="main-header">🚌 Dashboard de Mobilidade Urbana - RMR</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>Sobre:</b> Análise exploratória e preditiva da Pesquisa Origem-Destino 2016 (Região Metropolitana do Recife).
    Este dashboard foi feito para a cadeira de Projetos 5 pelo G13 e replica as análises do notebook Colab de forma interativa. 
    </div>
    """, unsafe_allow_html=True)
    
    # Carregar e preparar dados
    with st.spinner("Carregando dataset..."):
        df = load_data()
        df = prepare_data(df)
    
    # Sidebar - Navegação
    st.sidebar.title("📊 Navegação")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Selecione a página:",
        [
            "🏠 Visão Geral",
            "📊 Estatísticas Descritivas",
            "🚇 Tipo de Trajeto",
            "🚌 Modal Share",
            "🗺️ Análise por Localização",
            "🔄 Integração Multimodal",
            "👤 Perfil Usuários Integração",
            "👴🏼 Perfil Demográfico",
            "📉 Modelos de Regressão",
            "〽️ Modelos de Classificação",
            "📝 Conclusões"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.metric("Total de Registros", f"{len(df):,}")
    st.sidebar.metric("Número de Variáveis", len(df.columns))
    
    # Roteamento
    if page == "🏠 Visão Geral":
        show_overview(df)
    elif page == "📊 Estatísticas Descritivas":
        show_descriptive_stats(df)
    elif page == "🚇 Tipo de Trajeto":
        show_trajectory_types(df)
    elif page == "🚌 Modal Share":
        show_modal_share(df)
    elif page == "🗺️ Análise por Localização":
        show_location_analysis(df)
    elif page == "🔄 Integração Multimodal":
        show_multimodal_integration(df)
    elif page == "👤 Perfil Usuários Integração":
        show_integration_user_profile(df)
    elif page == "👴🏼 Perfil Demográfico":
        show_demographic_profile(df)
    elif page == "📉 Modelos de Regressão":
        show_regression_models(df)
    elif page == "〽️ Modelos de Classificação":
        show_classification_models(df)
    else:
        show_conclusions(df)

# ==================== PÁGINAS ====================
def show_overview(df):
    st.markdown('<h2 class="sub-header">🏠 Visão Geral dos Dados</h2>', unsafe_allow_html=True)
    
    # KPIs principais
    col1, col2, col3, col4 = st.columns(4)
    
    pct_trabalham = float(df["trabalha_flag"].mean())
    pct_estudam = float(df["estuda_flag"].mean())
    multimodal = ((df['tipo_trajeto_trabalho'] == 'multimodal') | 
                  (df['tipo_trajeto_aula'] == 'multimodal')).sum()
    avg_modais = df['num_modais'].mean()
    
    with col1:
        st.metric("👥 Trabalham", f"{pct_trabalham*100:.1f}%")
    with col2:
        st.metric("🎓 Estudam", f"{pct_estudam*100:.1f}%")
    with col3:
        st.metric("🔄 Multimodal", f"{multimodal:,}")
    with col4:
        st.metric("📊 Média Modais", f"{avg_modais:.2f}")
    
    st.markdown("---")
    
    # Distribuições básicas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Distribuição por Sexo")
        sexo_dist = df['sexo_desc'].value_counts()
        fig = px.pie(values=sexo_dist.values, names=sexo_dist.index, hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📅 Distribuição por Faixa Etária")
        idade_dist = df['faixa_etaria_desc'].value_counts().sort_index()
        fig = px.bar(x=idade_dist.index, y=idade_dist.values)
        st.plotly_chart(fig, use_container_width=True)

def show_descriptive_stats(df):
    st.markdown('<h2 class="sub-header">📊 Estatísticas Descritivas</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
    <b>📌 Nota:</b> Esta é uma análise geral de todos os respondentes da Pesquisa Origem-Destino 2016 da RMR.
    </div>
    """, unsafe_allow_html=True)
    # Estatísticas de Sexo
    st.markdown("### 1️⃣ Sexo")
    sexo_counts = df['sexo'].value_counts()
    sexo_pct = df['sexo'].value_counts(normalize=True) * 100
    sexo_df = pd.DataFrame({'Quantidade': sexo_counts, 'Percentual (%)': sexo_pct.round(2)})
    sexo_df.index = sexo_df.index.map(SEXO_MAP)
    st.dataframe(sexo_df)
    
    fig = px.bar(x=sexo_df.index, y=sexo_df['Quantidade'], 
                 labels={'x': 'Sexo', 'y': 'Quantidade'},
                 title='Distribuição por Sexo')
    st.plotly_chart(fig, use_container_width=True)
    
    # Faixa Etária
    st.markdown("### 2️⃣ Faixa Etária")
    idade_counts = df['faixa_etaria'].value_counts().sort_index()
    idade_pct = df['faixa_etaria'].value_counts(normalize=True).sort_index() * 100
    idade_df = pd.DataFrame({'Quantidade': idade_counts, 'Percentual (%)': idade_pct.round(2)})
    idade_df.index = idade_df.index.map(FAIXA_ETARIA_MAP)
    st.dataframe(idade_df)
    
    fig = px.bar(x=idade_df.index, y=idade_df['Quantidade'],
                 labels={'x': 'Faixa Etária', 'y': 'Quantidade'},
                 title='Distribuição por Faixa Etária')
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Renda
    st.markdown("### 3️⃣ Renda (Salário Mínimo)")
    renda_counts = df['renda'].value_counts().sort_index()
    renda_pct = df['renda'].value_counts(normalize=True).sort_index() * 100
    renda_df = pd.DataFrame({'Quantidade': renda_counts, 'Percentual (%)': renda_pct.round(2)})
    renda_df.index = renda_df.index.map(RENDA_MAP)
    st.dataframe(renda_df)
    
    fig = px.bar(x=renda_df.index, y=renda_df['Quantidade'],
                 labels={'x': 'Faixa de Renda', 'y': 'Quantidade'},
                 title='Distribuição por Renda')
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top 10 Bairros
    st.markdown("### 4️⃣ Bairros (Top 10)")
    
    top_bairros = df['bairro_residencia'].value_counts().head(10)
    top_bairros_pct = df['bairro_residencia'].value_counts(normalize=True).head(10) * 100
    bairros_df = pd.DataFrame({'Qtd': top_bairros, '%': top_bairros_pct.round(2)})
    
    # Inverter a ordem para mostrar do mais frequente (topo) para o menos frequente (embaixo)
    fig = px.bar(x=top_bairros.values[::-1], y=top_bairros.index[::-1], orientation='h',
                 labels={'x': 'Número de respondentes', 'y': 'Bairro'},
                 title='Top 10 bairros de residência (do mais para o menos frequente)')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(bairros_df)

def show_trajectory_types(df):
    st.markdown('<h2 class="sub-header">🚇 Tipo de Trajeto (Monomodal vs Multimodal)</h2>', 
                unsafe_allow_html=True)
    
    # Métricas sobre multimodalidade
    total_multimodal = ((df['tipo_trajeto_trabalho'] == 'multimodal') | 
                        (df['tipo_trajeto_aula'] == 'multimodal')).sum()
    pct_multimodal = ((df['tipo_trajeto_trabalho'] == 'multimodal') | 
                      (df['tipo_trajeto_aula'] == 'multimodal')).mean()
    total_monomodal = ((df['tipo_trajeto_trabalho'] == 'monomodal') | 
                       (df['tipo_trajeto_aula'] == 'monomodal')).sum()
    max_modais = int(df['num_modais'].max())
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Usuários Multimodais", f"{total_multimodal:,}".replace(',', '.'))
    with c2:
        st.metric("Usuários Monomodais", f"{total_monomodal:,}".replace(',', '.'))
    with c3:
        st.metric("% Multimodais", f"{pct_multimodal*100:.1f}%")
    with c4:
        st.metric("Máx. de Modais Usado", f"{max_modais}")
    
    st.markdown("---")
    
    # Distribuições por contexto
    def percent_series(series):
        ordem = ['monomodal', 'multimodal', 'sem_resposta']
        s = series.value_counts(normalize=True).reindex(ordem).fillna(0) * 100
        return s.round(1)
    
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("#### Trabalho")
        workers = df[df['trabalha_flag'] == True]
        s = percent_series(workers['tipo_trajeto_trabalho'])
        fig = px.bar(x=s.index, y=s.values, labels={'x': 'Tipo', 'y': '%'})
        fig.update_traces(text=[f"{v:.1f}%" for v in s.values], textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(s.rename('Percentual (%)'))
    
    with cols[1]:
        st.markdown("#### Aula")
        students = df[df['estuda_flag'] == True]
        s = percent_series(students['tipo_trajeto_aula'])
        fig = px.bar(x=s.index, y=s.values, labels={'x': 'Tipo', 'y': '%'})
        fig.update_traces(text=[f"{v:.1f}%" for v in s.values], textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(s.rename('Percentual (%)'))
    
    with cols[2]:
        st.markdown("#### Filhos")
        s = percent_series(df['tipo_trajeto_filhos'])
        fig = px.bar(x=s.index, y=s.values, labels={'x': 'Tipo', 'y': '%'})
        fig.update_traces(text=[f"{v:.1f}%" for v in s.values], textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(s.rename('Percentual (%)'))

def show_transport_apps(df):
    st.markdown('<h2 class="sub-header">🚕 Uso de Aplicativos de Transporte</h2>', 
                unsafe_allow_html=True)
    
    cols_apps = {
        "Trabalho": "utiliza_app_taxi_trabalho",
        "Estudo": "utiliza_app_taxi_aula",
        "Filhos": "utiliza_app_taxi_escola"
    }
    
    tabelas = {}
    for categoria, coluna in cols_apps.items():
        tabelas[categoria] = (
            df[coluna]
            .map(APP_TAXI_MAP)
            .value_counts(normalize=True)
            .reindex(["Nunca", "Às vezes", "Sempre", "Não declarado"])
            .dropna()
        )
    
    # Tabela resumo
    st.markdown("### 📊 Intensidade de Uso")
    for categoria, tabela in tabelas.items():
        st.markdown(f"**{categoria}:**")
        for k, v in tabela.items():
            st.write(f"  • {k}: {v:.1%}")
        st.write("")
    
    # Gráfico comparativo
    linhas = []
    for categoria, tabela in tabelas.items():
        for resposta, valor in tabela.items():
            linhas.append([categoria, resposta, valor * 100])
    
    df_plot = pd.DataFrame(linhas, columns=["Categoria", "Resposta", "Proporção"])
    
    fig = px.bar(df_plot, x="Categoria", y="Proporção", color="Resposta",
                 barmode='group', title="Intensidade de Uso dos Apps de Transporte")
    st.plotly_chart(fig, use_container_width=True)

def show_modal_share(df):
    st.markdown('<h2 class="sub-header">🚌 Modal Share</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>📌 Sobre esta análise:</b> Esta seção mostra a distribuição de <b>TODOS</b> os modais 
    utilizados na região (monomodais e multimodais combinados). Cada ocorrência de modal 
    é contada individualmente, incluindo modais que aparecem em viagens multimodais.
    </div>
    """, unsafe_allow_html=True)
    
    modal_trabalho = contar_modais(df['meio_transporte_trab'])
    modal_aula = contar_modais(df['transporte_aula'])
    modal_filhos = contar_modais(df['meios_transporte_filhos'])
    
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("### Trabalho")
        if not modal_trabalho.empty:
            top8 = modal_trabalho.head(8)
            total_top8 = top8.sum()  # Total apenas dos top 8
            for modal, qtd in top8.items():
                pct = qtd / total_top8 * 100  # Porcentagem relativa aos top 8
                st.write(f"• {modal}: {pct:.1f}% ({qtd} registros)")
            plot_modal_share_pie(top8, "Modal Share - Trabalho")
    
    with cols[1]:
        st.markdown("### Aula")
        if not modal_aula.empty:
            top8 = modal_aula.head(8)
            total_top8 = top8.sum()  # Total apenas dos top 8
            for modal, qtd in top8.items():
                pct = qtd / total_top8 * 100  # Porcentagem relativa aos top 8
                st.write(f"• {modal}: {pct:.1f}% ({qtd} registros)")
            plot_modal_share_pie(top8, "Modal Share - Aula")
    
    with cols[2]:
        st.markdown("### Filhos")
        if not modal_filhos.empty:
            top8 = modal_filhos.head(8)
            total_top8 = top8.sum()  # Total apenas dos top 8
            for modal, qtd in top8.items():
                pct = qtd / total_top8 * 100  # Porcentagem relativa aos top 8
                st.write(f"• {modal}: {pct:.1f}% ({qtd} registros)")
            plot_modal_share_pie(top8, "Modal Share - Filhos")

def show_location_analysis(df):
    st.markdown('<h2 class="sub-header">🗺️ Análise por Localização</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("### 🚇 Heatmap: Modal por Bairro (Trabalho)")
    
    # Preparar dados explodidos
    df_exploded = df.copy()
    df_exploded = df_exploded.explode("modal_trabalho_list")
    df_exploded = df_exploded[df_exploded["modal_trabalho_list"].notna()]
    df_exploded["modal_trabalho_list"] = df_exploded["modal_trabalho_list"].astype(int)
    df_exploded["modal_nome"] = df_exploded["modal_trabalho_list"].map(MODAL_MAP)
    
    # Crosstab
    tabela = pd.crosstab(df_exploded["bairro_residencia"], df_exploded["modal_nome"])
    top_bairros = tabela.sum(axis=1).sort_values(ascending=False).head(20).index
    tabela_top = tabela.loc[top_bairros]
    
    # Plotar heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(tabela_top, cmap='YlOrRd', annot=False, fmt='d', ax=ax)
    plt.title("Heatmap – Modal por Bairro (Trabalho)")
    plt.xlabel("Modal")
    plt.ylabel("Bairro")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    <div class="insight-box">
    <b>💡 Interpretação:</b> O heatmap mostra a distribuição de modais por bairro.
    Cores mais intensas indicam maior uso daquele modal no bairro.
    </div>
    """, unsafe_allow_html=True)

def show_multimodal_integration(df):
    st.markdown('<h2 class="sub-header">🔄 Integração Multimodal</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>📌 Sobre esta análise:</b> Esta seção foca especificamente em viagens <b>multimodais</b> 
    (que utilizam mais de um modo de transporte). Diferente das análises anteriores que consideram 
    TODAS as viagens, aqui analisamos apenas as combinações de modais em trajetos integrados.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🚏 Top Combinações de Modais")
    
    all_combinations = []
    
    # Trabalho
    for modals_str in df[df["tipo_trajeto_trabalho"] == "multimodal"]["meio_transporte_trab"].dropna():
        modals = clean_modal(modals_str)
        if len(modals) > 1:
            modal_names = [MODAL_MAP.get(m, "Outro") for m in modals]
            all_combinations.append(frozenset(modal_names))
    
    # Aula
    for modals_str in df[df["tipo_trajeto_aula"] == "multimodal"]["transporte_aula"].dropna():
        modals = clean_modal(modals_str)
        if len(modals) > 1:
            modal_names = [MODAL_MAP.get(m, "Outro") for m in modals]
            all_combinations.append(frozenset(modal_names))
    
    # Filhos
    for modals_str in df[df["tipo_trajeto_filhos"] == "multimodal"]["meios_transporte_filhos"].dropna():
        modals = clean_modal(modals_str)
        if len(modals) > 1:
            modal_names = [MODAL_MAP.get(m, "Outro") for m in modals]
            all_combinations.append(frozenset(modal_names))
    
    combination_counts = Counter(all_combinations)
    formatted_combinations = {
        " + ".join(sorted(list(k))): v for k, v in combination_counts.items()
    }
    
    df_combinations = pd.DataFrame(formatted_combinations.items(), 
                                   columns=['Combinacao', 'Contagem'])
    df_combinations = df_combinations.sort_values(by='Contagem', ascending=False).head(10)
    total = df_combinations['Contagem'].sum()
    df_combinations['Porcentagem'] = (df_combinations['Contagem'] / total) * 100
    
    st.dataframe(df_combinations)
    
    # Ordenar do maior pro menor no gráfico
    fig = px.bar(df_combinations.sort_values('Porcentagem', ascending=True), 
                 x='Porcentagem', y='Combinacao', orientation='h',
                 title='Top 10 Combinações de Modais Multimodais')
    st.plotly_chart(fig, use_container_width=True)

def show_integration_user_profile(df):
    """Análise do perfil dos usuários de integração entre modais"""
    st.markdown('<h2 class="sub-header">🚴‍♂️ Perfil dos Usuários de Integração</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>📌 Sobre esta análise:</b> Esta seção analisa o perfil demográfico dos usuários que 
    utilizam <b>integração formal</b> entre modais (terminais de integração). Diferente da análise 
    de multimodalidade, aqui focamos especificamente em quem declarou usar terminais de integração 
    para fazer suas viagens.
    </div>
    """, unsafe_allow_html=True)
    
    # Filtrar usuários de integração
    df['terminal_int_trabalho'] = df['terminal_int_trabalho'].astype(str).str.strip()
    df['terminal_aula'] = df['terminal_aula'].astype(str).str.strip()
    
    df["usa_int_trabalho"] = (
        (df["utiliza_terminal_int_trabalho"] == 1) | 
        ((df["terminal_int_trabalho"] != "0") & (df["terminal_int_trabalho"] != "nan"))
    )
    df["usa_int_aula"] = (
        (df["utiliza_integracao_aula"] == 1) | 
        ((df["terminal_aula"] != "0") & (df["terminal_aula"] != "nan"))
    )
    df["usuario_integracao"] = df["usa_int_trabalho"] | df["usa_int_aula"]
    
    # Dataset de usuários de integração
    df_int = df[df["usuario_integracao"]].copy()
    df_int = df_int[df_int['sexo'].isin([1, 2])]  # Apenas Masculino/Feminino
    
    # KPIs principais
    st.markdown("### 📊 Indicadores Principais")
    col1, col2, col3 = st.columns(3)
    
    total_usuarios = len(df_int)
    pct_populacao = (total_usuarios / len(df)) * 100
    sexo_predominante = df_int['sexo_desc'].value_counts().index[0]
    
    with col1:
        st.metric("👥 Total de Usuários", f"{total_usuarios:,}")
    with col2:
        st.metric("📈 % da População", f"{pct_populacao:.1f}%")
    with col3:
        st.metric("🎯 Sexo Predominante", sexo_predominante)
    
    st.markdown("---")
    
    # Análise por Sexo
    st.markdown("### 👫 Distribuição por Sexo")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        sexo_counts = df_int['sexo_desc'].value_counts()
        fig = px.bar(x=sexo_counts.index, y=sexo_counts.values,
                     labels={'x': 'Sexo', 'y': 'Número de Usuários'},
                     title='Usuários de Integração por Sexo',
                     color=sexo_counts.index,
                     color_discrete_map={'Masculino': '#1f77b4', 'Feminino': '#ff7f0e'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📋 Estatísticas")
        for sexo, qtd in sexo_counts.items():
            pct = (qtd / total_usuarios) * 100
            st.write(f"**{sexo}:**")
            st.write(f"• {qtd:,} usuários")
            st.write(f"• {pct:.1f}%")
            st.write("")
    
    st.markdown("---")
    
    # Análise por Faixa Etária
    st.markdown("### 📅 Distribuição por Faixa Etária")
    
    ordem_idade = ['Até 6 anos', '6 a 15 anos', '16 a 24 anos', 
                   '25 a 39 anos', '40 a 59 anos', 'Acima de 60 anos']
    
    idade_counts = df_int['faixa_etaria_desc'].value_counts()
    idade_counts = idade_counts.reindex(ordem_idade).dropna()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(x=idade_counts.index, y=idade_counts.values,
                     labels={'x': 'Faixa Etária', 'y': 'Número de Usuários'},
                     title='Usuários de Integração por Faixa Etária',
                     color=idade_counts.values,
                     color_continuous_scale='Viridis')
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📋 Estatísticas")
        for idade, qtd in idade_counts.items():
            pct = (qtd / total_usuarios) * 100
            st.write(f"**{idade}:**")
            st.write(f"• {qtd:,} usuários")
            st.write(f"• {pct:.1f}%")
    
    st.markdown("---")
    
    # Análise por Renda
    st.markdown("### 💰 Distribuição por Faixa de Renda")
    
    ordem_renda = ['Sem rendimento', 'Até 1 SM', '1 a 2 SM', '2 a 3 SM', 
                   '3 a 5 SM', '5 a 10 SM', '10 a 20 SM', '+ 20 SM', 'Sem declaração']
    
    renda_counts = df_int['renda_desc'].value_counts()
    renda_counts = renda_counts.reindex(ordem_renda).dropna()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(x=renda_counts.index, y=renda_counts.values,
                     labels={'x': 'Faixa de Renda', 'y': 'Número de Usuários'},
                     title='Usuários de Integração por Renda',
                     color=renda_counts.values,
                     color_continuous_scale='Magma')
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📋 Estatísticas")
        for renda, qtd in renda_counts.items():
            pct = (qtd / total_usuarios) * 100
            st.write(f"**{renda}:**")
            st.write(f"• {qtd:,} usuários")
            st.write(f"• {pct:.1f}%")
    
    st.markdown("---")
    
    st.markdown("""
    <div class="insight-box">
    <b>💡 Principais Insights:</b><br>
    • A maioria dos usuários de integração está na faixa economicamente ativa (25-59 anos)<br>
    • Predominância de faixas de renda mais baixas (até 3 SM), indicando que o sistema de integração 
      é crucial para a mobilidade de populações de menor poder aquisitivo<br>
    • O uso de terminais de integração é uma prática amplamente adotada na RMR
    </div>
    """, unsafe_allow_html=True)

def show_demographic_profile(df):
    st.markdown('<h2 class="sub-header">👥 Perfil Demográfico</h2>', 
                unsafe_allow_html=True)
    
    # Filtrar dados válidos
    df_exploded = df.copy()
    df_exploded = df_exploded.explode("modal_trabalho_list")
    df_exploded = df_exploded[df_exploded["modal_trabalho_list"].notna()]
    df_exploded["modal_trabalho_list"] = df_exploded["modal_trabalho_list"].astype(int)
    df_exploded["modal_nome"] = df_exploded["modal_trabalho_list"].map(MODAL_MAP)
    
    modais_validos = list(MODAL_MAP.values())[1:]  # Excluir "Não declarado"
    df_analise = df_exploded[df_exploded['modal_nome'].isin(modais_validos)].copy()
    
    # Por Sexo
    st.markdown("### 👫 Modal vs. Sexo")
    dist_sexo = pd.crosstab(
        df_analise['sexo_desc'],
        df_analise['modal_nome'],
        normalize='index'
    ) * 100
    
    st.dataframe(dist_sexo.round(1))
    
    fig, ax = plt.subplots(figsize=(15, 6))
    dist_sexo.plot(kind='bar', stacked=True, colormap='Paired', ax=ax)
    plt.title('Participação Modal por Sexo (Trabalho)')
    plt.ylabel('Percentual (%)')
    plt.xticks(rotation=0)
    plt.legend(title='Modal', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Por Renda
    st.markdown("### 💰 Modal vs. Renda")
    faixas_renda_validas = df_analise[~df_analise['renda_desc'].isin(['Sem rendimento', 'Sem declaração'])].copy()
    
    ordem_renda = ['Até 1 SM', '1 a 2 SM', '2 a 3 SM', '3 a 5 SM', '5 a 10 SM', '10 a 20 SM', '+ 20 SM']
    faixas_renda_validas['renda_ordenada'] = pd.Categorical(
        faixas_renda_validas['renda_desc'],
        categories=ordem_renda,
        ordered=True
    )
    
    dist_renda = pd.crosstab(
        faixas_renda_validas['renda_ordenada'],
        faixas_renda_validas['modal_nome'],
        normalize='index'
    ) * 100
    
    st.dataframe(dist_renda.round(1))
    
    fig, ax = plt.subplots(figsize=(18, 8))
    dist_renda.plot(kind='bar', stacked=True, colormap='Spectral', ax=ax)
    plt.title('Impacto da Renda na Escolha Modal', fontsize=16)
    plt.ylabel('Percentual (%)', fontsize=12)
    plt.xlabel('Faixa de Renda', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Modal', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig)

def show_regression_models(df):
    st.markdown('<h2 class="sub-header">📈 Modelos de Regressão</h2>', 
                unsafe_allow_html=True)
    
    # Preparar dados
    df_reg = df[
        (df['renda'].isin([1,2,3,4,5,6,7])) &
        (df['faixa_etaria'].isin([3,4,5])) &
        (df['num_modais_trabalho'] > 0)
    ].copy()
    
    st.write(f"Total de registros válidos: {len(df_reg)}")
    
    # Regressão 1: Renda vs Número de Modais
    st.markdown("### 📊 Regressão 1: Renda → Número de Modais")
    
    X_renda = df_reg[['renda']].values
    y_modais = df_reg['num_modais_trabalho'].values
    
    modelo = LinearRegression()
    modelo.fit(X_renda, y_modais)
    y_pred = modelo.predict(X_renda)
    
    r2 = r2_score(y_modais, y_pred)
    rmse = np.sqrt(mean_squared_error(y_modais, y_pred))
    
    st.write(f"**Equação:** Nº Modais = {modelo.intercept_:.4f} + {modelo.coef_[0]:.4f} × Renda")
    st.write(f"**R²:** {r2:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")
    
    # Gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_renda, y_modais, alpha=0.3, s=20, color='steelblue')
    x_line = np.array([[1], [2], [3], [4], [5], [6], [7]])
    y_line = modelo.predict(x_line)
    ax.plot(x_line, y_line, color='red', linewidth=3, label=f'R²={r2:.3f}')
    ax.set_xlabel('Faixa de Renda')
    ax.set_ylabel('Número de Modais')
    ax.set_title('Renda vs. Número de Modais')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def show_classification_models(df):
    st.markdown('<h2 class="sub-header">🤖 Modelos de Classificação</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>🎯 Objetivo:</b> Prever se uma pessoa utilizará integração multimodal (terminais ou integração formal)
    com base em suas características demográficas e comportamento de viagem.
    </div>
    """, unsafe_allow_html=True)
    
    # Criar variável target exatamente como no notebook
    df_class = df.copy()
    df_class['usa_integracao'] = (
        (df_class['utiliza_terminal_int_trabalho'] == 1) |
        (df_class['utiliza_integracao_aula'] == 1)
    ).astype(int)
    
    # SEMPRE RECRIA num_modais_trabalho - CONTA O NÚMERO REAL DE MODAIS da coluna meio_transporte_trab
    def contar_num_modais(valor):
        """Conta o número de modais em um registro (ex: '3,5,7' retorna 3)"""
        if pd.isna(valor):
            return 0
        valor = str(valor).replace(" ", "").split(",")
        return len([v for v in valor if v.isdigit() and int(v) > 0])
    
    # FORÇA a recriação usando a coluna correta
    df_class['num_modais_trabalho'] = df_class['meio_transporte_trab'].apply(contar_num_modais)
    
    # Aplicar filtros exatamente como no notebook
    df_class_clean = df_class[
        (df_class['renda'].isin([1,2,3,4,5,6,7])) &
        (df_class['faixa_etaria'].isin([3,4,5])) &
        (df_class['sexo'].isin([1,2])) &
        (df_class['num_modais_trabalho'] > 0)
    ].copy()
    
    # Preparar features e target
    features = ['renda', 'faixa_etaria', 'sexo', 'num_modais_trabalho']
    X = df_class_clean[features].values
    y = df_class_clean['usa_integracao'].values
    
    # Informações sobre os dados
    st.markdown("### 📊 Dados de Treinamento")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Registros", f"{len(X):,}")
    with col2:
        st.metric("Usa Integração", f"{y.sum():,}")
    with col3:
        st.metric("Não Usa", f"{(~y.astype(bool)).sum():,}")
    with col4:
        st.metric("% Positivos", f"{(y.mean()*100):.1f}%")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    st.write(f"**Treino:** {len(X_train):,} registros | **Teste:** {len(X_test):,} registros")
    
    st.markdown("---")
    
    # Treinar modelos
    st.markdown("### 🔧 Treinamento dos Modelos")
    
    models = {
        'Regressão Logística': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
    }
    
    results = {}
    with st.spinner("Treinando modelos..."):
        for name, model in models.items():
            # Treinar modelo
            model.fit(X_train, y_train)
            
            # Fazer predições
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Calcular métricas
            results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'cm': confusion_matrix(y_test, y_pred)
            }
    
    st.success("✅ Modelos treinados com sucesso!")
    
    st.markdown("---")
    
    # Tabela de métricas
    st.markdown("### 📈 Comparação de Métricas")
    
    metrics_df = pd.DataFrame({
        'Modelo': list(results.keys()),
        'Acurácia': [f"{results[m]['accuracy']:.4f}" for m in results],
        'Precisão': [f"{results[m]['precision']:.4f}" for m in results],
        'Recall': [f"{results[m]['recall']:.4f}" for m in results],
        'F1-Score': [f"{results[m]['f1']:.4f}" for m in results]
    })
    
    st.dataframe(metrics_df, use_container_width=True)
    
    # Gráfico comparativo de métricas
    metrics_plot = pd.DataFrame({
        'Modelo': list(results.keys()),
        'Acurácia': [results[m]['accuracy'] for m in results],
        'Precisão': [results[m]['precision'] for m in results],
        'Recall': [results[m]['recall'] for m in results],
        'F1-Score': [results[m]['f1'] for m in results]
    })
    
    fig = px.bar(metrics_plot.melt(id_vars='Modelo', var_name='Métrica', value_name='Score'),
                 x='Modelo', y='Score', color='Métrica', barmode='group',
                 title='Comparação de Desempenho dos Modelos')
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Matrizes de Confusão
    st.markdown("### 🎯 Matrizes de Confusão")
    
    cols = st.columns(3)
    for idx, (name, result) in enumerate(results.items()):
        with cols[idx]:
            cm = result['cm']
            
            # Criar figura com matplotlib para melhor controle
            fig_cm, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Não Usa', 'Usa'],
                       yticklabels=['Não Usa', 'Usa'])
            ax.set_ylabel('Real')
            ax.set_xlabel('Predito')
            ax.set_title(f'{name}')
            plt.tight_layout()
            st.pyplot(fig_cm)
            
            # Explicação da matriz
            tn, fp, fn, tp = cm.ravel()
            st.write(f"**VP:** {tp} | **FP:** {fp}")
            st.write(f"**FN:** {fn} | **VN:** {tn}")
    
    st.markdown("---")
    
    # Curva ROC
    st.markdown("### 📉 Curva ROC (Receiver Operating Characteristic)")
    
    fig_roc = go.Figure()
    
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_proba'])
        roc_auc = auc(fpr, tpr)
        
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'{name} (AUC = {roc_auc:.3f})',
            mode='lines'
        ))
    
    # Linha diagonal (classificador aleatório)
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        showlegend=False,
        name='Baseline (AUC = 0.5)'
    ))
    
    fig_roc.update_layout(
        title='Curva ROC - Comparação de Modelos',
        xaxis_title='Taxa de Falsos Positivos (FPR)',
        yaxis_title='Taxa de Verdadeiros Positivos (TPR)',
        height=500
    )
    
    st.plotly_chart(fig_roc, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>💡 Interpretação da Curva ROC:</b><br>
    • A curva ROC mostra o trade-off entre TPR (sensibilidade) e FPR<br>
    • Quanto mais próxima do canto superior esquerdo, melhor o modelo<br>
    • AUC (Area Under Curve) resume o desempenho: 1.0 = perfeito, 0.5 = aleatório<br>
    • Útil quando queremos avaliar o modelo em diferentes thresholds
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Curva Precision-Recall
    st.markdown("### 📊 Curva Precision-Recall")
    
    fig_pr = go.Figure()
    
    for name, result in results.items():
        precision, recall, _ = precision_recall_curve(result['y_test'], result['y_proba'])
        ap_score = average_precision_score(result['y_test'], result['y_proba'])
        
        fig_pr.add_trace(go.Scatter(
            x=recall, y=precision,
            name=f'{name} (AP = {ap_score:.3f})',
            mode='lines'
        ))
    
    fig_pr.update_layout(
        title='Curva Precision-Recall - Comparação de Modelos',
        xaxis_title='Recall',
        yaxis_title='Precisão',
        height=500
    )
    
    st.plotly_chart(fig_pr, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>💡 Interpretação da Curva Precision-Recall:</b><br>
    • Mostra o trade-off entre precisão e recall<br>
    • Mais útil que ROC quando classes estão desbalanceadas<br>
    • AP (Average Precision) resume o desempenho: quanto maior, melhor<br>
    • Alta precisão significa poucos falsos positivos<br>
    • Alto recall significa poucos falsos negativos
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Análise Textual
    st.markdown("### 📝 Análise dos Resultados")
    
    # Encontrar melhor modelo
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
    best_result = results[best_model_name]
    
    st.markdown(f"""
    #### 🏆 Melhor Modelo: **{best_model_name}**
    
    **Métricas de Desempenho:**
    - **Acurácia:** {best_result['accuracy']:.2%} - Percentual total de predições corretas
    - **Precisão:** {best_result['precision']:.2%} - Dos casos preditos como "Usa Integração", quantos realmente usam
    - **Recall:** {best_result['recall']:.2%} - Dos casos que realmente usam integração, quantos conseguimos identificar
    - **F1-Score:** {best_result['f1']:.4f} - Média harmônica entre Precisão e Recall
    
    #### 💼 Implicações Práticas:
    
    **Para Políticas Públicas:**
    - Modelos podem identificar perfis com alta probabilidade de usar integração multimodal
    - Útil para planejar investimentos em terminais de integração
    - Ajuda a priorizar áreas e perfis demográficos para campanhas de incentivo
    
    **Limitações Identificadas:**
    - Acurácia moderada (~{best_result['accuracy']:.0%}) indica que há outros fatores importantes não capturados
    - Variáveis como localização geográfica detalhada e tipo de trabalho poderiam melhorar o modelo
    - Desbalanceamento de classes pode afetar o desempenho
    
    **Próximos Passos:**
    - Incluir variáveis geográficas (distância ao terminal mais próximo)
    - Adicionar features temporais (horário de trabalho/estudo)
    - Testar modelos mais complexos (XGBoost, Neural Networks)
    - Realizar validação cruzada para resultados mais robustos
    """)
    
    # Comparação entre modelos
    st.markdown("#### 🔍 Comparação entre Modelos")
    
    # Contar quantos modelos empatados há para cada métrica
    def count_ties(metric_name):
        max_val = max(r[metric_name] for r in results.values())
        return sum(1 for r in results.values() if r[metric_name] == max_val)
    
    for name, result in results.items():
        with st.expander(f"📊 Análise: {name}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Pontos Fortes:**")
                
                # Só mostrar "melhor" se não houver empate em todas as métricas
                acc_ties = count_ties('accuracy')
                prec_ties = count_ties('precision')
                rec_ties = count_ties('recall')
                f1_ties = count_ties('f1')
                
                if result['accuracy'] == max(r['accuracy'] for r in results.values()):
                    if acc_ties > 1:
                        st.write(f"🟡 Acurácia igual aos melhores ({acc_ties} modelos empatados)")
                    else:
                        st.write("✅ Melhor acurácia geral")
                        
                if result['precision'] == max(r['precision'] for r in results.values()):
                    if prec_ties > 1:
                        st.write(f"🟡 Precisão igual aos melhores ({prec_ties} modelos empatados)")
                    else:
                        st.write("✅ Melhor precisão (menos falsos positivos)")
                        
                if result['recall'] == max(r['recall'] for r in results.values()):
                    if rec_ties > 1:
                        st.write(f"🟡 Recall igual aos melhores ({rec_ties} modelos empatados)")
                    else:
                        st.write("✅ Melhor recall (menos falsos negativos)")
                        
                if result['f1'] == max(r['f1'] for r in results.values()):
                    if f1_ties > 1:
                        st.write(f"🟡 F1-Score igual aos melhores ({f1_ties} modelos empatados)")
                    else:
                        st.write("✅ Melhor F1-Score (balanço geral)")
                
                # Se não tem nenhum ponto forte, avisar
                if not any([
                    result['accuracy'] == max(r['accuracy'] for r in results.values()),
                    result['precision'] == max(r['precision'] for r in results.values()),
                    result['recall'] == max(r['recall'] for r in results.values()),
                    result['f1'] == max(r['f1'] for r in results.values())
                ]):
                    st.write("ℹ️ Este modelo não atingiu o melhor resultado em nenhuma métrica")
            
            with col2:
                st.write("**Características:**")
                if name == 'Regressão Logística':
                    st.write("• Modelo linear simples e interpretável")
                    st.write("• Rápido para treinar e fazer predições")
                    st.write("• Assume relação linear entre features")
                elif name == 'Decision Tree':
                    st.write("• Modelo baseado em regras (if-then)")
                    st.write("• Fácil de visualizar e entender")
                    st.write("• Pode capturar relações não-lineares")
                else:  # Random Forest
                    st.write("• Ensemble de múltiplas árvores")
                    st.write("• Mais robusto que árvore única")
                    st.write("• Menor risco de overfitting")

def show_conclusions(df):
    st.markdown('<h2 class="sub-header">📝 Conclusões e Insights</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Principais Descobertas
    
    #### 🚌 Padrões de Mobilidade
    - **Ônibus é o modal dominante** em toda a RMR, especialmente em trajetos ao trabalho
    - **Integração multimodal é comum** mas uso de terminais formais ainda é baixo
    - **Variação regional significativa**: bairros periféricos dependem mais de transporte público
    
    #### 💰 Relação com Renda
    - **Ponto de inflexão em 5 SM**: acima dessa faixa, carro se torna modal dominante
    - **Modais ativos (a pé)** mais presentes em faixas de baixa renda
    - **Renda explica ~10-15% da variação** no número de modais utilizados
    
    #### 👥 Perfil Demográfico
    - **Mulheres** dependem mais de ônibus e carona familiar
    - **Homens** usam mais motocicleta e carro dirigindo
    - **Jovens (16-24)** têm maior uso de modais ativos e metrô
    
    #### 🗺️ Segregação Espacial
    - **Bairros periféricos** mostram dependência extrema do ônibus (>70%)
    - **Bairros nobres** (Casa Forte, Boa Viagem) têm domínio do carro (>50%)
    - **Fluxo metropolitano** concentrado em poucos corredores de transporte
    
    ### 📊 Sobre as Visualizações
    
    **Por que cada gráfico?**
    - **Barras empilhadas**: comparar composição de modais entre grupos
    - **Heatmap**: identificar padrões espaciais e concentrações
    - **Pizza**: mostrar proporções de um todo (quando poucas categorias)
    - **Sankey**: visualizar fluxos entre origem e destino
    - **Scatter + linha**: avaliar correlações e tendências lineares
    
    ### 🚧 Limitações e Próximos Passos
    - Modelos preditivos capturam apenas parte da complexidade
    - Variáveis geográficas detalhadas poderiam melhorar predições
    - Análise temporal revelaria tendências de mudança de comportamento
    - Explorar modelos não-lineares (XGBoost, Neural Networks)
    """)

# ==================== EXECUTAR ====================
if __name__ == "__main__":
    main()
