```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================

st.set_page_config(
    page_title="Sistema de Previsão Acadêmica",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# 1. GERAÇÃO DOS DADOS SINTÉTICOS
# ==========================================

@st.cache_data
def gerar_dados_escolares(n_amostras=1000, seed=42):

    np.random.seed(seed)

    # Variáveis independentes
    horas_estudo = np.random.uniform(0, 20, n_amostras)
    faltas = np.random.randint(0, 35, n_amostras)

    # Nota base
    nota_base = (
        4.0
        + (horas_estudo * 0.25)
        - (faltas * 0.12)
        + np.random.normal(0, 1.0, n_amostras)
    )

    nota = np.clip(nota_base, 0.0, 10.0)

    situacoes = []

    for h, f, n in zip(horas_estudo, faltas, nota):

        if f > 25 or n < 4.0:
            situacoes.append("Reprovado")

        elif n >= 7.0 and f <= 15:
            situacoes.append("Aprovado")

        elif 4.0 <= n < 7.0 and f <= 25:
            situacoes.append("Recuperação")

        else:
            if h >= 8 and n >= 5.5:
                situacoes.append("Recuperação")
            else:
                situacoes.append("Reprovado")

    df = pd.DataFrame({
        "Horas_de_estudo": np.round(horas_estudo, 1),
        "Faltas": faltas,
        "Nota": np.round(nota, 1),
        "Situacao": situacoes
    })

    return df


df_alunos = gerar_dados_escolares()


# ==========================================
# 2. TREINAMENTO DO MODELO
# ==========================================

X = df_alunos[
    ["Horas_de_estudo", "Faltas", "Nota"]
]

y = df_alunos["Situacao"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


@st.cache_resource
def treinar_modelo(X_train, y_train):

    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    return modelo


modelo_rf = treinar_modelo(X_train, y_train)


# ==========================================
# 3. AVALIAÇÃO DO MODELO
# ==========================================

y_pred = modelo_rf.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)


# ==========================================
# 4. FUNÇÃO DE PREVISÃO
# ==========================================

def prever_desempenho(horas, faltas, nota):

    df_entrada = pd.DataFrame(
        [[horas, faltas, nota]],
        columns=[
            "Horas_de_estudo",
            "Faltas",
            "Nota"
        ]
    )

    # Previsão
    predicao = modelo_rf.predict(df_entrada)[0]

    # Probabilidades
    probabilidades = modelo_rf.predict_proba(
        df_entrada
    )[0]

    classes = modelo_rf.classes_

    prob_dict = {
        classe: prob
        for classe, prob in zip(
            classes,
            probabilidades
        )
    }

    confianca = prob_dict[predicao] * 100

    # ======================================
    # EXPLICAÇÃO
    # ======================================

    explica = []

    if nota >= 7.0:

        explica.append(
            "Nota satisfatória acima da média de aprovação (7,0)."
        )

    elif nota < 4.0:

        explica.append(
            "Nota abaixo do mínimo necessário para recuperação (4,0)."
        )

    else:

        explica.append(
            "Nota na faixa intermediária de recuperação (4,0 a 6,9)."
        )

    if faltas > 25:

        explica.append(
            "Número de faltas crítico (superior a 25), "
            "representando alto risco de reprovação por frequência."
        )

    elif faltas <= 10:

        explica.append(
            "Boa frequência escolar."
        )

    if horas >= 8:

        explica.append(
            "Carga horária de estudos semanal adequada."
        )

    else:

        explica.append(
            "Poucas horas de dedicação semanal aos estudos."
        )

    explicacao = " ".join(explica)

    # ======================================
    # GRÁFICO
    # ======================================

    cores = []

    for classe in prob_dict.keys():

        if classe == "Aprovado":
            cores.append("#2e7d32")

        elif classe == "Recuperação":
            cores.append("#f57f17")

        else:
            cores.append("#c62828")

    fig = go.Figure(
        data=[
            go.Bar(
                x=list(prob_dict.keys()),
                y=[
                    p * 100
                    for p in prob_dict.values()
                ],
                marker_color=cores
            )
        ]
    )

    fig.update_layout(
        title="Distribuição de Probabilidade por Classe",
        yaxis=dict(
            title="Probabilidade (%)",
            range=[0, 100]
        ),
        xaxis=dict(
            title="Situação"
        ),
        height=350,
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        )
    )

    return (
        predicao,
        confianca,
        explicacao,
        fig,
        prob_dict
    )


# ==========================================
# 5. INTERFACE STREAMLIT
# ==========================================

st.title(
    "🎓 Sistema Inteligente de Previsão de Desempenho Escolar"
)

st.write(
    "Insira os dados do aluno abaixo para analisar "
    "a probabilidade de **Aprovação**, **Recuperação** "
    "ou **Reprovação**."
)


# ==========================================
# INFORMAÇÕES DO MODELO
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "📚 Amostras utilizadas",
        len(df_alunos)
    )

with col2:
    st.metric(
        "🤖 Acurácia do modelo",
        f"{acuracia:.2%}"
    )

with col3:
    st.metric(
        "🌲 Algoritmo",
        "Random Forest"
    )


st.divider()


# ==========================================
# ENTRADAS
# ==========================================

col_esquerda, col_direita = st.columns(
    2
)


with col_esquerda:

    st.subheader("📋 Dados do Aluno")

    horas = st.slider(
        "⏱️ Horas de Estudo por Semana",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5
    )

    faltas = st.slider(
        "📅 Número de Faltas no Semestre",
        min_value=0,
        max_value=40,
        value=5,
        step=1
    )

    nota = st.slider(
        "📝 Nota Média",
        min_value=0.0,
        max_value=10.0,
        value=6.5,
        step=0.1
    )

    calcular = st.button(
        "🔮 Calcular Previsão",
        type="primary",
        use_container_width=True
    )


# ==========================================
# RESULTADO
# ==========================================

with col_direita:

    st.subheader("📊 Resultado da Análise")

    if calcular:

        if (
            horas < 0
            or faltas < 0
            or not (0 <= nota <= 10)
        ):

            st.error(
                "Por favor, insira valores válidos."
            )

        else:

            (
                predicao,
                confianca,
                explicacao,
                fig,
                prob_dict
            ) = prever_desempenho(
                horas,
                faltas,
                nota
            )


            # ==================================
            # RESULTADO VISUAL
            # ==================================

            if predicao == "Aprovado":

                st.success(
                    f"✅ APROVADO\n\n"
                    f"Confiança: {confianca:.1f}%"
                )

            elif predicao == "Recuperação":

                st.warning(
                    f"⚠️ RECUPERAÇÃO\n\n"
                    f"Confiança: {confianca:.1f}%"
                )

            else:

                st.error(
                    f"❌ REPROVADO\n\n"
                    f"Confiança: {confianca:.1f}%"
                )


            # ==================================
            # EXPLICAÇÃO
            # ==================================

            st.info(
                f"💡 **Análise:** {explicacao}"
            )


            # ==================================
            # PROBABILIDADES
            # ==================================

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ==========================================
# 6. ANÁLISE DO DATASET
# ==========================================

with st.expander(
    "📈 Ver análise do dataset"
):

    st.subheader(
        "Distribuição das Classes"
    )

    distribuicao = (
        df_alunos["Situacao"]
        .value_counts()
    )

    st.dataframe(
        distribuicao,
        use_container_width=True
    )


    st.subheader(
        "Estatísticas Descritivas"
    )

    st.dataframe(
        df_alunos.describe(),
        use_container_width=True
    )


# ==========================================
# 7. RELATÓRIO DO MODELO
# ==========================================

with st.expander(
    "🤖 Ver desempenho do modelo"
):

    st.write(
        f"**Acurácia Geral:** {acuracia:.2%}"
    )

    relatorio = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    relatorio_df = pd.DataFrame(
        relatorio
    ).transpose()

    st.dataframe(
        relatorio_df,
        use_container_width=True
    )


# ==========================================
# RODAPÉ
# ==========================================

st.divider()

st.caption(
    "🎓 Sistema de Previsão de Desempenho Escolar "
    "• Modelo Random Forest • Dados sintéticos"
)
```

### Como executar

Salve o código como **`app.py`** e, no terminal, instale as dependências:

```bash
pip install streamlit pandas numpy plotly scikit-learn
```

Depois execute:

```bash
streamlit run app.py
```

O Streamlit abrirá automaticamente a aplicação no navegador.

**Uma diferença importante:** no Gradio você tinha um botão que disparava `prever_desempenho()`. No Streamlit, a interface é construída de forma reativa, então usei `st.button()` para manter exatamente esse comportamento.

Também aproveitei a conversão para adicionar **métricas de desempenho do modelo**, uma seção expansível com o dataset e outra com o relatório de classificação.
