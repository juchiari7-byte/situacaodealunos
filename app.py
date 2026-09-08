import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. GERAÇÃO E ANÁLISE DOS DADOS SINTÉTICOS
# ==========================================
def gerar_dados_escolares(n_amostras=1000, seed=42):
    np.random.seed(seed)
    
    # Gerando variáveis independentes dentro de limites realistas
    horas_estudo = np.random.uniform(0, 20, n_amostras)
    faltas = np.random.randint(0, 35, n_amostras)
    
    # A nota base correlaciona com horas de estudo e faltas
    nota_base = 4.0 + (horas_estudo * 0.25) - (faltas * 0.12) + np.random.normal(0, 1.0, n_amostras)
    nota = np.clip(nota_base, 0.0, 10.0)
    
    situacoes = []
    for h, f, n in zip(horas_estudo, faltas, nota):
        # Regras de negócio escolares plausíveis
        if f > 25 or n < 4.0:
            situacoes.append('Reprovado')
        elif n >= 7.0 and f <= 15:
            situacoes.append('Aprovado')
        elif 4.0 <= n < 7.0 and f <= 25:
            situacoes.append('Recuperação')
        else:
            # Casos limítrofes ponderados por horas de estudo
            if h >= 8 and n >= 5.5:
                situacoes.append('Recuperação')
            else:
                situacoes.append('Reprovado')
                
    df = pd.DataFrame({
        'Horas_de_estudo': np.round(horas_estudo, 1),
        'Faltas': faltas,
        'Nota': np.round(nota, 1),
        'Situacao': situacoes
    })
    return df

df_alunos = gerar_dados_escolares()

# Análise Básica dos Dados no Console
print("=== Análise do Dataset Gerado ===")
print(f"Total de amostras: {len(df_alunos)}")
print("\nDistribuição das Classes:")
print(df_alunos['Situacao'].value_counts())
print("\nEstatísticas Descritivas:")
print(df_alunos.describe())
print("-" * 40)

# ==========================================
# 2. TREINAMENTO E AVALIAÇÃO DO MODELO
# ==========================================
X = df_alunos[['Horas_de_estudo', 'Faltas', 'Nota']]
y = df_alunos['Situacao']

# Divisão dos dados com amostragem estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Treinamento do Modelo (Random Forest)
modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
modelo_rf.fit(X_train, y_train)

# Avaliação
y_pred = modelo_rf.predict(X_test)
print("\n=== Desempenho do Modelo (Random Forest) ===")
print(f"Acurácia Geral: {accuracy_score(y_test, y_pred):.2%}\n")
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))
print("-" * 40)

# ==========================================
# 3. LÓGICA DA PREVISÃO E INTERFACE GRADIO
# ==========================================
def prever_desempenho(horas, faltas, nota):
    # Validações manuais complementares
    if horas < 0 or faltas < 0 or not (0 <= nota <= 10):
        alerta_html = "<div style='color: red; font-weight: bold;'>Por favor, insira valores válidos.</div>"
        return alerta_html, go.Figure(), "Valores fora dos limites permitidos."

    df_entrada = pd.DataFrame([[horas, faltas, nota]], columns=['Horas_de_estudo', 'Faltas', 'Nota'])
    
    # Previsão da classe e probabilidades
    predicao = modelo_rf.predict(df_entrada)[0]
    probabilidades = modelo_rf.predict_proba(df_entrada)[0]
    classes = modelo_rf.classes_
    
    # Dicionário de probabilidades organizadas
    prob_dict = {classe: prob for classe, prob in zip(classes, probabilidades)}
    confianca = prob_dict[predicao] * 100

    # Estilização visual conforme o resultado
    estilos = {
        'Aprovado': {'cor': '#2e7d32', 'bg': '#e8f5e9', 'icone': '✅'},
        'Recuperação': {'cor': '#f57f17', 'bg': '#fffde7', 'icone': '⚠️'},
        'Reprovado': {'cor': '#c62828', 'bg': '#ffebee', 'icone': '❌'}
    }
    
    estilo = estilos.get(predicao, {'cor': '#333', 'bg': '#fff', 'icone': ''})

    # Bloco HTML destacado para a resposta
    resultado_html = f"""
    <div style="background-color: {estilo['bg']}; border: 2px solid {estilo['cor']}; border-radius: 10px; padding: 20px; text-align: center;">
        <h2 style="color: {estilo['cor']}; margin: 0;">{estilo['icone']} {predicao.upper()}</h2>
        <p style="font-size: 16px; color: #424242; margin-top: 10px;">
            Probabilidade de Confiança: <b>{confianca:.1f}%</b>
        </p>
    </div>
    """

    # Geração de texto explicativo
    explica = []
    if nota >= 7.0:
        explica.append("Nota satisfatória acima da média de aprovação (7.0).")
    elif nota < 4.0:
        explica.append("Nota abaixo do mínimo necessário para recuperação (4.0).")
    else:
        explica.append("Nota na faixa intermediária de recuperação (4.0 a 6.9).")

    if faltas > 25:
        explica.append("Número de faltas crítico (superior a 25), risco alto de reprovação por frequência.")
    elif faltas <= 10:
        explica.append("Boa frequência escolar.")

    if horas >= 8:
        explica.append("Carga horária de estudos semanal adequada.")
    else:
        explica.append("Poucas horas dedicação semanal aos estudos.")

    explicacao_texto = " ".join(explica)

    # Gráfico de barras das probabilidades com Plotly
    fig = go.Figure(data=[
        go.Bar(
            x=list(prob_dict.keys()),
            y=[p * 100 for p in prob_dict.values()],
            marker_color=['#2e7d32' if c == 'Aprovado' else '#f57f17' if c == 'Recuperação' else '#c62828' for c in prob_dict.keys()]
        )
    ])
    fig.update_layout(
        title="Distribuição de Probabilidade por Classe (%)",
        yaxis=dict(title="Probabilidade (%)", range=[0, 100]),
        xaxis=dict(title="Situação"),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return resultado_html, fig, explicacao_texto

# ==========================================
# 4. CONSTRUÇÃO DA INTERFACE VISUAL (GRADIO)
# ==========================================
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate"
)

with gr.Blocks(theme=theme, title="Sistema de Previsão Acadêmica") as demo:
    gr.Markdown(
        """
        # 🎓 Sistema Inteligente de Previsão de Desempenho Escolar
        Insira os dados do aluno abaixo para analisar a probabilidade de **Aprovação**, **Recuperação** ou **Reprovação**.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Dados do Aluno")
            input_horas = gr.Slider(
                minimum=0, maximum=20, step=0.5, value=5.0, 
                label="Horas de Estudo por Semana", info="Valores entre 0 e 20 horas."
            )
            input_faltas = gr.Slider(
                minimum=0, maximum=40, step=1, value=5, 
                label="Número de Faltas no Semestre", info="Valores entre 0 e 40 faltas."
            )
            input_nota = gr.Slider(
                minimum=0.0, maximum=10.0, step=0.1, value=6.5, 
                label="Nota Média", info="Valores entre 0.0 e 10.0."
            )
            btn_prever = gr.Button("🔮 Calcular Previsão", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Resultado da Análise")
            output_resultado = gr.HTML(label="Resultado da Previsão")
            output_explicacao = gr.Textbox(label="Análise Explicativa", interactive=False)
            output_grafico = gr.Plot(label="Probabilidades")

    btn_prever.click(
        fn=prever_desempenho,
        inputs=[input_horas, input_faltas, input_nota],
        outputs=[output_resultado, output_grafico, output_explicacao]
    )

# Executar a aplicação
if __name__ == "__main__":
    demo.launch()
