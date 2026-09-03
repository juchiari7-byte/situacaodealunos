# Sistema de Previsão de Situação Escolar

Projeto em Python que utiliza **Machine Learning** para prever a situação de um aluno — **Aprovado, Recuperação ou Reprovado** — com base em três informações:

- Horas de estudo
- Número de faltas
- Nota

A aplicação possui uma interface gráfica desenvolvida com **Gradio**, permitindo que o usuário informe os dados de um aluno e receba uma previsão.

## Tecnologias utilizadas

- Python
- Pandas
- Scikit-learn
- Gradio

## Como funciona

O projeto utiliza dados de alunos para treinar um modelo de classificação. As variáveis utilizadas como entrada são:

| Variável | Descrição |
|---|---|
| `Horas_de_estudo` | Quantidade de horas dedicadas aos estudos |
| `Faltas` | Número de faltas do aluno |
| `Nota` | Nota obtida pelo aluno |

A variável de saída é `Situacao`, que pode assumir três valores:

- **Aprovado**
- **Recuperação**
- **Reprovado**

Depois do treinamento, o usuário pode inserir os dados de um novo aluno pela interface e o modelo realiza a previsão.

## Estrutura do projeto

```text
.
├── app.py
└── README.md
```

- `app.py`: código principal da aplicação, contendo os dados, treinamento do modelo, função de previsão e interface Gradio.
- `README.md`: documentação do projeto.

## Instalação

Certifique-se de ter o Python instalado.

Depois, instale as bibliotecas necessárias:

```bash
pip install pandas scikit-learn gradio
```

## Executando o projeto

No terminal, dentro da pasta do projeto, execute:

```bash
python app.py
```

Após a execução, o Gradio disponibilizará a interface para realizar as previsões.

## Exemplo

Um aluno com:

```text
Horas de estudo: 6
Faltas: 2
Nota: 7.0
```

pode ser enviado ao modelo para obter uma previsão de situação escolar.

## Modelo de Machine Learning

O código original utiliza um **Decision Tree Classifier (Árvore de Decisão)** para realizar a classificação.

A separação dos dados em treinamento e teste é feita utilizando `train_test_split`, e o modelo é treinado com as variáveis de entrada e a situação correspondente.

## Interface

A aplicação utiliza o Gradio para criar uma interface simples na qual o usuário informa:

1. Horas de estudo;
2. Número de faltas;
3. Nota.

Em seguida, o sistema retorna a situação prevista pelo modelo.

## Melhorias futuras

Algumas melhorias que podem ser implementadas:

- Aumentar e diversificar a base de dados de treinamento;
- Balancear as classes de classificação;
- Avaliar o modelo utilizando accuracy, precision, recall e F1-score;
- Comparar diferentes algoritmos de Machine Learning;
- Exibir a probabilidade de cada previsão;
- Adicionar gráficos para facilitar a interpretação dos resultados;
- Criar uma interface mais moderna e responsiva;
- Adicionar validações para os valores inseridos pelo usuário.

## Observação

Os resultados dependem diretamente da qualidade e da quantidade dos dados utilizados no treinamento. Em uma aplicação real, seria necessário utilizar uma base de dados escolar maior, confiável e representativa.

---

**Projeto acadêmico de Machine Learning em Python.**
