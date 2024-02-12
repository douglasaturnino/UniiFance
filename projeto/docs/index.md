Este é um projeto fictício. A empresa, o contexto e as perguntas de negócios não são reais. 
Este portfólio está seguindo as recomendações do blog da Comunidade DS.

O Banco UniFinance, uma instituição financeira líder e confiável, destaca-se no
mercado pela sua dedicação em fornecer soluções de crédito acessíveis e sob
medida para empresários do setor comercial. Com foco em empréstimos
flexíveis e acessíveis, nossa equipe altamente qualificada trabalha em estreita
colaboração com os clientes para atender às suas necessidades financeiras
específicas.

Atualmente o banco está passando por uma revisão em como ele empresta o
dinheiro para os seus clientes, assim o objetivo é criar processos inteligentes
para a previsão de que alguém pode vim a passar por dificuldades financeiras
nos próximos dois anos.


## Entendimento do negócio

No Banco UniFinance, quando um cliente solicita um empréstimo, iniciamos um
processo de avaliação que inclui a análise de diversos fatores, um dos quais é a
possível ocorrência de dificuldades financeiras nos próximos dois anos. Isso é
crucial para identificar e mitigar riscos que poderiam levar à inadimplência, o
que, por sua vez, afetaria negativamente o banco. Nossa prioridade é garantir
empréstimos responsáveis e sustentáveis, tanto para o benefício do cliente
quanto para a segurança financeira da instituição.

## Premissas de negócio
- Todos os produtos de dados entregues devem ser acessíveis via internet.

As variáveis do dataset original são:

| Nome da variável                              | Descrição
| ----------------------------------------------|---------------------------------------------------------------------------------------|
| target                                        | Pessoa sofreu inadimplência de 90 dias                                                |
| TaxaDeUtilizacaoDeLinhasNaoGarantidas         | Saldo total em cartões de crédito e linhas de crédito pessoais, exceto imóveis e sem  |
| Idade                                         | Idade do cliente em anos                                                              |
| NumeroDeVezes30-59DiasAtrasoNaoPior           | Número de vezes que o mutuário apresentou atraso de 30 a 59 dias                      |
| TaxaDeEndividamento                           | Pagamentos mensais de dívidas, pensão alimentícia, custo de vida dividido pela renda  |
| RendaMensal                                   | Renda mensal                                                                          |
| NumeroDeLinhasDeCreditoEEmprestimosAbertos    | Número de empréstimos abertos (parcelamento, como empréstimo de carro ou hipoteca)    |
| NumeroDeVezes90DiasAtraso                     | Quantas vezes o mutuário esteve atrasado por 90 dias ou mais                          |
| NumeroDeEmprestimosOuLinhasImobiliarias       | Número de empréstimos hipotecários e imobiliários, incluindo linhas de crédito        |
| NumeroDeVezes60-89DiasAtrasoNaoPior           | Número de vezes que o mutuário apresentou atraso de 60 a 89 dias                      |
| NumeroDeDependentes                           | Número de dependentes na família excluindo eles próprios (cônjuge, filhos etc.)       |

## Planejamento da solução
### Produto final
O que será entregue efetivamente?

- Uma api onde será necessário enviar os dados para realizar a previsão.
- Um relatorio com o resultado do monitoramento feito pelo Evidently AI"

### Ferramentas
Quais ferramentas serão usadas no processo?

- Visual Studio code;
- Jupyter Notebook;
- Git, Github;
- Python;
- Mlflow;
- Docker.
- Mkdock

## Resultados para o negócio
Foi desenvolvida uma API onde os clientes podem enviar os dados necessários para realizar a predição de possíveis dificuldades financeiras nos próximos dois anos.

## Conclusão

* O objetivo do projeto foi alcançado, dado que os produtos de dados propostos foram gerados com sucesso.

<p>Confira o <a href="model_monitoring_report.html" target="_blank" rel="noopener noreferrer">Resultado do monitoramento</a>.</p>