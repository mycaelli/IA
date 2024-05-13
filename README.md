# Rede Neural

## Feedfoward
  - cada unidade escondida somas suas entradas ponderadas e aplica a função de ativação para computar sua saída, a enviando para a próxima camada

  ```
    aux_camada_escondida = bias_camada_escondida + ∑ entrada_camada_escondida x peso_camada_escondida

    saida_camada_escondida = sigmoide(aux_camada_escondida)
  ```

  - cada unidade de saída somas suas entradas ponderadas e aplica a função de ativação para calcular sua saída

  ```
    aux_saida = bias_saida + ∑ saida_camada_escondida x peso_saida

    output = sigmoide(aux_saida)
  ```

## Backpropagation

  - cada unidade de saída considera sua saída e a saída esperada para o dado de entrada para computador o erro
  - então calcula a correção dos pesos e bias e envia o termo de correção de erro para a camada anterior
  ```
  errok = (target - saídak) x derivada_sigmoide(saidak)

  correcao_pesok = learning_rate x errok x entrada_neuroniok

  correcao_biask = learning_rate x errok
  ``````

## Atualizacao de pesos
  - cada unidade de saida e escondida altera seus pesos e bias

  ```
    novo_pesok = peso_antigk + correcao_pesok

    novo_bias = bias_antigo + correcao_biask
  ```