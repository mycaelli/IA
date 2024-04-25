# Rede Neural

# Processamento de um neurônio
`Y_inj = bj + ∑(xi*wij)`

`yj = f(y_inj)`

- x = entradas do neurônio j
- wij = pesos nas conexões entre cada entrada i e o neurônio j
- bj = um bias
- y_inj = entrada total no neurônio j
- f(Y_inj) = função de ativação
- yj = saída no neurônio j


# Como alterar os pesos e o bias
```
if y != t
  wi(new) = wi(old) + αtxi
  b(new) = b(old) + at
else
  wi(new) = wi(old) 
  b(new) = b(old)
```

- y = saída do neurônio
- t = target 
- wi(new) = novo peso
- wi(old) = peso antigo
- a = taxa de aprendizado
- xi = entrada (todos os pesos do neurônio que errou devem passar pela alteracão de peso)
- b(new) = novo bias
- b(old) = bias antigo


# Condição de parada do treinamento 
Se nenhum peso mudou na época, pare; senão continue.


## Backpropagation
`ej(n) = dj(n) - yj(n)`

# Feed Forward 

