# Rede Neural

# Processamento de um neurônio
`Y_inj = bj + ∑(xi*wij)`

`yj = f(Y_inj)`

- x = entradas do neurônio j
- wij = pesos nas conexões entre cada entrada i e o neurônio j
- bj = um bias
- Y_inj = entrada total no neurônio j
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





# Feed Forward 

