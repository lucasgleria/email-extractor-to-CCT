# Guia Completo: Treinamento e Melhoria do Modelo de Extração

Este guia explica **como funciona** o sistema de extração, **como treinar** um modelo customizado e **estratégias avançadas** para torná-lo cada vez mais inteligente e preciso.

## 📚 Índice

1. [Como Funciona o Sistema Atual](#como-funciona-o-sistema-atual)
2. [Coletando Dados de Qualidade](#coletando-dados-de-qualidade)
3. [Treinando Seu Primeiro Modelo](#treinando-seu-primeiro-modelo)
4. [Entendendo o Pipeline de Extração](#entendendo-o-pipeline-de-extração)
5. [Estratégias para Melhorar o Modelo](#estratégias-para-melhorar-o-modelo)
6. [Troubleshooting e Dicas Avançadas](#troubleshooting-e-dicas-avançadas)

---

## Como Funciona o Sistema Atual

O sistema usa uma **abordagem híbrida** que combina três estratégias:

### 1. **Heurísticas Baseadas em Regras** (Sempre Ativas)
- **Label-based**: Procura por palavras-chave (aliases) como "MAWB", "REFERENCIA", "CONSIGNEE"
- **Pattern-based**: Usa expressões regulares para identificar padrões (ex: 11 dígitos = MAWB, 3 letras = IATA)
- **Proximity boost**: Quando encontra um label, busca valores próximos nas linhas seguintes

**Pontuação**: Cada método tem um peso:
- Label match: **0.4** (mais confiável)
- Pattern match: **0.3** (médio)
- Proximity: **0.2** (menor, mas útil)

### 2. **Modelo de Machine Learning** (Opcional, mas Recomendado)
- Usa **TF-IDF** (Term Frequency-Inverse Document Frequency) para extrair características do texto
- **Regressão Logística** para classificar se uma linha contém o valor correto
- Treinado com **seus dados corrigidos** para aprender padrões específicos dos seus e-mails

### 3. **Sistema de Pontuação Combinado**
O sistema **combina** as pontuações de heurísticas + modelo e escolhe o candidato com maior score.

```
Score Final = Score Heurísticas + Score Modelo
Melhor Candidato = Maior Score Final (após validação)
```

---

## Coletando Dados de Qualidade

### Por que Dados de Qualidade São Críticos?

O modelo aprende **exatamente** com os dados que você fornece. Se você corrigir erros consistentemente, o modelo aprenderá esses padrões.

### Estratégia de Coleta

#### Fase 1: Coleta Inicial (50-100 exemplos)
1. **Processe e-mails variados**: diferentes remetentes, layouts, formatos
2. **Corrija TODOS os erros**: mesmo pequenos, para ensinar o modelo
3. **Mantenha consistência**: se "REF" e "REFERENCIA" são a mesma coisa, sempre use o mesmo formato

#### Fase 2: Casos Especiais (50-100 exemplos adicionais)
1. **E-mails difíceis**: dark mode, baixa qualidade, layouts não padronizados
2. **Casos limite**: múltiplos MAWBs, valores ausentes, formatos incomuns
3. **Erros comuns**: identifique padrões de erro e adicione mais exemplos desses casos

#### Fase 3: Refinamento Contínuo (iterativo)
- Após cada treinamento, teste o modelo
- Identifique novos erros
- Adicione exemplos desses erros e retreine

### Dicas de Ouro para Dados de Qualidade

✅ **FAÇA:**
- Corrija valores mesmo quando parecem corretos mas estão em formato diferente
- Inclua exemplos onde campos estão **ausentes** (deixe vazio, não invente)
- Mantenha formato consistente (ex: sempre "GRU" não "São Paulo (GRU)")

❌ **NÃO FAÇA:**
- Deixar valores incorretos "porque está quase certo"
- Misturar formatos (às vezes "REF123", outras "123-REF")
- Incluir dados de teste ou fictícios

---

## Treinando Seu Primeiro Modelo

### Pré-requisitos

- Pelo menos **50-100 exemplos** corrigidos e salvos
- Acesso ao Google Colab (gratuito)
- Arquivo `training_data.json` exportado

### Passo a Passo Detalhado

#### Passo 1: Exportar Dados de Treinamento

1. Abra `index.html` no navegador
2. Clique em **"Export Training Data"**
3. Salve o arquivo `training_data.json` em local seguro

**Verificação**: Abra o JSON e confirme que contém:
- `raw_text`: texto completo do OCR
- `fields`: objeto com os campos corrigidos
- `timestamp`: data de criação

#### Passo 2: Configurar Google Colab

1. Acesse [Google Colab](https://colab.research.google.com/)
2. Clique em **"New notebook"**
3. Renomeie o notebook (ex: "Email Extractor Training")

#### Passo 3: Upload dos Dados

1. No painel esquerdo, clique no ícone **"Files"** 📁
2. Clique no ícone de **upload** (⬆️)
3. Selecione `training_data.json`
4. Aguarde o upload completar

#### Passo 4: Script de Treinamento Básico

Cole o script abaixo em uma célula e execute (`Shift+Enter`):

## Step 1: Export Your Training Data

1.  Open the `index.html` file in your browser.
2.  Process at least 50-100 images, ensuring you correct any mistakes made by the initial rule-based extractor. The more high-quality data you provide, the better your model will be.
3.  Click the **Export Training Data** button.
4.  Save the downloaded `training_data.json` file to your computer.

## Step 2: Set Up Your Free Training Environment (Google Colab)

1.  Go to [Google Colab](https://colab.research.google.com/).
2.  Click **New notebook**.
3.  You now have a ready-to-use Python environment in the cloud.

## Step 3: Upload Your Data to Colab

1.  In the left-hand panel of your Colab notebook, click the **Files** icon.
2.  Click the **Upload to session storage** icon and select the `training_data.json` file you downloaded in Step 1.

```python
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re

# --- 1. Load the Data ---
with open('training_data.json', 'r') as f:
    data = json.load(f)

print(f"✅ Carregados {len(data)} exemplos de treinamento.")

# --- 2. Preprocess and Featurize ---
def featurize(raw_text):
    """Divide o texto em linhas para análise"""
    return raw_text.split('\n')

# Campos que serão treinados
fields_to_train = ['REFERENCIA', 'MAWB', 'HAWB', 'DESTINO', 'DESTINO_FINAL', 'CONSIGNEE']
models = {}

for field in fields_to_train:
    print(f"\n{'='*50}")
    print(f"📊 Treinando modelo para: {field}")
    print(f"{'='*50}")

    # Criar dados de treinamento
    X_train = []
    y_train = []

    for item in data:
        lines = featurize(item['raw_text'])
        correct_value = item['fields'].get(field, '').strip()

        # Pular se não houver valor correto
        if not correct_value:
            continue

        # Para cada linha, criar um exemplo
        for line in lines:
            X_train.append(line)
            # Se o valor correto está nesta linha, é positivo (1), senão negativo (0)
            if correct_value.upper() in line.upper():
                y_train.append(1)
            else:
                y_train.append(0)

    # Verificar se há exemplos positivos
    positive_count = sum(y_train)
    total_count = len(y_train)
    
    if positive_count == 0:
        print(f"⚠️  Pulando {field}: nenhum exemplo positivo encontrado.")
        continue

    print(f"📈 Exemplos: {total_count} total, {positive_count} positivos ({100*positive_count/total_count:.1f}%)")

    # Dividir em treino e teste (opcional, para avaliação)
    if len(X_train) > 20:
        X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    else:
        X_tr, y_tr = X_train, y_train
        X_te, y_te = [], []

    # --- 3. Train the Model ---
    # TF-IDF: converte texto em vetores numéricos
    # Logistic Regression: classifica se a linha contém o valor
    model = make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 2),  # Palavras individuais e pares
            max_features=5000,   # Limite de features (reduz tamanho do modelo)
            min_df=2            # Ignorar palavras que aparecem menos de 2 vezes
        ),
        LogisticRegression(
            class_weight='balanced',  # Balanceia classes desiguais
            max_iter=1000
        )
    )
    
    model.fit(X_tr, y_tr)
    models[field] = model
    
    # Avaliar modelo (se houver dados de teste)
    if len(X_te) > 0:
        y_pred = model.predict(X_te)
        accuracy = sum(y_pred == y_te) / len(y_te)
        print(f"✅ Acurácia no teste: {accuracy*100:.1f}%")
    
    print(f"✅ Modelo para {field} treinado com sucesso!")

# --- 4. Export the Model ---
print(f"\n{'='*50}")
print("💾 Exportando modelo...")
print(f"{'='*50}")

exported_model = {}
for field, model in models.items():
    vectorizer = model.named_steps['tfidfvectorizer']
    classifier = model.named_steps['logisticregression']

    exported_model[field] = {
        'vocabulary': vectorizer.vocabulary_,
        'idf': list(vectorizer.idf_),
        'coef': list(classifier.coef_[0]),
        'intercept': list(classifier.intercept_)
    }

with open('model.json', 'w') as f:
    json.dump(exported_model, f)

print(f"✅ Modelo exportado para model.json")
print(f"📦 Tamanho do arquivo: {len(json.dumps(exported_model)) / 1024:.1f} KB")
```

### Explicação do Script

#### O que cada parte faz:

1. **TF-IDF Vectorizer**: 
   - Converte cada linha de texto em um vetor numérico
   - Pesos palavras importantes (que aparecem em poucos documentos) mais alto
   - `ngram_range=(1,2)`: considera palavras individuais E pares de palavras

2. **Logistic Regression**:
   - Aprende a classificar: "esta linha contém o valor que procuro?"
   - `class_weight='balanced'`: ajusta para dados desbalanceados (muitas linhas negativas, poucas positivas)

3. **Exportação**:
   - Salva apenas o necessário: vocabulário, pesos (coef) e intercept
   - Tamanho pequeno para rodar no navegador

#### Passo 5: Download e Instalação

1. No Colab, clique nos **três pontos** ao lado de `model.json`
2. Selecione **"Download"**
3. Salve na **mesma pasta** do `index.html`
4. Recarregue a página do `index.html`

O modelo será carregado automaticamente! 🎉

---

## Entendendo o Pipeline de Extração

### Como o Sistema Usa o Modelo

Quando você processa uma imagem, o sistema:

1. **Executa OCR** → obtém texto bruto
2. **Divide em linhas** → cada linha é um candidato
3. **Aplica heurísticas** → encontra candidatos por regras
4. **Aplica modelo** (se disponível) → para cada linha:
   ```javascript
   // Calcula TF-IDF da linha
   vector = tfidf(line, model.vocabulary, model.idf)
   
   // Calcula score do modelo
   score = vector * model.coef + model.intercept
   confidence = sigmoid(score)  // Converte para 0-1
   
   // Se confiança > 0.5, adiciona como candidato
   if (confidence > 0.5) {
       candidatos.push({line, via: 'model', score: confidence})
   }
   ```
5. **Combina scores** → heurísticas + modelo
6. **Seleciona melhor** → maior score após validação

### Por que o Modelo Ajuda?

- **Heurísticas** são boas para padrões conhecidos (11 dígitos = MAWB)
- **Modelo** aprende padrões específicos dos seus e-mails:
  - Formato de referência usado pelos seus remetentes
  - Onde CONSIGNEE geralmente aparece
  - Contexto textual que indica cada campo

---

## Estratégias para Melhorar o Modelo

### 1. **Coleta Estratégica de Dados**

#### Identifique Padrões de Erro
Após usar o modelo, anote:
- Quais campos erram mais?
- Que tipos de e-mail causam mais erros?
- Há formatos específicos que confundem o sistema?

#### Foque nos Casos Difíceis
- Adicione mais exemplos dos casos que erram
- Inclua variações: dark mode, baixa qualidade, layouts diferentes

### 2. **Ajustes no Script de Treinamento**

#### Aumentar Features (para mais dados)
```python
TfidfVectorizer(
    ngram_range=(1, 3),  # Incluir trigramas (palavras triplas)
    max_features=10000,  # Mais features
    min_df=1            # Incluir palavras raras
)
```

#### Ajustar Threshold de Confiança
No `index.html`, linha 436, você pode ajustar:
```javascript
if (confidence > 0.5) {  // Tente 0.4 para ser mais permissivo
    arr.push({i, line, via: 'model', score: confidence});
}
```

#### Treinar Modelos Separados por Remetente
Se você tem remetentes muito diferentes, treine modelos específicos:
```python
# Agrupar por remetente (se tiver essa info nos dados)
for sender in unique_senders:
    sender_data = [d for d in data if d.get('sender') == sender]
    # Treinar modelo específico
```

### 3. **Pré-processamento de Texto Melhorado**

Adicione normalização antes do treinamento:
```python
def preprocess_line(line):
    # Normalizar espaços
    line = ' '.join(line.split())
    # Remover caracteres especiais desnecessários
    line = re.sub(r'[^\w\s\-/]', '', line)
    return line.lower().strip()

# Usar no treinamento
X_train.append(preprocess_line(line))
```

### 4. **Validação Cruzada e Métricas**

Adicione ao script para entender melhor o modelo:
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Acurácia média (5-fold CV): {scores.mean():.3f} (+/- {scores.std()*2:.3f})")
```

### 5. **Ensemble de Modelos**

Combine múltiplos modelos:
```python
# Treinar modelo com diferentes parâmetros
model1 = make_pipeline(TfidfVectorizer(ngram_range=(1,1)), LogisticRegression())
model2 = make_pipeline(TfidfVectorizer(ngram_range=(1,2)), LogisticRegression())

# Usar média dos scores
```

---

## Troubleshooting e Dicas Avançadas

### Problema: Modelo não melhora após treinar

**Possíveis causas:**
1. **Poucos dados**: < 50 exemplos por campo
2. **Dados inconsistentes**: valores corrigidos de forma diferente
3. **Overfitting**: modelo decorou exemplos, não aprendeu padrões

**Soluções:**
- Colete mais dados (100+ exemplos)
- Revise correções para consistência
- Reduza `max_features` ou aumente `min_df`

### Problema: Modelo muito grande (> 1MB)

**Solução:**
```python
TfidfVectorizer(
    max_features=2000,  # Reduzir vocabulário
    min_df=3           # Ignorar palavras raras
)
```

### Problema: Algum campo nunca é encontrado

**Diagnóstico:**
```python
# Verificar quantos exemplos positivos existem
positive_examples = sum(y_train)
print(f"Exemplos positivos: {positive_examples}")
```

**Solução:**
- Adicione mais exemplos desse campo
- Verifique se o valor correto realmente está no `raw_text`
- Ajuste threshold de confiança para esse campo específico

### Dica Avançada: Feature Engineering Manual

Adicione features customizadas:
```python
def extract_features(line):
    features = []
    # Número de dígitos
    features.append(len(re.findall(r'\d', line)))
    # Número de letras maiúsculas
    features.append(len(re.findall(r'[A-Z]', line)))
    # Contém padrão MAWB?
    features.append(1 if re.search(r'\d{11}', line) else 0)
    return features
```

### Dica Avançada: Active Learning

Após treinar, identifique exemplos onde o modelo tem **baixa confiança**:
- Esses são os casos mais informativos para adicionar ao treinamento
- Priorize corrigir esses casos na próxima iteração

---

## Checklist de Melhoria Contínua

Use este checklist para melhorar iterativamente:

- [ ] Tenho pelo menos 50 exemplos corrigidos
- [ ] Exportei e treinei o primeiro modelo
- [ ] Testei o modelo em 10-20 novos e-mails
- [ ] Identifiquei os 3 principais tipos de erro
- [ ] Adicionei mais exemplos dos casos que erram
- [ ] Retreinei o modelo
- [ ] Repeti o ciclo até acurácia satisfatória

**Meta de Acurácia:**
- MAWB: ≥ 95%
- HAWB: ≥ 90%
- DESTINO (IATA): ≥ 90%
- Demais campos: ≥ 80%

---

## Próximos Passos

1. **Comece simples**: Treine com 50-100 exemplos
2. **Teste e itere**: Use o modelo, identifique erros, adicione exemplos
3. **Refine**: Ajuste parâmetros conforme necessário
4. **Automatize**: Após 200+ exemplos, o modelo deve estar muito melhor

**Lembre-se**: Machine Learning é um processo iterativo. Cada ciclo de coleta → treino → teste → correção torna o modelo mais inteligente! 🚀

---

## 📊 Interpretando os Resultados do Modelo

### Entendendo as Métricas

Quando você treina o modelo, ele mostra algumas métricas. Aqui está o que significam:

#### Acurácia (Accuracy)
- **O que é**: Porcentagem de previsões corretas
- **Bom**: > 80% para campos estruturados (MAWB, HAWB)
- **Atenção**: Pode ser enganosa se houver muitos exemplos negativos

#### Exemplos Positivos vs Negativos
- **Positivos**: Linhas que contêm o valor correto
- **Negativos**: Linhas que não contêm
- **Ideal**: 5-20% de positivos (dados desbalanceados são normais)

### Como o Modelo Decide

O modelo calcula um **score** para cada linha:

```python
score = (TF-IDF da linha) × (pesos aprendidos) + intercept
confiança = sigmoid(score)  # Converte para 0-1
```

- **confiança > 0.5**: Modelo acha que a linha contém o valor
- **confiança < 0.5**: Modelo acha que não contém

### Visualizando o que o Modelo Aprendeu

Adicione este código ao script para ver as palavras mais importantes:

```python
# Após treinar cada modelo
vectorizer = model.named_steps['tfidfvectorizer']
classifier = model.named_steps['logisticregression']

# Pegar top 10 palavras mais importantes
feature_names = vectorizer.get_feature_names_out()
coef = classifier.coef_[0]

# Ordenar por importância (coeficiente)
top_indices = coef.argsort()[-10:][::-1]
print(f"\n🔝 Top 10 palavras mais importantes para {field}:")
for idx in top_indices:
    print(f"  {feature_names[idx]}: {coef[idx]:.3f}")
```

Isso mostra quais palavras o modelo associa com cada campo!

---

## 🎯 Estratégias Específicas por Campo

### REFERENCIA
- **Desafio**: Formato muito variável
- **Estratégia**: Foque em exemplos com diferentes formatos (AB123, 123-ABC, etc.)
- **Dica**: O modelo aprende melhor quando há padrões consistentes no contexto

### MAWB
- **Desafio**: Pode ser confundido com outros números (telefone, CEP)
- **Estratégia**: Inclua exemplos negativos (números de 11 dígitos que NÃO são MAWB)
- **Dica**: O modelo deve aprender o contexto (próximo a "MAWB", "AWB")

### HAWB
- **Desafio**: Similar a REFERENCIA, mas geralmente mais curto
- **Estratégia**: Diferencie claramente HAWB de REFERENCIA nos dados
- **Dica**: Se aparecerem juntos, o modelo aprenderá a diferença

### DESTINO (IATA)
- **Desafio**: Pode haver múltiplos códigos IATA no texto
- **Estratégia**: Inclua exemplos com múltiplos IATAs e marque qual é o destino
- **Dica**: O modelo aprende contexto (próximo a "destino", "to", "para")

### DESTINO_FINAL
- **Desafio**: Texto livre, sem formato fixo
- **Estratégia**: Inclua muitas variações (cidades, DTA, recintos)
- **Dica**: O modelo precisa aprender palavras-chave contextuais

### CONSIGNEE
- **Desafio**: Nomes/razões sociais variam muito
- **Estratégia**: Inclua exemplos com e sem sufixos (LTDA, S/A, ME)
- **Dica**: O modelo aprende padrões de capitalização e estrutura

---

## 🔧 Script Avançado: Treinamento com Validação Detalhada

Use este script para obter insights mais profundos:

```python
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Carregar dados
with open('training_data.json', 'r') as f:
    data = json.load(f)

def featurize(raw_text):
    return raw_text.split('\n')

fields_to_train = ['REFERENCIA', 'MAWB', 'HAWB', 'DESTINO', 'DESTINO_FINAL', 'CONSIGNEE']
models = {}

for field in fields_to_train:
    print(f"\n{'='*60}")
    print(f"📊 {field}")
    print(f"{'='*60}")
    
    X_train = []
    y_train = []
    
    for item in data:
        lines = featurize(item['raw_text'])
        correct_value = item['fields'].get(field, '').strip()
        
        if not correct_value:
            continue
        
        for line in lines:
            X_train.append(line)
            y_train.append(1 if correct_value.upper() in line.upper() else 0)
    
    if sum(y_train) == 0:
        print(f"⚠️  Sem exemplos positivos")
        continue
    
    # Estatísticas
    pos = sum(y_train)
    total = len(y_train)
    print(f"📈 Exemplos: {total} total | {pos} positivos ({100*pos/total:.1f}%)")
    
    # Dividir treino/teste
    if total > 20:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
    else:
        X_tr, y_tr = X_train, y_train
        X_te, y_te = [], []
    
    # Treinar
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2),
        LogisticRegression(class_weight='balanced', max_iter=1000)
    )
    model.fit(X_tr, y_tr)
    models[field] = model
    
    # Avaliar
    if len(X_te) > 0:
        y_pred = model.predict(X_te)
        print(f"\n📊 Relatório de Classificação:")
        print(classification_report(y_te, y_pred, target_names=['Negativo', 'Positivo']))
        
        # Mostrar exemplos de erros
        print(f"\n🔍 Exemplos de Falsos Positivos (modelo achou, mas não é):")
        fp_count = 0
        for i, (line, true, pred) in enumerate(zip(X_te, y_te, y_pred)):
            if true == 0 and pred == 1 and fp_count < 3:
                print(f"  - {line[:80]}...")
                fp_count += 1
        
        print(f"\n🔍 Exemplos de Falsos Negativos (é, mas modelo não achou):")
        fn_count = 0
        for i, (line, true, pred) in enumerate(zip(X_te, y_te, y_pred)):
            if true == 1 and pred == 0 and fn_count < 3:
                print(f"  - {line[:80]}...")
                fn_count += 1
    
    # Top palavras
    vectorizer = model.named_steps['tfidfvectorizer']
    classifier = model.named_steps['logisticregression']
    feature_names = vectorizer.get_feature_names_out()
    coef = classifier.coef_[0]
    top_indices = coef.argsort()[-5:][::-1]
    print(f"\n🔝 Top 5 palavras importantes:")
    for idx in top_indices:
        print(f"  • {feature_names[idx]}: {coef[idx]:.3f}")

# Exportar
exported_model = {}
for field, model in models.items():
    vectorizer = model.named_steps['tfidfvectorizer']
    classifier = model.named_steps['logisticregression']
    exported_model[field] = {
        'vocabulary': vectorizer.vocabulary_,
        'idf': list(vectorizer.idf_),
        'coef': list(classifier.coef_[0]),
        'intercept': list(classifier.intercept_)
    }

with open('model.json', 'w') as f:
    json.dump(exported_model, f)

print(f"\n✅ Modelo exportado!")
```

Este script mostra:
- **Relatório detalhado** de precisão, recall, F1-score
- **Exemplos de erros** para entender o que o modelo confunde
- **Palavras mais importantes** que o modelo aprendeu

---

## 💡 Dicas Finais

1. **Paciência**: Melhorar um modelo leva tempo e iterações
2. **Qualidade > Quantidade**: 100 exemplos bem corrigidos > 500 mal corrigidos
3. **Consistência**: Mantenha o mesmo formato de correção sempre
4. **Teste regularmente**: Após cada treinamento, teste em novos dados
5. **Documente**: Anote quais tipos de e-mail causam mais erros

**Boa sorte com o treinamento!** 🎓🚀
