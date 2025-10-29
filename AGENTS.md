# AGENTS.md — Plano do Agente (MCP) para Extração em Browser e Envio ao Apps Script

## Visão Geral

Este documento especifica, de ponta a ponta, como o **agente MCP** deve implementar um coletor de dados a partir de **prints (screenshots) de e‑mails** com layout variável, executando **100% no navegador** (sem APIs pagas), usando **Tesseract.js (WASM)** para OCR, **heurísticas/regex** e inferência por **padrões/formatos**. O agente também define o **contrato de integração** com o projeto **Google Apps Script** já existente do usuário.

> Objetivo: A partir de uma imagem (PNG/JPG), extrair os campos:
>
> * **REFERENCIA** (aliases: `REFERÊNCIA`, `REFERENCIA`, `REF`, `ref`, `referencia` …)
> * **MAWB**
> * **HAWB**
> * **DESTINO** (e variações)
> * **DESTINO FINAL** (aliases: `REMOÇÃO`, `REMOCAO`, `DTA`, `remocçao` …)
> * **CONSIGNEE** (aliases: `CLIENTE`, `CNEE`, `Consignee` …)
>
> Muitas vezes os campos **não** estão explicitamente rotulados. A IA/heurísticas devem **inferir** pelo formato esperado, comprimento/charset, contexto, vizinhança textual e dados de treinamento rotulados.

---

## Arquitetura (Cliente, sem servidor)

1. **Upload da imagem** (input `<file>` ou drag‑and‑drop).
2. **Pré‑processamento** (Canvas API):

   * Redimensionar altura para ~**1600–2200 px** preservando proporção.
   * Conversão para **grayscale**; opcional **binarização leve** (threshold) e **deskew** simples (detectar bordas inclinadas via Hough e rotacionar).
3. **OCR** com **Tesseract.js** (WASM), idioma `por+eng`.
4. **Normalização de texto**:

   * Remover duplicatas visíveis; corrigir **quebra de palavras**; normalizar **acentos/diacríticos** (NFKD) e **casefold** para matching; manter uma **cópia bruta** `raw_text`.
5. **Extração** (pipeline de regras + inferência por formato):

   * **Rótulos → valor** quando existirem (tokenização por linha/bloco).
   * **Inferência sem rótulo** por **padrões** (ex.: MAWB = 11 dígitos; HAWB ≤ 11 chars [A‑Z0‑9-/]; IATA = 3 letras; DTA/remoção = texto/domínio específico; Consignee = linha com razão social/cliente próximo a aliases).
   * **Resolução de conflitos** por **pontuação de confiança** e distância/contexto.
6. **Validação & Normalização** por tipo de campo.
7. **Geração de JSON** conforme **Contrato de Saída**.
8. **Envio ao Apps Script** via `fetch()` (POST JSON). Se o Web App for privado, incluir **OAuth token** (fora do escopo deste doc) ou **token estático** (ver Segurança).

---

## Dicionário de Campos & Regras de Extração

### 1) REFERENCIA

* **Aliases**: `REFERENCIA`, `REFERÊNCIA`, `REF`, `ref`, `referencia`, `referência` (case‑insensitive; tolerar erros de OCR: `REFERENClA`, `REFERECIA`, etc. via distância de Levenshtein ≤ 2).
* **Formato esperado**: **alfa-numérico** curto (3–20 chars), frequentemente com `/`, `-`, `_`.
* **Heurísticas**:

  * Se linha contém alias ~ (± 20 chars), capturar o **token mais “ID‑like”** à direita ou na próxima linha.
  * Sem rótulo:

    * Procurar tokens **alfa-numéricos** com **mistura de letras e números** (p.ex. `AB12/345`), **não** parecidos com MAWB/HAWB, e que apareçam **uma única vez**; dar peso a termos próximos de `Assunto`/`Subject`.
* **Validação**: 3–30 chars, charset `[A-Z0-9\-_/]` pós‑normalização; remover espaços extremos.

### 2) MAWB

* **Formato canônico**: **11 dígitos** (IATA AWB): `NNN NNNNNNNN` (mas geralmente sem espaços/pontos). **Somente dígitos**.
* **Regex (tolerante a separadores)**: `(?<!\d)(\d[\s\.-]?){11}(?!\d)` → consolidar removendo separadores para obter exatamente 11 dígitos.
* **Heurísticas**:

  * Se houver **múltiplos** candidatos de 11 dígitos,

    * preferir aquele **perto** de aliases `MAWB`, `AWB`, `Master` (distância de linha pequena),
    * se não houver rótulos, usar o **maior score de “AWB‑like”** (apenas dígitos, 11, não parte de telefone/data/CEP → excluir por contextos e símbolos vizinhos).
* **Validação**: tamanho 11, todos dígitos; **checksum IATA** opcional (módulo 7) se disponível.

### 3) HAWB

* **Formato**: até **11 caracteres**; geralmente **alfa-numérico** com `-` ou `/`.
* **Regex**: `(?<![A-Z0-9])[A-Z0-9][A-Z0-9\-/]{1,10}(?![A-Z0-9])` (case‑insensitive no matching, mas normalizar para upper).
* **Heurísticas**:

  * Prefira candidato próximo a aliases `HAWB`, `House`, `HWB`.
  * Evitar colisão com **MAWB** (todos dígitos) e com **Referência** (se houver rótulo claro).
* **Validação**: comprimento ≤ 11; charset `[A-Z0-9\-/]`.

### 4) DESTINO

* **Aliases**: `DESTINO`, `Destination`, `To`, `Aeroporto destino`, `Airport` (contextual), `PORTO` (em alguns e‑mails, mas priorizar aéreo).
* **Formatos possíveis**:

  * **IATA 3‑letter**: `[A-Z]{3}` (ex.: `GRU`, `LAX`).
  * **Cidade/País**: palavra(s) com capitalização; se houver IATA + cidade, preferir **IATA**.
* **Heurísticas**:

  * Se encontrar **IATA** isolado próximo a aliases, considerar DESTINO.
  * Caso múltiplos IATAs, utilizar: o mais próximo de `DESTINO`, ou, se aparecer par `FROM/DE` → `TO/PARA`, escolher o lado `TO/PARA`.
* **Validação**: IATA deve ser `[A-Z]{3}`; caso cidade, manter string clara (ex.: `São Paulo (GRU)`).

### 5) DESTINO FINAL

* **Aliases**: `DESTINO FINAL`, `REMOÇÃO`, `REMOCAO`, `DTA`, `Remoção`, `Remocao`, `remocçao` (normalizar acentos/cedilha; distância Levenshtein ≤ 2).
* **Semântica**: local de **remoção/desembaraço** ou destino final terrestre (pós‑aéreo).
* **Formatos**:

  * Cidade/UF, armazém, recinto alfandegado, **DTA** indicado por sigla.
* **Heurísticas**:

  * Procurar linha com alias + valor textual a direita/abaixo.
  * Sem rótulo: quando aparecer `DTA` ou termos de **remessa interna**, preferir como `DESTINO FINAL` (se já existe `DESTINO` IATA distinto).
* **Validação**: string 2–60 chars; limpar múltiplos espaços.

### 6) CONSIGNEE

* **Aliases**: `CONSIGNEE`, `CNEE`, `CLIENTE`, `Cliente`, `Destinatário`.
* **Formato**: **nome/razão social** (pode incluir `LTDA`, `S/A`, `ME`, `EPP`).
* **Heurísticas**:

  * Procurar próximo a aliases.
  * Sem rótulo: procurar **linha/título** que pareça **razão social** (palavras com muitas letras, eventuais sufixos empresariais, ausência de números longos), **perto** de outras chaves logísticas (MAWB/HAWB).
  * Se houver **e‑mail de domínio corporativo** próximo, considerar linha acima como `CONSIGNEE`.
* **Validação**: min 3 chars; remover caudas como `:`; preservar capitalização original.

---

## Normalização, Pontuação e Conflitos

* **Pontuação por candidato** (0–1): soma ponderada de:

  * *Match de rótulo* (0.4)
  * *Formato válido* (0.3)
  * *Proximidade a aliases* (0.2)
  * *Unicidade/consistência global* (0.1)
* **Desempate**: maior score; se empate < 0.05, marcar `low_confidence`.
* **Confianças no JSON**: incluir por campo (ex.: `confidences.MAWB = 0.86`).
* **Normalização**:

  * **MAWB**: só dígitos, 11 chars.
  * **HAWB/REFERENCIA**: upper‑case; remover espaços extremos; manter `-`/`/`.
  * **IATA**: upper‑case 3 letras.
  * **Nomes**: título como no OCR (não forçar upper).

---

## Pré‑processamento (Canvas API)

* Redimensionar: `max(height, 2200)`; se > 3000 px, reduzir para **~2200 px**; se < 1200 px, **upar** para ~1600 px (melhora OCR).
* **Contraste**: aplicar `ctx.filter = 'contrast(1.2) brightness(1.05)'` (leve).
* **Binarização** (opcional): Otsu simples em JS ou limiar fixo adaptativo por média local.
* **Deskew**: heurística de detectar linhas horizontais e rotacionar (pode ser omitido inicialmente).

---

## Implementação (esqueleto de front‑end)

```html
<input type="file" id="img" accept="image/*" />
<pre id="out"></pre>
<script src="https://unpkg.com/tesseract.js@5/dist/tesseract.min.js"></script>
<script>
const WEBAPP_URL = "https://script.google.com/macros/s/SEU_DEPLOY_ID/exec"; // configurar

// --- Aliases (normalizados) ---
const aliases = {
  REFERENCIA: ["referencia","referência","ref"],
  DESTINO: ["destino","destination","to","porto","airport"],
  DESTINO_FINAL: ["destino final","remoção","remocao","dta","remocçao"],
  CONSIGNEE: ["consignee","cnee","cliente","destinatário","destinatario"],
  MAWB: ["mawb","awb","master"],
  HAWB: ["hawb","house","hwb"]
};

function norm(s){return s.normalize('NFKD').replace(/[\u0300-\u036f]/g,'').toLowerCase();}

function isIATA(tok){return /^[A-Z]{3}$/.test(tok);} // após toUpperCase()
function asMAWB(s){ const d = (s||'').replace(/[^0-9]/g,''); return d.length===11 ? d : null; }
function isHAWB(tok){ return /^[A-Z0-9][A-Z0-9\-/]{1,10}$/.test(tok); }
function isRefLike(tok){ return /^[A-Z0-9][A-Z0-9\-_/]{2,29}$/.test(tok); }

async function ocrImage(file){
  const ab = await file.arrayBuffer();
  const { data:{ text } } = await Tesseract.recognize(ab, 'por+eng');
  return text;
}

function extract(text){
  const raw = text; const up = text.toUpperCase(); const low = norm(text);
  const lines = raw.split(/\r?\n/);

  const cand = {REFERENCIA:[], MAWB:[], HAWB:[], DESTINO:[], DESTINO_FINAL:[], CONSIGNEE:[]};

  // 1) label-based
  lines.forEach((line,i)=>{
    const L = norm(line);
    for(const key of Object.keys(aliases)){
      for(const al of aliases[key]){
        if(L.includes(norm(al))){ cand[key].push({i, line, via:'label', score:0.4}); break; }
      }
    }
  });

  // 2) pattern-based scanning
  lines.forEach((line,i)=>{
    const U = line.toUpperCase();
    // MAWB candidates
    const ma = U.match(/[0-9][0-9 \.-]{9,20}[0-9]/g)||[];
    ma.forEach(m=>{ const v=asMAWB(m); if(v) cand.MAWB.push({i, line:m, val:v, via:'pattern', score:0.3}); });

    // HAWB/REF tokens
    U.split(/\s+/).forEach(tok=>{
      const t = tok.replace(/[^A-Z0-9\-/]/g,'');
      if(isHAWB(t)) cand.HAWB.push({i, line:tok, val:t, via:'pattern', score:0.3});
      if(isRefLike(t)) cand.REFERENCIA.push({i, line:tok, val:t, via:'pattern', score:0.2});
      if(isIATA(t)) cand.DESTINO.push({i, line:tok, val:t, via:'pattern', score:0.3});
    });
  });

  // 3) proximity boost (labels → nearest value)
  const boostNear = (key, pred) => {
    const labels = cand[key].filter(c=>c.via==='label');
    labels.forEach(lbl=>{
      for(let j=lbl.i; j<Math.min(lbl.i+3, lines.length); j++){
        const U = lines[j].toUpperCase();
        const toks = U.split(/\s+/).map(t=>t.replace(/[^A-Z0-9\-/]/g,''));
        toks.forEach(t=>{ if(pred(t)) cand[key].push({i:j, line:t, val:t, via:'prox', score:0.2}); });
      }
    });
  };
  boostNear('MAWB', t=>asMAWB(t));
  boostNear('HAWB', isHAWB);
  boostNear('REFERENCIA', isRefLike);
  boostNear('DESTINO', t=>/^[A-Z]{3}$/.test(t));

  // 4) scoring & selection
  function pick(key, validate){
    const arr = cand[key];
    // agrupar por valor normalizado
    const map = new Map();
    arr.forEach(c=>{
      const v = (c.val || c.line || '').toString().toUpperCase();
      if(!v) return; const prev = map.get(v)||{score:0, best:c};
      map.set(v,{score:prev.score + c.score, best: prev.best});
    });
    let bestV=null, bestS=-1;
    for(const [v, obj] of map){
      const ok = validate ? validate(v) : true;
      if(ok && obj.score>bestS){ bestS=obj.score; bestV=v; }
    }
    return bestV ? {value:bestV, confidence: Math.min(1, bestS)} : {value:null, confidence:0};
  }

  const out = {
    REFERENCIA: pick('REFERENCIA', v=>isRefLike(v)).value,
    MAWB: (()=>{ const p=pick('MAWB', v=>!!asMAWB(v)); return p.value? asMAWB(p.value):null; })(),
    HAWB: pick('HAWB', isHAWB).value,
    DESTINO: pick('DESTINO', v=>/^[A-Z]{3}$/.test(v)).value,
    DESTINO_FINAL: pick('DESTINO_FINAL', v=>v.length>1).value, // heurística textual
    CONSIGNEE: pick('CONSIGNEE', v=>v.length>=3).value,
    raw_text: raw
  };
  return out;
}

async function main(){
  const inp = document.getElementById('img');
  inp.addEventListener('change', async (e)=>{
    const f=e.target.files[0]; if(!f) return;
    const text = await ocrImage(f);
    const ex = extract(text);

    // Contrato de saída (ver seção abaixo)
    const payload = {
      source: 'screenshot',
      sender_name: null,
      sender_email: null,
      subject: null,
      sent_at: null,
      total_amount: null,
      currency: null,
      fields: {
        REFERENCIA: ex.REFERENCIA,
        MAWB: ex.MAWB,
        HAWB: ex.HAWB,
        DESTINO: ex.DESTINO,
        DESTINO_FINAL: ex.DESTINO_FINAL,
        CONSIGNEE: ex.CONSIGNEE
      },
      confidences: {},
      raw_text: ex.raw_text.slice(0, 20000)
    };

    document.getElementById('out').textContent = JSON.stringify(payload, null, 2);

    // Envio (opcional):
    // await fetch(WEBAPP_URL, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  });
}
main();
</script>
```

---

## Contrato de Saída (JSON → Apps Script)

```json
{
  "source": "screenshot",
  "sender_name": null,
  "sender_email": null,
  "subject": null,
  "sent_at": null,
  "total_amount": null,
  "currency": null,
  "fields": {
    "REFERENCIA": "…",
    "MAWB": "…",
    "HAWB": "…",
    "DESTINO": "…",
    "DESTINO_FINAL": "…",
    "CONSIGNEE": "…"
  },
  "confidences": {
    "REFERENCIA": 0.0,
    "MAWB": 0.0,
    "HAWB": 0.0,
    "DESTINO": 0.0,
    "DESTINO_FINAL": 0.0,
    "CONSIGNEE": 0.0
  },
  "raw_text": "<até 20k chars do OCR>"
}
```

> Observação: mesmo que as confianças não estejam implementadas inicialmente, manter o campo para evolução.

---

## Integração com Google Apps Script

* O projeto do usuário já possui (ou terá) um **Web App** no Apps Script com `doPost(e)` para receber o JSON.
* **URL**: `https://script.google.com/macros/s/SEU_DEPLOY_ID/exec`
* **Autorização**:

  * Ambiente de testes: `Anyone (even anonymous)`.
  * Produção: restringir acesso e usar **token estático** no header (ex.: `X-API-Key`) ou **OAuth**.
* **Contrato de gravação**: o `doPost` deve:

  1. parsear JSON;
  2. gravar em planilha (aba `Inbox`), colunas mínimas: `timestamp`, `REFERENCIA`, `MAWB`, `HAWB`, `DESTINO`, `DESTINO_FINAL`, `CONSIGNEE`, `raw_text`;
  3. retornar `{ok:true}`.

### Exemplo de `doPost` (Apps Script)

```javascript
function doPost(e){
  try{
    const body = JSON.parse(e.postData.contents);
    const f = body.fields || {};
    const ss = SpreadsheetApp.openById('ID_DA_PLANILHA');
    const sh = ss.getSheetByName('Inbox') || ss.insertSheet('Inbox');
    sh.appendRow([
      new Date(), f.REFERENCIA||'', f.MAWB||'', f.HAWB||'', f.DESTINO||'', f.DESTINO_FINAL||'', f.CONSIGNEE||'',
      (body.raw_text||'').slice(0,20000)
    ]);
    return ContentService.createTextOutput(JSON.stringify({ok:true}))
      .setMimeType(ContentService.MimeType.JSON);
  }catch(err){
    return ContentService.createTextOutput(JSON.stringify({ok:false,error:err.message}))
      .setMimeType(ContentService.MimeType.JSON);
  }
}
```

---

## Dados de Treinamento (opcional, sem nuvem paga)

Embora o pipeline inicial baseie‑se em **heurísticas**, o agente pode evoluir usando **aprendizado supervisionado local**:

* **Rotulador simples no browser**: salvar exemplos de `raw_text` + valores corretos dos campos.
* **Features**: posição relativa (linha/coluna), presença de aliases, formato (regex match), charset, comprimento.
* **Modelos leves**:

  * **CRF** (Conditional Random Fields) para sequence labeling (executável via **WASM** com libs JS ou pré‑treinar offline e embarcar pesos);
  * **Logistic Regression / SVM** para classificar candidatos por campo.
* **Runtime**: usar **ONNX Runtime Web** ou **TensorFlow.js** para modelos pequenos. Treino pode ser **offline** e apenas o **inference** roda no navegador.

> Priorize iniciar com **regras + coleta de exemplos**. Só após 200–500 exemplos rotulados considerar modelo leve.

---

## Segurança e Privacidade

* **Local‑first**: todo OCR e extração ocorrem no **browser**; somente o **JSON** final é enviado ao Apps Script.
* **Sanitização**: truncar `raw_text` (≤ 20k chars); remover dados sensíveis não usados.
* **Chaves**: se usar `X-API-Key`, manter em variável de ambiente no deploy (quando houver) ou em Storage com escopo restrito; em ambiente puramente estático, preferir **Web App privado** + sessão autenticada Google.
* **Auditoria**: opcional, salvar hash da imagem (não a imagem) para referenciar o processamento.

---

## Testes & Aceite

* **Testes unitários de regex** para MAWB/HAWB/REFERENCIA/IATA.
* **Conjunto de 30–50 prints** com variações (dark mode, zoom, colunas, assinaturas longas) → meta:

  * MAWB **≥ 95%**, HAWB **≥ 90%**, DESTINO (IATA) **≥ 90%**, demais **≥ 80%** inicialmente.
* **Casos limite**: múltiplos MAWBs/HAWBs; nenhuma linha com aliases; números parecidos (telefone, CEP, CNPJ); PDF rasterizado; prints inclinados.
* **Critérios de aceite**: geração de JSON válido e gravação correta na planilha; latência ≤ 6 s para prints ≤ 2500 px em notebook comum.

---

## Roadmap

1. MVP (regras + UI simples de upload + JSON no `<pre>` + botão "Enviar").
2. Pré‑processamento melhor (deskew + binarização adaptativa).
3. Pontuação de confiança por campo.
4. Rotulador de exemplos no browser (localStorage/IndexedDB).
5. Modelo leve para desempate (CRF/Regressão) via ONNX/Tf.js.
6. Integração OAuth com Web App privado.

---

## Entregáveis do Agente MCP

* `index.html` + JS embutido (ou `main.js`) com Tesseract.js e pipeline de extração.
* `config.json` com `WEBAPP_URL` e flags (debug, enviar_automatico).
* `README.md` com instruções de uso.
* **Este `AGENTS.md`** como referência técnica e contrato.

> Qualquer divergência entre este documento e o código deve resultar em **log de aviso** e **não** bloquear o envio, apenas reduzir a confiança e marcar para revisão manual.
