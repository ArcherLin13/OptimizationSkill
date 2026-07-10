// Greedy CTC decode (CPU): mirrors production decodetext logic.
// blank index = 0

const BLANK_IDX = 0;

function makeCharset(charSize) {
  const charset = new Array(charSize);
  charset[BLANK_IDX] = "";
  for (let i = 1; i < charSize; i++) {
    charset[i] = String.fromCharCode(0x4e00 + (i % 0x3400));
  }
  return charset;
}

function findMaxProbability(probsRow, charSize) {
  let bestK = 0;
  let bestP = probsRow[0];
  for (let k = 1; k < charSize; k++) {
    const p = probsRow[k];
    if (p > bestP) {
      bestP = p;
      bestK = k;
    }
  }
  return [bestK, bestP];
}

/**
 * Original path: scan full probs row each timestep.
 * Returns { prob_preds, loc_preds, pred_texts } per batch item.
 */
function decodeTextFromProbs(probs, seqlen, charSize, charset) {
  const prob_preds = [];
  const loc_preds = [];
  const pred_texts = [];
  let last_p = -1;

  for (let t = 0; t < seqlen; t++) {
    const row = probs.subarray(t * charSize, (t + 1) * charSize);
    const [p, max_prob] = findMaxProbability(row, charSize);
    if (p !== last_p && p !== BLANK_IDX) {
      prob_preds.push(max_prob);
      loc_preds.push(t);
      pred_texts.push(charset[p]);
    }
    last_p = p;
  }

  return { prob_preds, loc_preds, pred_texts };
}

/**
 * Fused path: token_ids / max_probs already computed on GPU (or CPU fused kernel).
 */
function decodeTextFromArgmax(tokenIds, maxProbs, seqlen, charset) {
  const prob_preds = [];
  const loc_preds = [];
  const pred_texts = [];
  let last_p = -1;

  for (let t = 0; t < seqlen; t++) {
    const p = tokenIds[t];
    const max_prob = maxProbs[t];
    if (p !== last_p && p !== BLANK_IDX) {
      prob_preds.push(max_prob);
      loc_preds.push(t);
      pred_texts.push(charset[p]);
    }
    last_p = p;
  }

  return { prob_preds, loc_preds, pred_texts };
}

function decodeResultsEqual(a, b) {
  if (a.prob_preds.length !== b.prob_preds.length) return false;
  for (let i = 0; i < a.prob_preds.length; i++) {
    if (a.loc_preds[i] !== b.loc_preds[i]) return false;
    if (a.pred_texts[i] !== b.pred_texts[i]) return false;
    if (Math.abs(a.prob_preds[i] - b.prob_preds[i]) > 1e-5) return false;
  }
  return true;
}

/** CPU fused: argmax on logits + softmax prob at argmax (1× exp scan). */
function softmaxArgmaxFused(logits, tokenIds, maxProbs, seqlen, charSize) {
  for (let j = 0; j < seqlen; j++) {
    const off = j * charSize;
    let bestK = 0;
    let mx = logits[off];
    for (let k = 1; k < charSize; k++) {
      const v = logits[off + k];
      if (v > mx) {
        mx = v;
        bestK = k;
      }
    }
    let sum = 0;
    for (let k = 0; k < charSize; k++) {
      sum += Math.exp(logits[off + k] - mx);
    }
    tokenIds[j] = bestK;
    maxProbs[j] = 1 / sum;
  }
}

module.exports = {
  BLANK_IDX,
  makeCharset,
  findMaxProbability,
  decodeTextFromProbs,
  decodeTextFromArgmax,
  decodeResultsEqual,
  softmaxArgmaxFused,
};
