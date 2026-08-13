/**
 * Browser fallback NLP: deterministic TF-IDF vectorisation + cosine similarity.
 * Mirrors backend/nlp/embeddings.py:TfidfBackend so results match the Python
 * fallback exactly. The Python service uses Sentence Transformers when running.
 */

const TOKEN_RE = /[a-z0-9+#.]+/g;
const STOPWORDS = new Set([
  "the", "and", "for", "with", "you", "our", "are", "will", "have", "has", "this", "that",
  "from", "your", "who", "was", "were", "not", "but", "can", "all", "any", "out", "use",
  "using", "into", "about", "their", "them", "they", "his", "her", "she", "him", "its",
  "job", "role", "work", "team", "years", "year", "new", "per", "via", "etc",
]);

export function tokenize(text: string): string[] {
  return (text.toLowerCase().match(TOKEN_RE) ?? []).filter(
    (t) => t.length > 2 && !STOPWORDS.has(t),
  );
}

export function encodeTfidf(texts: string[]): number[][] {
  const docs = texts.map(tokenize);
  const vocab = Array.from(new Set(docs.flat())).sort();
  const index = new Map(vocab.map((tok, i) => [tok, i]));
  const nDocs = Math.max(1, docs.length);
  const df = new Map<string, number>();
  for (const doc of docs) {
    for (const tok of new Set(doc)) df.set(tok, (df.get(tok) ?? 0) + 1);
  }
  return docs.map((doc) => {
    const counts = new Map<string, number>();
    for (const tok of doc) counts.set(tok, (counts.get(tok) ?? 0) + 1);
    const total = Math.max(1, doc.length);
    const vec = new Array(vocab.length).fill(0);
    for (const [tok, count] of counts) {
      const idf = Math.log((1 + nDocs) / (1 + (df.get(tok) ?? 0))) + 1;
      vec[index.get(tok)!] = (count / total) * idf;
    }
    return vec;
  });
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (!a.length || !b.length || a.length !== b.length) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i += 1) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}
