# 中文 RAG 检索不想只靠向量：手写 BM25 + RRF 混合检索，用 hit@3 / MRR 证明它有效

> 本文所有代码来自我的开源项目 [rag-knowledge-base](https://github.com/rwzdk/rag-knowledge-base)（[线上 Demo](https://rag-knowledge-base.qixin-study.workers.dev)，零 API Key 可完整体验），检索层全部手写、零外部依赖，评估纳入 CI。

## 0. 先说结论

- 纯向量检索对中文知识库常见查询（含专业术语、缩写、产品名）**不够用**；BM25 词项检索恰好互补
- 两路结果用 **RRF（倒数排名融合）** 合并，只需要排名、不需要调分数权重，一行公式解决量纲不可比问题
- 检索质量要用 **hit@3 / MRR** 量化，并把它做成 CI 门禁——否则"调了个参数"永远是拍脑袋
- 千级分块规模下，纯 JS 的 BM25 + 暴力余弦是毫秒级，**不需要向量数据库**

## 1. 为什么向量检索不够

做 RAG Demo 最顺手的路径是：embedding 接口 + 向量库 + 相似度 Top-K。但真实知识库的查询长这样：

- "BM25 和向量检索分别适合什么场景？"——查询里的 **BM25** 是精确词项，向量把它"糊"进语义空间后，可能和一篇泛泛讲"检索策略"的文档相似度更高
- 产品编号、人名、接口名、缩写——这些**字面精确**的信号，embedding 反而不敏感

词项检索（BM25）在这些场景恰好相反：字面命中强，但"表述不同含义相同"就抓不住。所以正确的答案是**两路都跑，融合排序**——这正是搜索行业的标准做法（Elastic 的 RRF、各类 hybrid search）。

## 2. 无词典的中文分词：bigram

混合检索的第一步是分词。引 jieba？可以，但我选了更轻的方案——**CJK 二元组（bigram）**：

```ts
export function tokenize(text: string): string[] {
  const tokens: string[] = [];
  for (const word of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) tokens.push(word);
  for (const run of text.match(/[\u4e00-\u9fff]+/g) ?? []) {
    if (run.length === 1) tokens.push(run);
    else for (let i = 0; i + 1 < run.length; i++) tokens.push(run.slice(i, i + 2));
  }
  return tokens;
}
// "倒排索引" → [倒排, 排索, 索引]
// "BM25 是什么" → [bm25, 是什, 什么]
```

理由：**无词典、无外部依赖、召回高于单字**。查询"倒排索引"和文档"倒排索引结构"会共享 bigram（倒排/索引），即使文档里没有连续的四字词。代价是索引膨胀，但对千级分块完全无感。

## 3. BM25：40 行的排序函数

BM25 的直觉：一个词在这篇文档出现越多（TF↑）、在全体文档出现越少（IDF↑），且文档本身不太长（长度归一化），得分越高。我用的是 Lucene 风格 IDF（`ln(1 + (N-df+0.5)/(df+0.5))`），避免出现负值：

```ts
export function bm25(queryTokens: string[], docs: LexDoc[]): Scored[] {
  const N = docs.length;
  const avgdl = docs.reduce((s, d) => s + d.tokens.length, 0) / N;
  const queries = [...new Set(queryTokens)];
  // df：包含该词的文档数
  const df = new Map<string, number>();
  for (const q of queries) {
    let count = 0;
    for (const d of docs) if (d.tokens.includes(q)) count++;
    df.set(q, count);
  }
  const results: Scored[] = [];
  for (const d of docs) {
    const tf = new Map<string, number>();
    for (const t of d.tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
    let score = 0;
    for (const q of queries) {
      const f = tf.get(q) ?? 0;
      if (f === 0) continue;
      const idf = Math.log(1 + (N - (df.get(q) ?? 0) + 0.5) / ((df.get(q) ?? 0) + 0.5));
      score += (idf * (f * (K1 + 1))) / (f + K1 * (1 - B + (B * d.tokens.length) / avgdl));
    }
    if (score > 0) results.push({ id: d.id, score });
  }
  return results.sort((a, b) => b.score - a.score);
}
```

k1=1.5、b=0.75 是经典默认值。**它是纯函数**——这是后文能做确定性评估的前提。

## 4. RRF：为什么不用分数加权

现在有两路排序：BM25 榜单（分数量纲：无上界的 IDF 加权和）和余弦榜单（量纲：[-1, 1]）。怎么合并？

加权和 `α·bm25 + β·cos` 要调 α、β，且两路分数分布完全不同，调参是玄学。RRF 只用**排名**：

```ts
export function rrfFuse(rankings: { id: number; score: number }[][], k = 60): FusedScore[] {
  const scores = new Map<number, number>();
  for (const ranking of rankings) {
    ranking.forEach((item, i) => {
      scores.set(item.id, (scores.get(item.id) ?? 0) + 1 / (k + i + 1));
    });
  }
  return [...scores.entries()]
    .map(([id, score]) => ({ id, score }))
    .sort((a, b) => b.score - a.score);
}
```

`score(d) = Σ 1/(60 + rank_i(d))`，k=60 是原论文经验值。两路都靠前的文档自然登顶，天然规避量纲问题，**零参数可调**。我的线上 Demo 里引用卡片显示的 RRF 分数 `0.0164` 就是单路第一名 1/61——公式直接可验。

## 5. 评估：让"检索变好了"变成一句话数据

这是大多数 RAG Demo 缺失的一环。做法很朴素：

1. 固定 6 篇中文语料 × 6 个标注问题（每题标注正确分块 id）
2. 跑本地检索取 Top-3，算两个指标：
   - **hit@3**：正确分块进前 3 的比例
   - **MRR**：正确分块排名倒数的平均值（第 1 名 = 1.0，第 2 名 = 0.5…）
3. **断言阈值，纳入 CI**

```ts
it('hit@3 ≥ 0.83，MRR ≥ 0.78', () => {
  let hits = 0, mrrSum = 0;
  for (const { query, expectId } of annotations) {
    const top3 = retrieveTop(query, 3);
    const rank = top3.indexOf(expectId);
    if (rank !== -1) { hits++; mrrSum += 1 / (rank + 1); }
  }
  const hitAt3 = hits / annotations.length;
  const mrr = mrrSum / annotations.length;
  console.info(`hit@3 = ${hitAt3.toFixed(2)}，MRR = ${mrr.toFixed(2)}`);
  expect(hitAt3).toBeGreaterThanOrEqual(0.83);
  expect(mrr).toBeGreaterThanOrEqual(0.78);
});
```

当前基线：**hit@3 = 1.00，MRR = 1.00**。从此以后任何人改分块大小、换 BM25 参数、加 rerank，CI 会立刻告诉你有没有退步——"我觉得变好了"变成"hit@3 从 0.83 升到 1.00"。

## 6. 两个工程决策

**零 Key 降级**：没配 embedding Key 就纯 BM25，没配 LLM Key 就把命中分块做摘要式回答——整条链路在任何配置下都能跑。这让招聘方点开 Demo 时不需要任何凭据就能看到完整产品形态，也让全链路可以离线测试。

**什么时候换真向量索引**：千级分块以下，暴力余弦 O(n·d) 就是毫秒级，纯 JS 足够；到了万级再上 sqlite-vec / Turso Vector。"什么时候换、为什么换"比"上来就用向量库"更有说服力。

## 7. 总结

- 混合检索不是可选项：词项与语义是互补信号，RRF 让融合零调参
- 评估先行：hit@k / MRR 标注集 + CI 门禁，是 RAG 项目从 Demo 到工程的分水岭
- 全部代码可抄作业：[rag-knowledge-base](https://github.com/rwzdk/rag-knowledge-base)（Next.js 15 + Cloudflare Workers + D1 部署，[线上 Demo](https://rag-knowledge-base.qixin-study.workers.dev)）
- 同系列：[mini-agents](https://github.com/rwzdk/mini-agents)——零依赖的 TypeScript 多智能体编排库，另一篇设计记 soon

---


