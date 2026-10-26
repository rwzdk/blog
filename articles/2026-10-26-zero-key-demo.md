# 零 API Key 的 AI Demo：我是怎么让全栈 Agent 应用"空配置也能完整体验"的

## 一、先算一笔账：你的 Demo 经得起 30 秒考验吗

想象一个场景：某位招聘方从你简历里的链接点开了你的在线项目。他扫了一眼 README，看到的第一行命令是 `cp .env.example .env`，第二行是"去 OpenAI 申请 API Key"。

这 30 秒里发生的事情，大概率是关掉标签页。

这不是夸张。绝大多数 AI 项目的"体验成本"是这样的：注册第三方平台 → 绑卡充值 → 拿 Key → 填环境变量 → 重启服务。五步流程里每一步都有流失，而"注册账号 + 拿 Key"是我见过最陡峭的一步——对一个只是"随便看看"的访客来说，这个成本远高于他的好奇心。作品集项目的流量本来就是一次性的：他这次没体验成，不会有下一次。

更糟的是另一个常见方案：往线上环境塞一个公共 Key 的"演示账号"。第二天额度被脚本刷爆，Demo 直接 500。这不是体验，是定时炸弹。

所以我给自己两个全栈项目立了一条硬规矩——**不配置任何 API Key，项目必须完整可体验**。不是"能打开首页"，而是核心链路全都真实跑起来：

- **Agent Workbench**（工具调用型 Agent 工作台）：流式输出、多步工具调用、工具过程可视化、笔记读写，零 Key 全部可用；
- **RAG 知识库问答**：文档上传、分块、检索、带引用的回答，零 Key 照样完整走通。

这篇文章拆解背后的三层降级设计、演示模式的实现细节，以及一个"打包器免疫"的小技巧。

## 二、三层降级设计总览

先给全景。两个项目里，所有依赖外部服务的层都设计了降级路径，一张表看完：

| 层 | 满配形态（有 Key） | 零 Key 降级形态 | 判断依据 |
| --- | --- | --- | --- |
| LLM 层 | OpenAI 兼容接口：流式多步 Agent / 引用式回答 | Agent Workbench：手写数据流的**演示模式**（模拟提问 + 真实工具执行）；RAG：本地摘要式回答 | `OPENAI_API_KEY` 是否存在 |
| 检索层 | BM25 + 向量双路召回，RRF 融合排序 | 纯 BM25 词项检索（中文 bigram 分词，无需词典） | `EMBEDDINGS_API_KEY` + `EMBEDDINGS_MODEL` |
| 数据层 | Cloudflare D1（Workers 部署） | 本地 SQLite 文件（libSQL）；只读文件系统再回退临时目录 | 运行时探测 Cloudflare 环境，不看环境变量 |

三个设计要点：

**降级是能力降级，不是功能消失。** 零 Key 时用户依然能提问、能检索、能看到流式回答和引用卡片——只是检索从"语义 + 关键词混合"退到"关键词"，回答从"LLM 生成"退到"本地摘要"。功能骨架完整保留。

**模式必须可观测。** RAG 项目有一个 `/api/config` 接口，前端用徽章实时显示 `llm` / `embeddings` 当前是否可用。访客永远知道自己在用什么模式，而不是被蒙在鼓里。

**降级判断收敛在服务端。** 前端不感知 Key 的有无，只消费能力状态。这样同样的代码可以同时服务满配用户和零 Key 访客。

## 三、演示模式的精髓：不是假数据，而是"模拟提问 + 真实执行"

零 Key 场景下，最容易想到的方案是"录一段假数据播放"。但假数据骗不过内行：工具卡片里的结果是硬编码的，笔记列表里查不到刚"保存"的那条笔记——一旦被发现是演的，可信度直接为负。面试官顺手一问"这个计算是真的吗"，项目就塌了。

我的方案是：**只模拟"提问"，不模拟"执行"**。Agent Workbench 在未配置 `OPENAI_API_KEY` 时，会播放一次「模拟提问 + 真实工具执行」的完整过程——表达式用自研递归下降解析器真实求值，笔记真实写入数据库。来看真实代码（`src/app/api/chat/route.ts`，有精简）：

```ts
// 入口只有一个 if，真实模式的代码零改动
const apiKey = process.env.OPENAI_API_KEY;
if (!apiKey) return createDemoResponse(user.id);

function createDemoResponse(userId: number) {
  return createDataStreamResponse({
    execute: async (writer) => {
      const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
      const expression = '(1234 * 56 / 7) ^ 0.5';
      // 第一步真实求值：自研解析器，不走 eval
      const value = evaluateExpression(expression);
      const noteContent = `${expression} = ${value.toFixed(4)}`;

      writer.write(formatDataStreamPart('text',
        '🔧 **演示模式**（未配置 `OPENAI_API_KEY`）：以下是一次「模拟提问 + **真实工具执行**」的 Agent 过程演示。\n\n' +
        '> 帮我算 (1234 × 56 ÷ 7) ^ 0.5，并把结果记成笔记\n\n'));
      await sleep(600);

      // 模拟"模型决定调用计算器"
      writer.write(formatDataStreamPart('tool_call',
        { toolCallId: 'demo-calc', toolName: 'calculator', args: { expression } }));
      await sleep(800);
      writer.write(formatDataStreamPart('tool_result',
        { toolCallId: 'demo-calc', result: { ok: true, expression, value } }));
      await sleep(600);

      // 模拟"模型决定保存笔记"——但写入是真的
      writer.write(formatDataStreamPart('tool_call',
        { toolCallId: 'demo-note', toolName: 'save_note',
          args: { title: '计算结果', content: noteContent } }));
      await sleep(700);
      await ensureSchema();
      const db = await getDb();
      const rs = await db.execute({
        sql: 'INSERT INTO notes (user_id, title, content) VALUES (?, ?, ?)',
        args: [userId, '计算结果', noteContent],   // 真实落库！
      });
      writer.write(formatDataStreamPart('tool_result',
        { toolCallId: 'demo-note',
          result: { ok: true, noteId: Number(rs.lastInsertRowid), /* ... */ } }));

      await sleep(400);
      writer.write(formatDataStreamPart('text',
        `✅ 两步工具调用完成：计算结果 **≈ ${value.toFixed(2)}**，笔记已真实写入 SQLite（之后可以问"我记过哪些笔记"来检索它）。\n\n` +
        '在 `.env` 中配置 `OPENAI_API_KEY` 后，任意问题都会走真实的多步 Agent 流程。'));
    },
  });
}
```

拆解几个关键决策：

**1. 流式分片是手写的协议，不是 setInteval 吐字符串。** `createDataStreamResponse + formatDataStreamPart` 是 Vercel AI SDK 数据流协议的底层原语——`streamText` 内部产出的也是同样的分片格式。我按真实 Agent 的节奏依次写入 `text`、`tool_call`、`tool_result` 分片，中间用 600~800ms 的 `sleep` 制造"模型在思考"的间隙。结果是**前端零改动**：`useChat` 分辨不出这是手写的流，打字机文本、工具卡片折叠、状态流转全都照常渲染。

**2. 工具执行为什么必须是真的？** 三个理由。其一，可信可验证——访客看完演示后问一句"我记过哪些笔记"，`search_notes` 工具能把刚才那条查出来，因为 `INSERT INTO notes` 真的执行了。其二，演示模式复用的 `evaluateExpression` 和数据库写入路径，就是生产模式的同一条路径，不存在"演示专用代码腐化"的问题。其三，这条路径本身被单元测试覆盖，演示坏了 CI 会报警，而不是招聘方先发现。

**3. 诚实的文案是加分项。** 演示开头明说"这是演示模式、怎么升级"，结尾再给一次提示。藏着掖着一旦被识破是灾难，坦诚反而显得工程素养在线。

## 四、数据库双实现：一份业务代码跑本地和 Cloudflare

第三层降级在数据层。两个项目都用同一套方案：本地开发用 SQLite 文件，Cloudflare Workers 部署用 D1，而业务代码只有一份。核心是先定义一个极简接口：

```ts
export interface DbLike {
  execute(stmt: DbStatement): Promise<DbResult>;
  batch(stmts: DbStatement[]): Promise<DbResult[]>;
}

export async function getDb(): Promise<DbLike> {
  if (cached) return cached;
  try {
    // Cloudflare（OpenNext）环境：优先用绑定的 D1
    const { getCloudflareContext } = await import('@opennextjs/cloudflare');
    const { env } = await getCloudflareContext();
    const d1 = (env as { DB?: D1Like }).DB;
    if (d1) return (cached = createD1Adapter(d1));
  } catch { /* 非 Cloudflare 环境，走本地 libSQL */ }
  return (cached = await createLocalClient());
}
```

动态 `import` 探测运行环境，业务代码从头到尾只见过 `execute / batch` 两个方法，对底下是 D1 还是 SQLite 完全无感。

真正值得一提的坑在 `createLocalClient` 里——**用 `createRequire` 在运行时加载 libSQL 客户端**：

```ts
async function createLocalClient(): Promise<DbLike> {
  // 用 createRequire 在运行时加载，避免打包器（OpenNext/esbuild）尝试把
  // 含原生二进制的 libSQL 客户端打进 worker bundle——Cloudflare 上永远走不到这个分支
  const { createRequire } = await import('node:module');
  const nodeRequire = createRequire(path.join(process.cwd(), 'package.json'));
  const { createClient } = nodeRequire('@libsql/client') as typeof import('@libsql/client');
  const client = createClient({
    url: process.env.LIBSQL_URL ?? localFileUrl(),
    authToken: process.env.LIBSQL_AUTH_TOKEN,
  });
  /* 适配成 DbLike ... */
}
```

为什么不能直接 `import { createClient } from '@libsql/client'`？因为 esbuild / OpenNext 做的是**静态依赖分析**：只要源码里出现了这条 import，打包器就会把 `@libsql/client` 连同它带的原生二进制一起塞进 worker bundle，构建直接报错——哪怕这段代码在 Cloudflare 上根本不会执行。改用 `createRequire` 之后，这条依赖路径对打包器不可见：Workers 环境走 D1 分支，永远碰不到它；本地 Node 环境在运行时正常加载。我称之为"打包器免疫"技巧，同类思路适用于一切"条件性原生依赖"。

顺带一提 `localFileUrl` 里还有第三层小降级：默认写 `data/app.db`；如果文件系统只读（部分 Serverless 环境），自动回退到 `os.tmpdir()` 下的临时目录。能跑起来的路径，永远比理想的路径多一条。

## 五、降级的工程原则

做完这两个项目，我把零配置体验沉淀成三条原则：

**原则一：能降级的功能才敢上线。** "必须配 Key 才能用"意味着你的 Demo 和一份编译不过的代码没有区别——它永远处于"未激活"状态。反过来，每一层都有降级路径的系统，部署时不需要祈祷环境变量齐全，任何人任何时刻点开链接都是活的。

**原则二：降级路径本身要被测试覆盖。** "零 Key 可跑"不是口号，是 CI 事实：两个项目合计 43+ 个单元测试（Vitest），全部离线可跑、不碰任何网络。RAG 项目单独占了 26+，覆盖分块器、中文 bigram 分词、BM25、余弦相似度、RRF 融合和评估器——换句话说，零 Key 模式依赖的每一块纯函数逻辑都有断言。检索质量还有量化门禁：`npm run eval` 在 6 篇标注语料 × 6 个问题上跑本地检索，断言 **hit@3 ≥ 0.83、MRR ≥ 0.78**，改分块参数或 BM25 的 k1/b 导致指标退化，CI 直接挡下。降级代码的回归在 CI 被抓住，而不是在招聘方面前暴露。

**原则三：把降级当产品功能设计，而不是异常处理。** 徽章实时显示当前模式、演示模式开头坦诚说明、结尾给出升级路径——降级体验的每一步都被认真设计过。访客看到的不只是"这个项目能跑"，而是"作者考虑过我没有 Key 的情况"。这本身就是最好的工程能力证明。

## 六、写在最后

"零配置可体验"看起来是个锦上添花的小功能，但它改变的是项目的性质：从"需要激活的代码"变成"永远在线的产品"。而实现它的过程——手写数据流协议、设计降级链路、解决跨运行时打包——每一处都是可以写进简历、经得起面试追问的硬细节。

两个项目都在线上，点开即用，不需要任何 Key：

- Agent Workbench（全栈 AI Agent 工作台）：<https://agent-workbench.qixin-study.workers.dev>
- RAG 知识库问答（混合检索 + 引用溯源）：<https://rag-knowledge-base.qixin-study.workers.dev>
- GitHub 主页（源码 + 完整 README）：<https://github.com/rwzdk>


