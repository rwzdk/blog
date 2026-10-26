# 零依赖写一个多智能体编排库：mini-agents 设计记

> 本文所有代码来自我的开源项目 [mini-agents](https://github.com/rwzdk/mini-agents)（npm：[@ordeal/mini-agents](https://www.npmjs.com/package/@ordeal/mini-agents)）。它是零依赖的 TypeScript 多智能体编排库：Agent 循环、工具调用、pipeline / supervisor 两种编排模式，全部核心逻辑约 500 行可读源码，18 个确定性测试纳入 CI，运行时 `npm ls --prod` 是空的。

## 0. 先说结论

- Agent 的核心不是一个神秘框架，而是一个 `while (true)` 循环：**LLM 生成 → 解析工具调用 → 校验参数 → 本地执行 → 结果回传 → 再生成**。500 行 TypeScript 能把它写完整
- 三个最值得展开的设计决策：**工具错误当结果回传而非抛异常**、**maxSteps 到达后收走工具而非直接截断**、**事件流用 AsyncGenerator 而非回调**
- supervisor 的路由不是自由文本，而是给 router 注入 `delegate` / `finish` 两个**普通工具**——结构化委派可校验、可回放，router 无法"幻觉"出不存在的工人
- LLM 抽象成 `LlmClient` 接口注入，测试换成脚本化 Mock 后，**Agent 代码也能有传统意义上的单元测试**
- 诚实的定位：生产项目我照样用框架；这个轮子的价值是让"Agent 内部怎么运转"的答案写在代码里，而不是文档里

## 1. 为什么造这个轮子

先交代诚实的取舍。如果明天要上线一个生产级 Agent 功能，我会直接用 LangChain / LlamaIndex——重试、限流、记忆管理、可观测，框架踩过的坑比我的 500 行多得多，"生产用框架"这个结论我不打算推翻。

但**使用框架和理解机制是两回事**。被问"Agent 循环到底怎么运转、工具结果怎么回传、多智能体之间怎么路由"时，如果答案只能引用文档，说明理解还停在 API 层。所以我给这个库定的目标是：每一层都能亲手讲清楚——工具调用协议长什么样、参数校验失败谁兜底、编排循环怎么终止、测试怎么绕开网络。

顺带一个工程收益：零依赖意味着供应链为零。连 zod 都不用——OpenAI function calling 的 tools 参数本来就要求传 JSON Schema，直接复用这个协议格式做参数校验，不引入任何新概念。npm 安装它，等于只安装了你看到的这几百行代码。

## 2. 核心循环：Agent 到底是什么

先看主循环全貌（省略了 try/catch 和 `llm-request` 事件的 yield，完整版在 `src/agent.ts`）：

```ts
export async function* runAgent(options: RunAgentOptions): AsyncGenerator<AgentEvent> {
  const { agent, client } = options;
  const maxSteps = agent.maxSteps ?? 5;
  const messages: ChatMessage[] = [
    { role: 'system', content: agent.instructions },
    { role: 'user', content: options.input },
  ];
  let toolRounds = 0;
  let output = '';
  while (true) {
    // 达到轮数上限后收走工具，逼模型给出最终文本
    const tools = toolRounds >= maxSteps ? undefined : agent.tools;
    const res = await client.chat(messages, tools);
    messages.push({ role: 'assistant', content: res.content, toolCalls: res.toolCalls });

    if (res.toolCalls.length === 0) { output = res.content; break; } // 无工具调用 → 结束
    toolRounds++;
    for (const call of res.toolCalls) {
      const result = await executeToolSafely(agent, call.name, call.args);
      messages.push({ role: 'tool', toolCallId: call.id, content: JSON.stringify(result) });
    }
  }
  yield { type: 'agent-end', agent: agent.name, output, toolRounds };
}
```

拆开看，每一层都很薄：

1. **LLM 生成**：`client.chat(messages, tools)`。`LlmClient` 是库里唯一的外部边界，换模型、换厂商零改动
2. **解析工具调用**：`toolCalls` 为空说明模型认为信息够了，把文本作为最终输出，循环结束
3. **校验 + 执行**：`executeToolSafely` 统一兜底，下文展开
4. **回传**：结果以 `role: 'tool'` 消息追加进 messages，回到第 1 步——这就是全部循环

### 2.1 决策一：错误当结果回传，而不是抛异常

工具调用有三种失败方式：工具名不存在、参数没过校验、`execute` 抛异常。直觉写法是向上抛——但那样一轮 Agent 就整个中断了。在 Agent 语义下，"这一次工具调用失败"只是循环里的一次普通反馈：

```ts
async function executeToolSafely(
  agent: AgentDefinition, name: string, args: JsonObject,
): Promise<JsonObject> {
  const tool = agent.tools?.find((t) => t.name === name);
  if (!tool) return { error: `未知工具：${name}` };
  const validationError = validateArgs(tool.parameters, args);
  if (validationError) return { error: `参数校验失败：${validationError}` };
  try {
    return await tool.execute(args);
  } catch (e) {
    return { error: `工具执行出错：${e instanceof Error ? e.message : String(e)}` };
  }
}
```

模型拿到 `{ error: '未知工具：not_exist' }` 后，下一轮自己决定换工具、改参数，或者干脆放弃工具直接回答。测试里就有这条路径：先喂一个不存在的工具名，再喂一组类型错误的参数，模型最终输出"工具不可用，我直接心算：14"。**错误回传把"失败"降级成了"信息"**——比中断整个循环更符合 Agent 语义。

参数校验本身也是零依赖设计。校验器只覆盖 JSON Schema 最常用的子集——required 存在性加五种基本类型，但足以覆盖真实工具的全部需求：

```ts
export function validateArgs(schema: JsonSchema, args: JsonObject): string | null {
  for (const key of schema.required ?? []) {
    if (!(key in args)) return `缺少必填参数 "${key}"`;
  }
  for (const [key, rule] of Object.entries(schema.properties)) {
    const value = args[key];
    if (value === undefined) continue;
    const actual = Array.isArray(value) ? 'array' : typeof value;
    // ...逐项比较 expected 与 actual，不符则返回错误消息
  }
  return null;
}
```

注意它返回错误消息而不是抛异常——校验失败同样走"回传模型"这条路，和工具执行错误共用同一套机制。

### 2.2 决策二：maxSteps 到达后，收走工具而不是截断

模型可能陷入循环：反复调用同一个工具，或者永远觉得"还差一步"，所以要有 maxSteps（默认 5）。但"到步数直接掐断"会返回空字符串或半截话，用户拿到的是空气。mini-agents 的做法就是循环开头那一行：

```ts
const tools = toolRounds >= maxSteps ? undefined : agent.tools;
```

到达上限后**不再提供工具**，再问模型一次。没有工具可调，模型唯一的路就是把已掌握的信息组织成文本。这保证任何情况下都有文本产出，永不空转。对应的测试断言很直白：`maxSteps: 2` 时第三次 LLM 调用必须不带工具——`expect(client.calls[2].tools).toBeUndefined()`。

### 2.3 决策三：AsyncGenerator 事件流，而不是回调

`runAgent` 是 `AsyncGenerator<AgentEvent>`：每次 LLM 请求、每次工具调用、每次结果回传都是事件（`llm-request` / `tool-call` / `tool-result` / `agent-end` / `agent-error`）。回调的问题是消费方式在写库时就定死了——onEvent？onToolCall？每多一种消费场景就要加一个钩子。事件流把"怎么消费"完全交给调用方：

- `runAgentToEnd` 把事件攒成数组返回，测试和批处理直接用
- CLI / UI 要进度展示，就 `for await` 消费并逐个渲染
- 要接可观测系统，就是把事件翻译成 span

更妙的是编排层的组合：supervisor 里 worker 的事件原样透传到外层——`for (const event of result.events) yield event`。Generator 的组合是天然的，一层包装代码都不用写。

## 3. supervisor：把"路由"做成两个元工具

多智能体编排里，supervisor 要回答的问题是：router 怎么"指挥" worker？常见做法是让 router 输出自由文本（"我选择 translator，任务是……"），再用正则或 JSON 解析去抠。这条路有两个问题：抠出来的东西不可校验；router 完全可能幻觉出一个不存在的工人名，你只能在解析层兜底。

mini-agents 的做法：**路由就是两个注入给 router 的普通工具**。

```ts
const delegate = defineTool({
  name: 'delegate',
  description: '把一项具体任务交给指定的工人执行',
  parameters: {
    type: 'object',
    properties: {
      worker: { type: 'string', description: '工人名称' },
      task: { type: 'string', description: '交给工人的具体任务描述' },
    },
    required: ['worker', 'task'],
  },
  execute: async () => ({}), // 真正的执行在主循环里按名字拦截
});
const finish = defineTool({ /* ...answer 参数，同样空实现 */ });
```

注意 `execute` 是空的——delegate / finish 不在工具层执行，而是 supervisor 主循环里按名字拦截：见到 `delegate` 就用参数里的 worker 名找到工人，`runAgentToEnd` 跑完一遍，把工人输出作为 tool 消息回传 router；见到 `finish` 就跳出循环，把 `answer` 作为最终输出。

这样路由获得了三个白送的性质：

1. **可校验**：worker / task 是工具参数，先过 schema 校验，缺参数根本进不了路由逻辑
2. **幻觉有兜底且语义一致**：工人名单写进 router 的 system prompt（名字 + instructions 第一行前 80 字）；router 报出不存在的工人名时，回传的是 `未知工人：xxx。可用：translator, calculator`——和单 Agent 的错误回传是同一套机制，router 下一轮自己纠正
3. **可回放**：整个委派过程就是事件流里的 tool-call / tool-result，加上 worker 事件的透传，一次完整编排从头到尾每一步都有记录

委派轮数同样有上限（`maxRounds`，默认 6），超限后不再问模型，直接在输出上追加"（已达最大委派轮数，强制结束）"——和单 Agent 的 maxSteps 是同一个思路在不同层级的重复。至于 pipeline（researcher 带工具 → writer 纯生成），本质是一个 for 循环：把前一个 `agent-end` 的 output 作为下一个的 input，事件继续往下 yield，这里不展开。

## 4. 确定性测试：脚本化 Mock LLM

Agent 代码难测，是因为 LLM 是循环里唯一的不确定源。把 LLM 抽象成只有一个 `chat(messages, tools)` 方法的接口后，测试里注入一个脚本化客户端——按顺序吐预设响应，同时记录每次收到的 messages 和 tools：

```ts
function scripted(responses: LlmResponse[]): LlmClient & { calls: ChatMessage[][] } {
  const calls: ChatMessage[][] = [];
  let i = 0;
  return {
    calls,
    async chat(messages, tools) {
      calls.push([...messages]);
      if (i >= responses.length) throw new Error('脚本响应已耗尽');
      return responses[i++];
    },
  };
}
```

一个完整的测试用例：

```ts
it('调用工具后基于结果作答', async () => {
  const client = scripted([
    { content: '', toolCalls: [{ id: 'c1', name: 'calc', args: { a: 6, b: 7 } }], finishReason: 'tool_calls' },
    { content: '6 × 7 = 42', toolCalls: [], finishReason: 'stop' },
  ]);
  const result = await runAgentToEnd({
    agent: { name: 'math', instructions: '计算器助手', tools: [calc] },
    input: '6 乘 7 等于几？', client,
  });
  expect(result.ok).toBe(true);
  expect(result.output).toBe('6 × 7 = 42');
  expect(result.toolRounds).toBe(1);
  // 工具结果以 role:tool 消息回传给模型
  const toolMsg = client.calls[1].find((m) => m.role === 'tool');
  expect(JSON.parse((toolMsg as { content: string }).content).value).toBe(42);
});
```

这个测试断言的不只是"输出对不对"，还有循环的**内部行为**：第二轮发给模型的 messages 里必须有一条 `role: 'tool'`、内容是 JSON 的 42。"脚本响应已耗尽"这条防御是个隐式断言——如果循环意外多发一次 LLM 请求，测试直接红。18 个测试覆盖工具循环的成功 / 未知工具 / 校验失败 / 执行抛错 / maxSteps 强制收尾 / LLM 异常，加上 pipeline 链式传参、supervisor 委派 / 未知工人 / 轮数上限，全部无网络依赖，跑在 GitHub Actions 里。Agent 代码从此有传统意义上的单元测试：给定输入序列，断言执行路径。

## 5. 诚实复盘：局限与 Roadmap

说完好处说局限，三个都是我明确知道、且暂时选择不做的：

- **无流式**：`chat` 返回完整响应，事件流是"步级"粒度不是"字级"。UI 能显示"正在调用 search 工具"，做不了打字机效果。支持流式意味着 `LlmClient` 接口要改成 AsyncIterable，值得单独一次设计，而不是顺手加
- **无并行工具执行**：一轮里的多个 toolCalls 是串行 for 循环。并行不难，难在结果消息与 toolCallId 的配对语义要设计清楚
- **无持久化记忆**：messages 是一次 run 内的局部变量，跨 run 记忆、断点恢复都不存在——那属于存储层，零依赖的库不打算绑死任何存储

Roadmap 按优先级：parallel / DAG 编排模式与循环终止条件 → 事件流录制与确定性回放（调试工具，脚本化 Mock 的反向操作）→ 事件流一键转 OpenTelemetry span → 用它驱动我的 agent-workbench 项目，让应用与框架互相印证。

## 6. 总结

- Agent 循环没有魔法：`while (true)` + `role: 'tool'` 消息回传，500 行写完
- 三个决策值得抄走：错误当结果回传（把失败降级为信息）、maxSteps 收走工具（保证永远有产出）、AsyncGenerator 事件流（消费方式交给调用方）
- supervisor 的 delegate / finish 是注入的普通工具——结构化路由可校验、可回放，幻觉工人会被同一套错误机制纠正
- `LlmClient` 注入 + 脚本化 Mock = Agent 代码的传统单元测试，18 个用例零网络跑 CI
- 全部代码可抄作业：npm 包 [@ordeal/mini-agents](https://www.npmjs.com/package/@ordeal/mini-agents)、GitHub 仓库 [mini-agents](https://github.com/rwzdk/mini-agents)、更多文章见[博客首页](https://github.com/rwzdk/blog)

---


