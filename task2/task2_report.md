# Task 2 
问题二主要包含三层任务：

## Subtask 1: Macro-Comparative Analysis of Aggregation Rules
问题背景与目标 (Objective)：
在《与星共舞》(DWTS) 的历史上，主要存在两种截然不同的评分合成规则：
1. 排名制 (Rank-based System, S1-2 & S28+)：将评委排名与粉丝排名相加，数值越小越好。
2. 比例制 (Percent-based System, S3-27)：将评委得分占比与粉丝得票率相加，数值越大越好。
   
本任务的目标是构建一个反事实模拟框架 (Counterfactual Simulation Framework)，在所有历史周次上同时运行这两套规则，并量化地回答一个核心问题：哪种规则更能体现粉丝的意志？即，哪种规则对粉丝投票的“顺从度”更高？

### 统一量纲与数据映射 (Data Standardization)
为了公平比较两种数学性质完全不同的规则，我们需要建立统一的度量衡。
- 输入向量：
  - 评委维度：我们同时拥有评委排名向量 $$\mathbf{R}_J$$ 和评委分数占比向量 $$\mathbf{P}_J$$。
  - 粉丝维度：从 Task 1 中，我们获得了潜在粉丝得票率向量 $$\mathbf{P}_F$$。
- 关键转换：
  - 为了在同一维度上衡量“偏离度”，我们将粉丝得票率 $$\mathbf{P}_F$$ 转换为粉丝排名 $$\mathbf{R}_F$$：
  - 从粉丝投票占比得出粉丝投票排名的转化公式：
  $$\mathbf{R}_{F} = \text{Argsort}(\text{Argsort}(-\mathbf{P}_{F})) + 1$$

### 双轨模拟机制 (Dual-Track Simulation)
对于每一个赛季 $$s$$ 的每一周 $$w$$（参赛人数为 $$n$$），我们构建两个平行宇宙：

宇宙 A：强制排名制 (Scenario A: Rank-based)
在此场景下，无论历史真实规则如何，我们强制使用“名次相加法”：
$$\text{Score}_{A,i} = R_{J,i} + R_{F,i}$$
根据 $$\text{Score}_{A,i}$$ 从小到大排序，得到该规则下的最终模拟排名 $$\text{R}_{final,A}$$
$$\text{R}_{final,A} = \text{Argsort}(\text{Argsort}(\text{Score}_{A,i})) + 1$$
- 机制特性：这是“线性加权”。它抹平了分差的大小，只看相对位置。一个在评委分上落后 10 分的选手，和落后 1 分的选手，在这里可能都只是“落后 1 个名次”
- 
宇宙 B：强制比例制 (Scenario B: Percent-based)
在此场景下，我们强制使用“比例相加法”：
$$\text{Score}_{B,i} = P_{J,i} + S_{F,i}$$
根据 $$\text{Score}_{B,i}$$ 从大到小排序，得到该规则下的最终模拟排名 $$\text{R}_{final,B}$$
$$\text{R}_{final,B} = \text{Argsort}(\text{Argsort}(-\text{Score}_{B,i})) + 1$$(由于原排序顺序有所差别，所以公式有负号的改变)

- 机制特性：这是“幅度敏感”的。如果某位选手粉丝得票率极高（如 Bobby Bones），其 $$\text{S}_{F,i}$$ 的数值优势可以抵消掉巨大的评委劣势。

### 评估指标：粉丝偏离指数 (Fan Deviation Index, FDI)
为了量化规则对粉丝意愿的“背离程度”，我们定义了 粉丝偏离指数 (FDI)。该指标衡量了“最终结果”与“纯粉丝排名”之间的曼哈顿距离。
对于规则 $$k \in \{A, B\}$$，其在第 $$w$$ 周的偏离度定义为：
$$\text{FDI}_k^{(w)} = \frac{1}{N_w} \sum_{i=1}^{N_w} \left| R_{final, k, i} - R_{F, i} \right|$$
- $$R_{final, k, i}$$: 规则 $$k$$下选手 $$i$$的最终排名。
- $$R_{F, i}$$: 仅由粉丝投票决定的选手 $$i$$的排名（即大众的原始意愿）
- $$N_w$$: 当周参赛人数（归一化因子，消除人数变化的影响）
[图片：Distribution of FDI for both rules]
FDI 越小：说明最终结果越接近单纯的粉丝投票结果。即该规则“更听粉丝的话” (Fan-Friendly)
FDI 越大：说明最终结果被评委的分数“扭曲”得越厉害。即该规则“更偏向精英/评委” (Judge-Dominant)
比较逻辑与假设检验 (Hypothesis Testing)
我们在全量历史数据（34个赛季）上计算 $$\text{FDI}_{Rank}$$ 和 $$\text{FDI}_{Percent}$$。
胜负判定：
- 在每一周，我们比较$$\Delta = \text{FDI}_{Rank} - \text{FDI}_{Percent}$$
- 若$$\Delta> 0$$，则 Rank 规则偏离更大，说明 Percent 规则更顺从粉丝。
- 若 $$\Delta< 0$$，则 Percent 规则偏离更大，说明 Rank 规则更顺从粉丝。
[图片：结果统计图表，在大多数周次中Percent规则的FDI更低]
在此我们可以验证以下统计直觉：
- 比例制 (Percent) 通常具有更低的 FDI。因为在比例制中，粉丝投票的方差（Variance）通常大于评委评分的方差（评委习惯给7, 8, 9分，差异小；而粉丝投票可能出现 40% vs 1% 的悬殊差异）。这种方差的不匹配导致粉丝票在加法中占据了主导地位。
- 排名制 (Rank) 它在数学上更强行地实现了“50% vs 50%”的权力分配，从而导致它相对于“纯粉丝意愿”的偏离度更高。
  
## Subtask 2: Mechanism Analysis of Controversial Survivors via Counterfactual Simulation

### 研究目标与背景 (Objective)
在《与星共舞》的历史中，存在一类特殊的“低分幸存者 (Low-Score Survivors)”（如 Bobby Bones, Bristol Palin）。他们长期占据评委评分的底端，却依靠庞大的粉丝基础（Fanbase）屡屡晋级甚至夺冠。这种现象引发了关于赛制公平性的巨大争议。
本任务的目标是构建一个微观反事实模拟器 (Micro-Counterfactual Simulator)，通过重构历史场景，回答一个核心假设性问题：
“如果当年采用了不同的计分规则或引入了评委拯救机制，这些争议选手的命运会被改写吗？”

### 样本选择 (Sample Selection)
为了最大化模型的检测效力，我们选取了 DWTS 历史上粉丝投票与评委评分偏离度最大 (Highest Discrepancy) 的四位代表性选手作为测试样本：
1. Jerry Rice (S2): 排名制时代的典型代表，评委分垫底但晋级决赛。
2. Billy Ray Cyrus (S4): 比例制早期的“流量明星”，依靠粉丝票弥补巨大的评委分差。
3. Bristol Palin (S11,S15): 著名的“保送”案例，多次在评委分最低的情况下幸存。
4. Bobby Bones (S27): 导致赛制改革的直接导火索，以极低的专业分夺冠。
   
- 分别进行四种模拟：
纯Rank规则
Rank+Judges' Save
纯Percent规则
Percent+Judges' Save
得出与实际情况不符的数据条：
暂时无法在飞书文档外展示此内容
- ACTUAL: 真实历史结果。
  - Safe: 晋级。
  - ELIMINATED: 淘汰。
  - 这是我们用来对比的“基准线”。
- Sim_Rank: 纯排名制（S1-2 规则）下的结果。
  - 算法：评委排名 + 粉丝排名。总和最大者淘汰。
- Sim_Rank_Save: 排名制 + 评委拯救机制。
  - 算法：先找倒数两名（Bottom 2），然后评委救其中评委分高的那个。
  - 状态解释：
    - SAVED (Judge): 掉入倒数两名，但被评委救回来了。
    - ELIMINATED: 掉入倒数两名，且评委也没救（或者本来就是倒数第一）。
- Sim_Percent: 纯比例制（S3-27 规则）下的结果。
  - 算法：评委占比 + 粉丝占比。总分最小者淘汰。
- Sim_Percent_Save: 比例制 + 评委拯救机制。
  - 算法：先找总分最低的两人，评委救那个评委分占比高的。
这几列由代码自动生成，用于标记“历史是否被改写”。如果为空，说明模拟结果与历史一致。
- Diff_Sim_Rank
- Diff_Sim_Rank_Save
- Diff_Sim_Percent
- Diff_Sim_Percent_Save
这四列中可能出现的关键词含义如下：
- DANGER (危险/反转)：
  - 含义：真实结果是 Safe（幸存），但在该规则下变成了 ELIMINATED。
  - 论文用法：这是最有力的证据！说明该规则能有效“杀死”低分选手，修正了比赛结果。例如：“在 Rank+Save 规则下，Bristol Palin 会在第 5 周就显示 DANGER，说明她本该在那时回家。”
- SURVIVED (幸存/反转)：
  - 含义：真实结果是 ELIMINATED（淘汰），但在该规则下变成了 Safe。
  - 论文用法：说明该规则保护了这位选手。
- (空值):
  - 含义：模拟结果与真实结果一致，没有发生改变。
  
### 模拟原理：四个平行宇宙 (Simulation Logic: Four Parallel Universes)
对于每一位目标选手 $$c$$，在其参赛的每一周 $$t$$，我们保持当周所有选手的表现（评委分 $$\mathbf{S}_J$$）和人气（粉丝得票率 $$\mathbf{V}_F$$）不变，仅改变聚合规则 (Aggregation Rule)，构建四个平行宇宙：

宇宙 I：纯排名制 (Rank-based)
- 规则：$$Score = Rank(S_J) + Rank(V_F)$$。
- 淘汰判定：总排名数值最大（即最后一名）的选手被淘汰
- 假设：测试“线性加权”是否能抑制高人气选手
  
宇宙 II：排名制 + 评委拯救 (Rank + Judges' Save)
- 规则：
  1. 计算总排名，确定 倒数两名 (Bottom Two) 集合 $$B = \{c_1, c_2\}$$。
  2. 评委裁决 (Veto)：比较 $$c_1, c_2$$的评委排名。
  3. 若 $$Rank_J(c_{target}) < Rank_J(c_{opponent})$$，则目标选手获救；否则淘汰
- 假设：测试“底线防御机制”能否在最后关头拦截低分选手
  
宇宙 III：纯比例制 (Percent-based)
- 规则：$$Score = \%S_J + \%V_F$$。
- 淘汰判定：总得分最小的选手被淘汰。
- 假设：这是 Bobby Bones 等人实际获胜的规则，作为对照组（Control Group）
  
宇宙 IV：比例制 + 评委拯救 (Percent + Judges' Save)
- 规则：
  1. 计算总得分，确定得分最低的 Bottom Two。
  2. 评委裁决：比较两者的评委得分占比。若目标选手占比更高，则获救
   
### 历史反转检测算法 (Reversal Detection Algorithm)
为了量化规则改变的影响，我们定义了“反转状态 (Reversal Status)”
对于每一周$$t$$，令 $$R_{actual}$$ 为历史真实结果（Safe/Eliminated），$$R_{sim}$$ 为模拟结果。
危险信号 (DANGER / False Negative):$$R_{actual} = \text{Safe} \quad \land \quad R_{sim} = \text{Eliminated}$$
- 含义：该选手在历史上幸存了，但在新规则下应当被淘汰。
- 推论：证明了该新规则（如评委拯救）能有效修正“德不配位”的现象。
  
### 算法流程 (Process Workflow)
1. 数据注入：加载 Task 1 生成的 final_est_share (粉丝得票率) 和原始 judge_score。
2. 场景重构：针对目标选手$$C$$所在的每一周，提取当周所有 $$N$$名对手的数据。
3. 规则运算：分别代入上述四个公式，计算该周的模拟淘汰者集合 $$\mathcal{E}_{sim}$$
4. 判定与记录：
  - 若 $$C \in \mathcal{E}_{sim}$$ 且 $$C$$历史上未被淘汰，标记为 DANGER。
  - 若 $$C$$ 进入 Bottom 2 但通过评委分优势胜出，标记为 SAVED (Judge)。
5. 输出分析：生成反转时间轴，统计不同规则对该选手“生存寿命”的缩减程度。
   
### 可视化结果
[图片 * 5: 争议人物在不同规则下的命运时间轴]

## Task 2 Subtask3: Evaluation and Policy Recommendation Model
### 模型背景与目标 (Objective)
在确定了粉丝投票的潜在分布（Task 1）并模拟了历史上的反事实结果（Task 2 Subtask 1 & 2）后，分任务三的核心目标是建立一个多维量化评估体系，用以回答以下两个核心策略问题：
1. 基础赛制选择：在“排名制 (Rank)”与“比例制 (Percent)”之间，哪种机制更能平衡专业性与娱乐性？
2. 修正机制评估：引入“评委拯救机制 (Judges' Save)”是否利大于弊？
   
我们不仅关注单一指标，而是将问题视为一个多目标优化问题 (Multi-Objective Optimization)，在“公平性（Fairness）”与“粉丝满意度（Fan Satisfaction）”之间寻找帕累托最优解。

### 核心量化指标 (Quantitative Metrics)
为了客观评价每种规则的表现，我们定义了以下数学指标：

公平性指数 (Fairness Index, $$I_{fair}$$)
衡量最终排名对专业评委意见的尊重程度。
- 定义：最终模拟排名 ($$R_{final}$$) 与评委排名 ($$R_{judge}$$) 的 斯皮尔曼等级相关系数 (Spearman's Rank Correlation)。
- 公式：$$I_{fair} = \rho(R_{final}, R_{judge}) = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$
- 其中 $$d_i$$是两组排名的差值，$$n$$ 是当周参赛人数。
- 意义：$$I_{fair}$$越接近 1，表示赛制越专业、越公平。
  
粉丝满意度指数 (Fan Satisfaction Index, $$I_{fan}$$)
衡量最终排名对大众投票意愿的反映程度。
- 定义：最终模拟排名 ($$R_{final}$$) 与粉丝投票排名 ($$R_{fan}$$) 的斯皮尔曼等级相关系数。
- 公式：$$I_{fan} = \rho(R_{final}, R_{fan})$$
- 意义：$$I_{fan}$$越接近 1，表示赛制越能让观众开心，商业价值可能越高。
  
极值风险率 (Extreme Risk Rate, $$R_{risk}$$)
衡量赛制发生“灾难性误判”的概率。我们定义两种风险：
- 专业崩塌风险：评委心中的第 1 名被淘汰的概率。
- 人气崩塌风险：观众心中的第 1 名被淘汰的概率。
- 公式：$$R_{risk} = \frac{1}{T} \sum_{t=1}^{T} \mathbb{I}(\text{Top1 Candidate is Eliminated in week } t)$$
  
### 高级评估方法 (Advanced Evaluation Methods)
单纯比较平均分是不够的，我们引入了灵敏度分析来模拟不同的决策偏好。

动态权重评分 (Sensitivity Analysis):
考虑到不同利益相关者（节目组、赞助商、专业协会）对节目定位的理解不同，我们引入粉丝权重参数$$\alpha$$ ($$0 \le \alpha \le 1$$)。
- 综合得分函数：
- $$Score(\alpha) = (1 - \alpha) \cdot I_{fair} + \alpha \cdot I_{fan}$$分析过程：
- 让 $$\alpha$$ 从 0.1 遍历到 0.9。
  - 精英主义视角 ($$\alpha < 0.3$$)：重视专业性，看哪种规则胜出。
  - 平衡视角 ($$0.4 \le \alpha \le 0.6$$)：寻找兼顾双方的最佳规则。
  - 民粹主义视角 ($$\alpha > 0.7$$)：重视收视率，看哪种规则胜出。
  
### 结论 (Expected Outcome)
1. Rank vs Percent：Rank 规则更优。因为它将评委分和粉丝票归一化到了同一量纲（名次），避免了Percent规则中因一方分差过大（如粉丝投票极度集中）而完全主导比赛的情况。
]
[图片：数据和图表形式的展示 Sensitivity Analysis: Composite Score vs Fan Weight]

2. Judges' Save：非常有必要
显然Judges' Save的效果显然不会比Base的效果差（因为Save包含了评委无为而治的结果）
[图片：Extreme Risk Analysis: Probability of Eliminating Top Contestants under Different Rules]
- 对于Rank-Based:
  可以有效遏制观众投票数较多但是评委得分低的选手一路进入决赛
- 对于Percent-Based:
可以防止评委得分高的选手收到观众的针对而被淘汰
