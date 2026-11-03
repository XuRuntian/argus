# 规范来源

`skill_spec_v1.yaml` 是动作级精简规范，版本为 `argus_subtasks_v1`，不是旧的 skill/phase 文件。

定义提取自 `robo-subtask-alignment` 仓库提交
`ae1329e90a20bae3c46acd684a2825da106ea6b3` 的
`robolabel/config/skill_templates.yaml`：保留全部 29 个动作的含义、开始和结束规则，以及主体枚举；移除模板、slots、示例和内部阶段。运行时不依赖该仓库，也不读取测试集答案。

原文件未定义开始规则的 scoop、shake、strike、wipe_scrub、manipulate、custom，采用“该动作有效运动开始”的缺省规则；none 从无任务相关动作开始。片段开始时已在进行的动作从第 0 帧开始；中断或视频结束时截断动作。区间统一为半开区间。

manipulate 和 custom 的原结束规则依赖人工指定边界；没有提供该边界时用 `action: null`，不自行编造。

主体、物品表示和完整覆盖规则由 `label/subtasks.py` 定义。`object` 是直接操作的物品或交互目标；工具操作取目标物，pour 取所持源容器，交接主体取递出方。复位、底盘移动和空闲的物品为空。上述规则是本适配层明确采用的约定。

默认开放模式下，动作与主体列表是优先词表，不是封闭枚举。明确识别但无匹配定义时允许自由命名；识别不确定仍使用 `action: null` 或 `subject: unknown`。自由命名动作从有效运动开始，到观察到完成、停止或中断时结束，受视频边界限制；不更改已有动作的边界规则。可通过 `prepare --action-vocabulary closed` 将动作限制为规范中的名称或 `null`；主体与物品仍保持原有规则，两种模式使用相同的动作定义。
