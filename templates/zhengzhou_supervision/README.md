# 郑州独立监督模板

这些 CSV 只有表头，没有观测、评级或训练标签。`prepare_evidence_packet.py` 会把它们复制到本地证据目录；填写真实证据后仍需独立性审计和目标定义。

| 模板 | 每行单位 | 填写要求 |
|---|---|---|
| `survey_protocol_review.csv` | 一个原始调查事件的协议核查 | 核实完整目标清单、有效时长、人数、路线/面积、坐标系和来源；报告时间跨度不直接作为时长 |
| `protection_assessments.csv` | 一个单元×季节年份×真实评价者的评价 | 实测响应与管理优先等级分别存；证据、真实评价者、日期和是否盲评必须可追溯 |
| `restoration_feasibility.csv` | 地块×具体措施的可行性核查 | 必要项用 `yes/no/unknown`；未知不能写成 no。总体结论及阻碍需引用证据 |
| `restoration_assessments.csv` | 地块×措施×真实评价者/实测响应 | `target_type` 区分 `expected_priority` 与 `measured_response`；预期评分不能称实际收益 |
| `restoration_projects.csv` | 项目中的一个处理或对照地块 | 实施日期、措施、面积、几何与匹配组有真实来源 |
| `restoration_monitoring_events.csv` | 一次标准化监测访问 | 即使零水鸟，也保留访问行；只有完整调查支持时才确认零水鸟 |
| `restoration_species_records.csv` | 调查事件中的一个物种记录 | 通过 `event_id` 链接访问；保留真实数量、检测方法与来源 |
| `label_provenance.csv` | 一个样本×监督任务的来源审计 | 检查标签来源以及底模 fit/tune/calibrate 重叠，记录空间组、用途、锁定测试与阻碍 |

## 字段约定

- `event_id/site_id/plot_id` 经真实事件和空间单元核查后确定；候选哈希不能未经审定变成最终调查 ID。
- `season_year`：12 月冬季归下一年；同时保存原始日期。空白为缺证据，真实零值单独保留。
- `complete_target_checklist/blinded_to_model_outputs` 用 `yes/no/unknown`；数值努力量缺失保持空白。
- `priority_grade_0_4` 是序数管理等级，填写前统一量表并登记判据；没有记录不能评 0。AI 草案不代表真实评级。
- `target_protection/target_restoration` 留空，直到响应、尺度、损失函数和来源审计完成。0–4 除以 4 不自动成为概率，负修复收益也不能裁剪成 0。
- `eligible_for_training` 仅在核查通过后填 true；unknown、未核实或泄漏样本保持 false。
- `derived_from_m_or_q` 必须如实登记；由模型输出生成的目标不能作为独立监督。
- 一份来源表可含多种协议，源级固定时长不能自动广播到所有事件。保存评价者的原始评级与分歧，不覆盖为一个未经说明的共识分。

方法与量表草案见[独立监督实施方案](../../docs/郑州市试点_LST与独立监督方案.md)。
