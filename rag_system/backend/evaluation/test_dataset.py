"""
test_dataset.py — 标注测试数据集
================================================================================
技术决策记录:
- 测试集是 RAG 评估的基础。没有标注数据，评估就无法量化。
- 30 条测试用例，覆盖简单/中等/困难三种复杂度。
- 测试用例需要覆盖不同的文档类型和查询模式。
"""

from __future__ import annotations

from typing import Literal


def get_test_dataset() -> list[dict]:
    """
    获取 RAG 系统测试数据集

    技术决策:
    - 测试用例需要覆盖不同的查询类型（事实型/推理型/对比型/综合型）
    - 每条用例包含 question、ground_truth、category（复杂度分类）
    - category 用于分层评估：简单查询的阈值可以更严格
    """

    return [
        # ========== Simple (简单事实型) ==========
        {
            "question": "供应商 A-2024-001 的付款条款是什么？",
            "ground_truth": "付款条款为预付30%，交货后60天内支付余款70%",
            "category": "simple",
            "query_type": "fact_lookup",
            "doc_reference": "供应商合同模板 v3.2",
        },
        {
            "question": "产品质量检测报告的最新版本号是多少？",
            "ground_truth": "最新版本号为 QAR-2024-V5.3，发布日期为2024年3月15日",
            "category": "simple",
            "query_type": "fact_lookup",
            "doc_reference": "质量检测标准文档",
        },
        {
            "question": "公司年假政策中关于工龄满5年的规定是什么？",
            "ground_truth": "工龄满5年的员工享有15天带薪年假",
            "category": "simple",
            "query_type": "fact_lookup",
            "doc_reference": "员工手册 2024版",
        },
        {
            "question": "X产品的建议零售价是多少？",
            "ground_truth": "X产品的建议零售价为人民币2999元",
            "category": "simple",
            "query_type": "fact_lookup",
            "doc_reference": "产品定价表 2024Q2",
        },
        {
            "question": "项目 Y 的预计交付日期是什么时候？",
            "ground_truth": "项目Y的预计交付日期为2024年9月30日",
            "category": "simple",
            "query_type": "fact_lookup",
            "doc_reference": "项目计划书 Y",
        },
        {
            "question": "数据安全合规负责人是谁？",
            "ground_truth": "数据安全合规负责人为张伟，联系方式为 zhangwei@company.com",
            "category": "simple",
            "query_type": "fact_lookup",
            "doc_reference": "组织架构与职责说明",
        },
        {
            "question": "合同模板中的违约金比例是多少？",
            "ground_truth": "违约金比例为合同总金额的20%",
            "category": "simple",
            "query_type": "fact_lookup",
            "doc_reference": "标准合同模板",
        },
        {
            "question": "本次系统升级的维护窗口是什么时间段？",
            "ground_truth": "维护窗口为每周日凌晨2:00至6:00（北京时间）",
            "category": "simple",
            "query_type": "fact_lookup",
            "doc_reference": "IT运维手册",
        },
        {
            "question": "公司报销流程的审批人是哪个部门？",
            "ground_truth": "报销金额在5000元以下由部门经理审批，超过5000元需财务总监审批",
            "category": "simple",
            "query_type": "fact_lookup",
            "doc_reference": "财务报销制度 v2.1",
        },
        {
            "question": "会议室 C 的最大容纳人数是多少？",
            "ground_truth": "会议室C的最大容纳人数为12人",
            "category": "simple",
            "query_type": "fact_lookup",
            "doc_reference": "办公资源使用指南",
        },

        # ========== Moderate (中等复杂度) ==========
        {
            "question": "供应商A和供应商B的交付能力有何差异？",
            "ground_truth": "供应商A平均交付周期为7天准时率95%；供应商B交付周期15天准时率88%，A在交付速度和可靠性上优于B",
            "category": "moderate",
            "query_type": "comparison",
            "doc_reference": "供应商评估报告 2024",
        },
        {
            "question": "A政策和B政策在数据保护条款上有何异同？",
            "ground_truth": "两者都要求数据加密存储，但A政策额外要求每季度安全审计，B政策仅要求年度审计",
            "category": "moderate",
            "query_type": "comparison",
            "doc_reference": "A政策文档 / B政策文档",
        },
        {
            "question": "针对X产品的质量投诉，历史上有哪些主要原因？",
            "ground_truth": "主要原因包括：外观瑕疵（占比35%）、功能异常（占比28%）、包装破损（占比22%）、说明文档缺失（占比15%）",
            "category": "moderate",
            "query_type": "aggregation",
            "doc_reference": "客户服务质量报告 2024H1",
        },
        {
            "question": "请分析本季度客户投诉的整体趋势和主要问题",
            "ground_truth": "本季度投诉量环比下降12%，但关于交付延迟的投诉上升8%，主要发生在华东地区",
            "category": "moderate",
            "query_type": "analysis",
            "doc_reference": "客户反馈季度报告",
        },
        {
            "question": "哪些员工在去年完成了超过40小时的合规培训？",
            "ground_truth": "完成超过40小时合规培训的包括：销售部全体15人、研发部8人、市场部5人，共28人",
            "category": "moderate",
            "query_type": "filter_aggregation",
            "doc_reference": "培训记录 2024",
        },
        {
            "question": "对于合同金额超过100万的采购，需要哪些审批层级？",
            "ground_truth": "100万以上采购需经：部门总监→采购总监→CFO→CEO 四级审批",
            "category": "moderate",
            "query_type": "procedure",
            "doc_reference": "采购管理制度 v4.0",
        },
        {
            "question": "过去6个月里，哪类产品的退货率最高？",
            "ground_truth": "电子类产品退货率最高，为8.3%，其次是家居类4.2%，服装类2.1%",
            "category": "moderate",
            "query_type": "aggregation",
            "doc_reference": "售后数据报告 2024",
        },
        {
            "question": "公司目前在哪些地区设有分支机构？各机构的业务重点是什么？",
            "ground_truth": "公司在华北（北京）、华东（上海）、华南（广州）、西南（成都）设有4个分支机构，北京侧重研发，上海侧重金融客户，广州侧重制造业，成都侧重政府项目",
            "category": "moderate",
            "query_type": "enumeration",
            "doc_reference": "公司组织架构说明",
        },
        {
            "question": "2024年的营收目标是多少？上半年完成情况如何？",
            "ground_truth": "2024年营收目标为5亿元，上半年完成2.1亿元，完成率42%，略低于时间进度预期",
            "category": "moderate",
            "query_type": "fact_comparison",
            "doc_reference": "年度经营计划 / 中期报告",
        },
        {
            "question": "哪些项目在近两个季度出现了进度延期？原因分别是什么？",
            "ground_truth": "项目X延期3周，原因包括需求变更和资源冲突；项目Y延期2周，原因主要是第三方依赖延迟",
            "category": "moderate",
            "query_type": "filter_analysis",
            "doc_reference": "项目进度报告",
        },

        # ========== Difficult (复杂多跳/推理) ==========
        {
            "question": "如果供应商X在下季度断供，哪些客户的产品交付会受到直接影响？",
            "ground_truth": "供应商X主要供应A客户（占比60%订单）和B客户（占比25%订单）的核心组件，X断供将直接影响A客户Q3全部订单（约5000台）和B客户60%订单（约1500台），预计影响交付约2周",
            "category": "difficult",
            "query_type": "multi_hop",
            "doc_reference": "供应商清单 / 客户订单 / 物料清单",
        },
        {
            "question": "请分析过去一年客户流失的主要原因，并按影响程度排序",
            "ground_truth": "流失原因按影响程度：1)价格因素（38%）—竞品价格低20%；2)服务质量（27%）—响应速度慢；3)产品功能不足（22%）—缺少移动端；4)其他（13%）",
            "category": "difficult",
            "query_type": "multi_hop_analysis",
            "doc_reference": "客户流失分析报告 / 客服工单 / 竞品对比",
        },
        {
            "question": "哪些员工既完成了合规培训又参与了绩效考核，且绩效评级为A？",
            "ground_truth": "同时满足三个条件（合规培训>40h + 参与绩效评估 + 评级A）的员工：研发部李明、王芳，销售部张强，共3人",
            "category": "difficult",
            "query_type": "multi_filter",
            "doc_reference": "培训记录 / 绩效评估 / 人事档案",
        },
        {
            "question": "公司目前面临的五大风险是什么？哪些已有缓解措施？",
            "ground_truth": "五大风险：1)供应链风险（已有备选供应商）2)人才流失风险（已有股权激励）3)合规风险（已有季度审计）4)网络安全风险（已有零信任架构）5)市场竞争风险（已在评估差异化策略）",
            "category": "difficult",
            "query_type": "synthesis",
            "doc_reference": "风险管理报告 / 相关政策文档",
        },
        {
            "question": "请分析产品X在国内外市场的表现差异及其背后原因",
            "ground_truth": "国内市场：占有率15%，增速8%，原因：口碑好、渠道广；国际市场：占有率3%，增速25%，原因：价格竞争力强但品牌认知度低、渠道建设不足",
            "category": "difficult",
            "query_type": "comparison_analysis",
            "doc_reference": "市场分析报告 / 销售数据 / 竞品分析",
        },
        {
            "question": "本季度跨部门协作中存在哪些主要瓶颈？如何改进？",
            "ground_truth": "主要瓶颈：1)需求评审周期长（平均延误5天），改进：建立需求优先级矩阵；2)测试资源不足（影响3个项目），改进：引入外包测试团队；3)信息共享不畅，改进：搭建跨部门知识库",
            "category": "difficult",
            "query_type": "diagnosis_recommendation",
            "doc_reference": "运营分析报告 / 部门反馈 / 项目复盘",
        },
        {
            "question": "哪些法规变化会对公司明年的业务产生重大影响？",
            "ground_truth": "1)《数据安全法》修订版：要求用户数据本地化存储，需迁移海外数据中心，预算200万；2)环保法规收紧：高排放产品需加装过滤装置，成本增加15%；3)劳动法新规：灵活就业人员比例受限",
            "category": "difficult",
            "query_type": "reasoning_impact",
            "doc_reference": "法规跟踪报告 / 合规评估 / 业务影响分析",
        },
        {
            "question": "请对比分析A方案和B方案的优劣，并给出推荐建议",
            "ground_truth": "A方案：投资低（500万）、周期短（6个月）、风险小，但扩展性有限；B方案：投资高（1200万）、周期长（18个月）、扩展性强。A方案适合当前规模，B方案适合3年后预期规模增长3倍的场景，建议分阶段实施",
            "category": "difficult",
            "query_type": "comparison_reasoning",
            "doc_reference": "方案可行性报告A / 方案可行性报告B",
        },
        {
            "question": "过去两年里，哪些因素对客户满意度的影响最为显著？",
            "ground_truth": "相关性分析显示：1)交付准时性（相关系数0.78）— 影响最大；2)售后响应速度（0.65）；3)产品质量（0.58）；4)价格竞争力（0.42）。交付准时性每提升10%，满意度平均提升8%",
            "category": "difficult",
            "query_type": "multi_factor_analysis",
            "doc_reference": "客户满意度调研 / 运营数据 / 历史分析报告",
        },
        {
            "question": "综合分析公司人才结构，识别关键岗位继任风险",
            "ground_truth": "高风险岗位：CTO（无明确继任者）、核心架构师（仅1人掌握全部技术栈）、大客户总监（2人竞争）。建议：建立技术委员会、实施AB角制度、加强高潜人才培养",
            "category": "difficult",
            "query_type": "synthesis_diagnosis",
            "doc_reference": "组织架构 / 人才盘点报告 / 岗位职责说明",
        },
    ]


def get_test_dataset_by_category(
    category: Literal["simple", "moderate", "difficult"],
) -> list[dict]:
    """按复杂度分类获取测试数据集"""
    return [tc for tc in get_test_dataset() if tc["category"] == category]


def get_test_dataset_summary() -> dict:
    """获取测试数据集统计摘要"""
    dataset = get_test_dataset()
    summary = {
        "total": len(dataset),
        "by_category": {},
        "by_query_type": {},
    }

    for tc in dataset:
        cat = tc["category"]
        qtype = tc["query_type"]
        summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1
        summary["by_query_type"][qtype] = summary["by_query_type"].get(qtype, 0) + 1

    return summary
