"""
防毒效果可视化展示页面

通过交互式图表展示三阶段防护系统对各类投毒攻击的检测效果。
包含：核心指标、阶段分布、类型检出率、隐蔽等级对比、
防护模式对比、领域覆盖、风险分布、详细报告。
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# 添加路径
_project_root = Path(__file__).parent.parent.parent.parent  # pages/ → poisonedrag/ → src/ → PoisonedRAG/
from dotenv import load_dotenv
load_dotenv(_project_root / ".env")

src_dir = _project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from poisonedrag.config import get_config, reset_config, ProtectionMode, apply_protection_mode
from poisonedrag.resecurity import (
    DocumentReviewer,
    RuleChecker,
    create_document_reviewer,
    create_rule_checker,
)
from poisonedrag.llm import get_review_llm
from poisonedrag.security.filter import get_content_filter, ContentFilter
from poisonedrag.security.validator import get_response_validator, ResponseValidator
from poisonedrag.embeddings import EmbeddingModel


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="防毒效果可视化",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.title("🛡️ 防毒效果")
    st.markdown("---")
    st.subheader("页面导航")
    if st.button("💬 对话", use_container_width=True):
        st.switch_page("app.py")
    if st.button("📚 知识库管理", use_container_width=True):
        st.switch_page("pages/knowledge_management.py")
    if st.button("📥 人工审核", use_container_width=True):
        st.switch_page("pages/review.py")
    if st.button("⚙️ 设置", use_container_width=True):
        st.switch_page("pages/settings.py")
    if st.button("🛡️ 防毒效果", use_container_width=True, type="primary"):
        st.switch_page("pages/visualization.py")


# ============================================================
# 工具函数
# ============================================================

def load_all_poison_samples() -> list[dict]:
    """加载所有投毒语料"""
    poison_dir = _project_root / "data" / "poison"
    all_samples = []

    for f in sorted(poison_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
            for item in data:
                if "samples" in item:
                    for s in item["samples"]:
                        s["_category"] = item.get("category", "unknown")
                        all_samples.append(s)
                else:
                    item["_category"] = "unknown"
                    all_samples.append(item)

    return all_samples


def load_normal_samples() -> list[dict]:
    """加载正常知识语料（用于误杀率测试）"""
    kb_dir = _project_root / "data" / "knowledge"
    all_docs = []

    for f in kb_dir.glob("*.json"):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                all_docs.extend(data)
            elif isinstance(data, dict):
                all_docs.append(data)

    return all_docs


def run_batch_with_llm(samples: list, reviewer, rule_checker, content_filter, progress_bar=None, status_text=None) -> pd.DataFrame:
    """完整测试：规则引擎 + 批量 LLM 审查 + 关键词过滤（禁用语义异常检测）"""
    results = []
    total = len(samples)

    # Step 1: 规则引擎预筛（快速）
    for i, sample in enumerate(samples):
        content = sample.get("content", "")
        rule_result = rule_checker.check(content)

        r = {
            "sample_id": sample.get("id", "unknown"),
            "poison_type": sample.get("poison_type", "unknown"),
            "stealth_level": sample.get("stealth_level", "unknown"),
            "target_domain": sample.get("target_domain", sample.get("_category", "unknown")),
            "content_preview": content[:80],
            "blocked": False,
            "block_stage": "none",
            "ingest_risk_score": 0.0,
            "filter_risk_score": 0.0,
            "_rule_blocked": False,
        }

        if rule_result.risk_level == "high":
            r["blocked"] = True
            r["block_stage"] = "ingest (rule)"
            r["ingest_risk_score"] = 0.75
            r["_rule_blocked"] = True
        elif rule_result.risk_level == "medium":
            r["blocked"] = True
            r["block_stage"] = "ingest (rule)"
            r["ingest_risk_score"] = 0.5
            r["_rule_blocked"] = True
        elif rule_result.suspicious_packages or rule_result.has_suspicious_urls:
            r["blocked"] = True
            r["block_stage"] = "ingest (rule)"
            r["ingest_risk_score"] = 0.6
            r["_rule_blocked"] = True

        results.append(r)

    # Step 2: 对未拦截的样本进行批量 LLM 审查
    pending_indices = [i for i, r in enumerate(results) if not r["_rule_blocked"]]
    pending_samples = [samples[i] for i in pending_indices]
    pending_texts = [s.get("content", "") for s in pending_samples]

    if pending_texts and reviewer and reviewer.llm:
        batch_size = reviewer.batch_size if hasattr(reviewer, 'batch_size') else 20

        for batch_start in range(0, len(pending_texts), batch_size):
            batch = pending_texts[batch_start:batch_start + batch_size]
            review_results = reviewer.batch_quick_review(batch, use_parallel=False)

            for j, rev_result in enumerate(review_results):
                idx = pending_indices[batch_start + j]
                results[idx]["ingest_risk_score"] = rev_result.risk_score

                if rev_result.risk_score >= 0.5:
                    results[idx]["blocked"] = True
                    results[idx]["block_stage"] = "ingest (LLM)"

            if progress_bar and status_text:
                done = len(pending_indices) - len(pending_texts) + min(batch_start + batch_size, len(pending_texts))
                progress_bar.progress(done / total)
                status_text.text(f"LLM 审查中 {done}/{total} ...")

    # Step 3: 检索过滤（Stage 2）— 仅对漏过规则+LLM 的样本进行语义过滤
    if content_filter:
        retrieval_indices = [i for i, r in enumerate(results) if not r["blocked"]]
        retrieval_samples = [samples[i] for i in retrieval_indices]

        if retrieval_samples:
            status_text.text(f"正在检索过滤 {len(retrieval_samples)} 条漏过样本...")
            from langchain_core.documents import Document

            for j, sample in enumerate(retrieval_samples):
                idx = retrieval_indices[j]
                content = sample.get("content", "")
                doc = Document(page_content=content, metadata={"source": "poison_test", "type": "poison"})
                filter_result = content_filter.filter_document(doc)
                results[idx]["filter_risk_score"] = filter_result.risk_score

                if not filter_result.is_safe:
                    results[idx]["blocked"] = True
                    results[idx]["block_stage"] = "retrieval"

    # 更新进度
    total_blocked = sum(1 for r in results if r["blocked"])
    if progress_bar and status_text:
        progress_bar.progress(1.0)
        status_text.text(f"三阶段检测完成：已拦截 {total_blocked}/{total}")

    df = pd.DataFrame(results)
    # 清理临时字段
    df = df.drop(columns=["_rule_blocked"], errors="ignore")
    return df


def run_normal_test(normal_docs: list, content_filter) -> dict:
    """测试正常文档的误杀率（旧版，保留兼容）"""
    total = len(normal_docs)
    false_positives = 0

    for doc_data in normal_docs:
        content = doc_data.get("content", "")
        from langchain_core.documents import Document
        doc = Document(page_content=content, metadata={"source": "normal_test", "type": "knowledge"})
        filter_result = content_filter.filter_document(doc)
        if not filter_result.is_safe:
            false_positives += 1

    return {
        "total": total,
        "false_positives": false_positives,
        "false_positive_rate": false_positives / total if total > 0 else 0,
    }


def _run_normal_test_fast(normal_docs: list, rule_checker) -> dict:
    """快速误杀率测试：仅规则引擎（无 Embedding，91 条约 0.05 秒）"""
    total = len(normal_docs)
    false_positives = 0

    for doc_data in normal_docs:
        content = doc_data.get("content", "")
        r = rule_checker.check(content)
        # 规则引擎判为 high/medium 的视为误杀
        if r.risk_level in ("high", "medium") or r.has_suspicious_urls:
            false_positives += 1

    return {
        "total": total,
        "false_positives": false_positives,
        "false_positive_rate": false_positives / total if total > 0 else 0,
    }


def run_protection_mode_comparison(samples: list, rule_checker) -> dict:
    """对比不同防护模式的拦截效果（仅规则引擎，快速）"""
    modes = {}

    for mode in ProtectionMode:
        mode_config = apply_protection_mode(mode)

        ingest_blocks = 0
        retrieval_blocks = 0
        missed = 0

        for sample in samples:
            content = sample.get("content", "")
            blocked_this = False

            # Stage 1: 入库审查（规则引擎）
            if mode_config["enable_ingest_review"]:
                rule_result = rule_checker.check(content)
                if rule_result.risk_level in ("high", "medium") or rule_result.suspicious_packages or rule_result.has_suspicious_urls:
                    ingest_blocks += 1
                    blocked_this = True

            # Stage 2: 检索过滤（简化规则检查）
            if mode_config["enable_retrieval_filter"] and not blocked_this:
                rule_result = rule_checker.check(content)
                if rule_result.risk_level in ("high", "medium") or rule_result.has_suspicious_urls:
                    retrieval_blocks += 1
                    blocked_this = True

            if not blocked_this:
                missed += 1

        modes[mode.value] = {
            "ingest_blocks": ingest_blocks,
            "retrieval_blocks": retrieval_blocks,
            "total_blocked": ingest_blocks + retrieval_blocks,
            "missed": missed,
            "block_rate": (ingest_blocks + retrieval_blocks) / len(samples) * 100 if samples else 0,
        }

    return modes


# ============================================================
# 可视化渲染函数
# ============================================================

def render_metric_cards(df: pd.DataFrame, normal_result: dict):
    """模块 1: 核心指标卡片"""
    total = len(df)
    blocked = len(df[df["blocked"] == True])
    missed = total - blocked
    block_rate = blocked / total * 100 if total > 0 else 0

    ingest_count = len(df[df["block_stage"].str.contains("ingest", na=False)])
    retrieval_count = len(df[df["block_stage"] == "retrieval"])
    missed_count = missed

    fp_rate = normal_result.get("false_positive_rate", 0) * 100
    recall_rate = (1 - fp_rate) if normal_result.get("total", 0) > 0 else 100

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("🎯 拦截率", f"{block_rate:.1f}%", f"拦截 {blocked}/{total}")
    col2.metric("🔍 入库检出", f"{ingest_count}", f"入库阶段拦截")
    col3.metric("🚫 误杀率", f"{fp_rate:.1f}%", f"正常文档误判")
    col4.metric("✅ 正常召回", f"{recall_rate:.1f}%", f"正常文档通过")
    col5.metric("⚡ 检索过滤", f"{retrieval_count}", f"检索阶段拦截")
    col6.metric("📊 测试样本", f"{total}", f"投毒语料总数")


def render_stage_breakdown(df: pd.DataFrame):
    """模块 2: 各阶段拦截分布"""
    stage_counts = {
        "入库审查 (规则)": len(df[df["block_stage"].str.contains("rule", na=False)]),
        "入库审查 (LLM)": len(df[df["block_stage"].str.contains("LLM", na=False)]),
        "检索过滤": len(df[df["block_stage"] == "retrieval"]),
        "漏检": len(df[df["blocked"] == False]),
    }

    fig = go.Figure()

    colors = ["#2ecc71", "#27ae60", "#3498db", "#e74c3c"]
    labels = list(stage_counts.keys())
    values = list(stage_counts.values())
    total = sum(values)

    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v} ({v/total*100:.1f}%)" for v in values],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>数量: %{y}<br>占比: %{text}<extra></extra>",
    ))

    fig.update_layout(
        title="📊 各阶段拦截分布",
        xaxis_title="拦截阶段",
        yaxis_title="样本数",
        height=350,
        margin=dict(l=40, r=40, t=50, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_poison_type_chart(df: pd.DataFrame):
    """模块 3: 各投毒类型检出率"""
    type_stats = df.groupby("poison_type").agg(
        total=("poison_type", "count"),
        blocked=("blocked", "sum"),
    ).reset_index()
    type_stats["detection_rate"] = type_stats["blocked"] / type_stats["total"] * 100
    type_stats = type_stats.sort_values("detection_rate", ascending=True)

    fig = px.bar(
        type_stats,
        x="detection_rate",
        y="poison_type",
        orientation="h",
        color="detection_rate",
        color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
        text=type_stats.apply(lambda r: f"{r['blocked']}/{r['total']} ({r['detection_rate']:.0f}%)", axis=1),
    )

    fig.update_layout(
        title="📊 各投毒类型检出率",
        xaxis_title="检出率 (%)",
        yaxis_title="投毒类型",
        xaxis=dict(range=[0, 105]),
        height=400,
        margin=dict(l=150, r=80, t=50, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_stealth_level_chart(df: pd.DataFrame):
    """模块 4: 隐蔽等级对比"""
    stealth_stats = df.groupby("stealth_level").agg(
        total=("stealth_level", "count"),
        blocked=("blocked", "sum"),
    ).reset_index()
    stealth_stats["detection_rate"] = stealth_stats["blocked"] / stealth_stats["total"] * 100

    # 过滤只保留已知的隐蔽等级
    known_levels = ["low", "medium", "high"]
    stealth_stats = stealth_stats[stealth_stats["stealth_level"].isin(known_levels)]

    if stealth_stats.empty:
        st.info("⚠️ 无有效的隐蔽等级数据")
        return

    labels_map = {"low": "低隐蔽", "medium": "中隐蔽", "high": "高隐蔽"}
    stealth_stats["label"] = stealth_stats["stealth_level"].map(labels_map)
    stealth_stats["sort_key"] = stealth_stats["stealth_level"].map({k: i for i, k in enumerate(known_levels)})
    stealth_stats = stealth_stats.sort_values("sort_key")

    fig = go.Figure()

    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    for i, (_, row) in enumerate(stealth_stats.iterrows()):
        color_idx = int(row["sort_key"])
        fig.add_trace(go.Bar(
            name=row["label"],
            x=[row["label"]],
            y=[row["detection_rate"]],
            marker_color=colors[color_idx],
            text=f"{int(row['blocked'])}/{int(row['total'])} ({row['detection_rate']:.1f}%)",
            textposition="outside",
            hovertemplate=f"<b>{row['label']}</b><br>检出率: {row['detection_rate']:.1f}%<br>{int(row['blocked'])}/{int(row['total'])}<extra></extra>",
        ))

    fig.update_layout(
        title="📊 隐蔽等级对比",
        xaxis_title="隐蔽等级",
        yaxis_title="检出率 (%)",
        yaxis=dict(range=[0, 105]),
        height=350,
        barmode="group",
        margin=dict(l=40, r=40, t=50, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_protection_mode_comparison(samples: list, rule_checker):
    """模块 5: 防护模式对比"""
    modes = run_protection_mode_comparison(samples, rule_checker)

    mode_names = list(modes.keys())
    block_rates = [modes[m]["block_rate"] for m in mode_names]
    missed_counts = [modes[m]["missed"] for m in mode_names]

    labels_map = {
        "strict": "严格模式",
        "standard": "标准模式",
        "performance": "性能模式",
        "development": "开发模式",
        "disabled": "关闭防护",
    }

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="拦截率 (%)",
        x=[labels_map.get(m, m) for m in mode_names],
        y=block_rates,
        marker_color="#2ecc71",
        text=[f"{r:.1f}%" for r in block_rates],
        textposition="outside",
    ))

    fig.update_layout(
        title="📊 防护模式对比",
        xaxis_title="防护模式",
        yaxis_title="拦截率 (%)",
        yaxis=dict(range=[0, 105]),
        height=350,
        margin=dict(l=40, r=40, t=50, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # 详细模式统计
    mode_df = pd.DataFrame([
        {
            "模式": labels_map.get(m, m),
            "拦截数": modes[m]["total_blocked"],
            "入库拦截": modes[m]["ingest_blocks"],
            "检索拦截": modes[m]["retrieval_blocks"],
            "漏检": modes[m]["missed"],
            "拦截率": f"{modes[m]['block_rate']:.1f}%",
        }
        for m in mode_names
    ])
    st.dataframe(mode_df, hide_index=True, use_container_width=True)


def render_domain_coverage_chart(df: pd.DataFrame):
    """模块 6: 领域覆盖检测率（雷达图）"""
    # 过滤掉空值/未知领域
    df_filtered = df[df["target_domain"].notna() & (df["target_domain"] != "unknown") & (df["target_domain"] != "")]

    if df_filtered.empty:
        st.info("⚠️ 无有效的领域数据")
        return

    domain_stats = df_filtered.groupby("target_domain").agg(
        total=("target_domain", "count"),
        blocked=("blocked", "sum"),
    ).reset_index()
    domain_stats["detection_rate"] = domain_stats["blocked"] / domain_stats["total"] * 100

    domain_labels = domain_stats["target_domain"].tolist()
    rates = domain_stats["detection_rate"].tolist()

    # 补充闭合点
    domain_labels.append(domain_labels[0])
    rates.append(rates[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=rates,
        theta=domain_labels,
        fill="toself",
        fillcolor="rgba(46, 204, 113, 0.3)",
        line_color="#2ecc71",
        marker_color="#27ae60",
        name="检出率",
        hovertemplate="<b>%{theta}</b><br>检出率: %{r:.1f}%<extra></extra>",
    ))

    # 添加目标线
    fig.add_trace(go.Scatterpolar(
        r=[80] * (len(rates)),
        theta=domain_labels,
        fill="none",
        line=dict(color="#e74c3c", dash="dash"),
        name="目标线 (80%)",
    ))

    fig.update_layout(
        title="📊 领域覆盖检测率",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
        ),
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_risk_score_distribution(df: pd.DataFrame):
    """模块 7: 风险分数分布"""
    fig = go.Figure()

    blocked_scores = df[df["blocked"] == True]["ingest_risk_score"]
    missed_scores = df[df["blocked"] == False]["ingest_risk_score"]

    fig.add_trace(go.Violin(
        y=blocked_scores,
        name="已拦截",
        box_visible=True,
        meanline_visible=True,
        fillcolor="rgba(46, 204, 113, 0.3)",
        line_color="#2ecc71",
    ))

    fig.add_trace(go.Violin(
        y=missed_scores,
        name="漏检",
        box_visible=True,
        meanline_visible=True,
        fillcolor="rgba(231, 76, 60, 0.3)",
        line_color="#e74c3c",
    ))

    # 添加阈值线
    fig.add_shape(
        type="line",
        x0=-0.5, x1=1.5,
        y0=0.5, y1=0.5,
        line=dict(color="#e74c3c", dash="dash", width=2),
    )
    fig.add_annotation(
        x=0.8, y=0.52,
        text="LLM 阈值 0.5",
        showarrow=False,
        font=dict(color="#e74c3c", size=12),
    )

    fig.update_layout(
        title="📊 风险分数分布（入库审查）",
        yaxis_title="风险分数",
        yaxis=dict(range=[0, 1]),
        height=350,
        margin=dict(l=40, r=40, t=50, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_detail_table(df: pd.DataFrame):
    """模块 8: 详细检测报告"""
    st.subheader("📋 详细检测报告")

    # 筛选器
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        result_filter = st.selectbox(
            "结果筛选",
            options=["全部", "仅已拦截", "仅漏检"],
            key="result_filter",
        )
    with col2:
        type_filter = st.selectbox(
            "投毒类型",
            options=["全部"] + sorted(df["poison_type"].unique().tolist()),
            key="type_filter",
        )
    with col3:
        stealth_filter = st.selectbox(
            "隐蔽等级",
            options=["全部"] + sorted(df["stealth_level"].unique().tolist()),
            key="stealth_filter",
        )
    with col4:
        domain_filter = st.selectbox(
            "目标领域",
            options=["全部"] + sorted(df["target_domain"].unique().tolist()),
            key="domain_filter",
        )

    # 应用筛选
    filtered = df.copy()
    if result_filter == "仅已拦截":
        filtered = filtered[filtered["blocked"] == True]
    elif result_filter == "仅漏检":
        filtered = filtered[filtered["blocked"] == False]
    if type_filter != "全部":
        filtered = filtered[filtered["poison_type"] == type_filter]
    if stealth_filter != "全部":
        filtered = filtered[filtered["stealth_level"] == stealth_filter]
    if domain_filter != "全部":
        filtered = filtered[filtered["target_domain"] == domain_filter]

    # 显示表格
    labels_map = {
        "low": "低",
        "medium": "中",
        "high": "高",
    }

    display_df = filtered[[
        "sample_id", "poison_type", "stealth_level", "target_domain",
        "block_stage", "ingest_risk_score", "blocked", "content_preview"
    ]].copy()

    display_df["结果"] = display_df["blocked"].apply(lambda x: "✅ 已拦截" if x else "❌ 漏检")
    display_df["隐蔽等级"] = display_df["stealth_level"].map(labels_map)
    display_df["风险分数"] = display_df["ingest_risk_score"].apply(lambda x: f"{x:.2f}")

    display_df = display_df.rename(columns={
        "sample_id": "样本ID",
        "poison_type": "投毒类型",
        "target_domain": "目标领域",
        "block_stage": "拦截阶段",
        "content_preview": "内容预览",
    })

    display_df = display_df[[
        "样本ID", "投毒类型", "隐蔽等级", "目标领域",
        "拦截阶段", "风险分数", "结果", "内容预览"
    ]]

    st.dataframe(display_df, hide_index=True, use_container_width=True, height=400)


# ============================================================
# 主函数
# ============================================================

def main():
    st.title("🛡️ PoisonedRAG 防毒效果可视化")
    st.caption("三阶段防护系统（入库审查 → 检索过滤 → 生成校验）对各类投毒攻击的检测效果")

    # 初始化 session state
    if "test_results" not in st.session_state:
        st.session_state.test_results = None
    if "test_running" not in st.session_state:
        st.session_state.test_running = False

    # 操作栏
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        run_button = st.button("▶️ 运行完整测试", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ 清除结果", use_container_width=True)
    with col3:
        export_button = st.button("📥 导出 JSON 报告", use_container_width=True)

    # 清除结果
    if clear_button:
        st.session_state.test_results = None
        st.session_state.test_running = False
        st.rerun()

    # 运行测试
    if run_button:
        st.session_state.test_running = True

        with st.spinner("正在初始化组件..."):
            reset_config()
            config = get_config()

            # 初始化组件
            rule_checker = create_rule_checker()

            # LLM 审查（核心防御，默认启用）
            reviewer = None
            content_filter = None

            try:
                llm = get_review_llm()
                reviewer = __import__("poisonedrag.resecurity.document_reviewer", fromlist=["create_document_reviewer"]).create_document_reviewer(llm=llm.llm)
            except Exception as e:
                st.error(f"❌ LLM 审查初始化失败，无法进行测试: {e}")
                st.session_state.test_running = False
                st.stop()

            # 使用 Ollama 本地 Embedding（快速，不依赖云端 API）
            try:
                emb = EmbeddingModel(provider="ollama")
                content_filter = get_content_filter(embedding_model=emb)
                mode_label = "规则引擎 + LLM 批量审查 + Ollama 本地语义过滤"
            except Exception:
                # Ollama 不可用时仅使用规则引擎 + LLM
                content_filter = None
                mode_label = "规则引擎 + LLM 批量审查（无语义过滤）"

        progress_bar = st.progress(0)
        status_text = st.empty()

        start_time = time.time()

        # 加载投毒语料
        with st.spinner("正在加载投毒语料..."):
            poison_samples = load_all_poison_samples()
            status_text.text(f"已加载 {len(poison_samples)} 条投毒语料")

        # 运行测试：规则预筛 + 批量 LLM 审查
        with st.spinner("正在运行批量测试（规则预筛 + LLM 批量审查）..."):
            results_df = run_batch_with_llm(
                poison_samples, reviewer, rule_checker, content_filter,
                progress_bar=progress_bar,
                status_text=status_text,
            )

        elapsed = time.time() - start_time

        # 正常文档误杀率测试（规则引擎，快速）
        with st.spinner("正在测试正常语料误杀率..."):
            normal_docs = load_normal_samples()
            normal_result = _run_normal_test_fast(normal_docs, rule_checker)

        status_text.success(f"✅ 测试完成！耗时 {elapsed:.1f}s，共测试 {len(poison_samples)} 条投毒语料 + {len(normal_docs)} 条正常语料（{mode_label}）")
        progress_bar.empty()

        # 保存结果
        st.session_state.test_results = {
            "poison_df": results_df,
            "normal_result": normal_result,
            "elapsed": elapsed,
            "total_poison": len(poison_samples),
            "total_normal": len(normal_docs),
            "use_llm_review": True,
            "test_mode": mode_label,
            "timestamp": datetime.now().isoformat(),
        }
        st.session_state.test_running = False
        st.rerun()

    # 显示结果
    if st.session_state.test_results:
        results = st.session_state.test_results
        df = results["poison_df"]

        st.divider()

        # 测试信息
        st.caption(
            f"测试时间: {results['timestamp']} | "
            f"投毒语料: {results['total_poison']} 条 | "
            f"正常语料: {results['total_normal']} 条 | "
            f"耗时: {results['elapsed']:.1f}s | "
            f"测试模式: {results.get('test_mode', '未知')}"
        )

        # 模块 1: 核心指标
        render_metric_cards(df, results["normal_result"])

        st.divider()

        # 模块 2 + 4: 阶段分布 + 隐蔽等级
        col1, col2 = st.columns(2)
        with col1:
            render_stage_breakdown(df)
        with col2:
            render_stealth_level_chart(df)

        st.divider()

        # 模块 3 + 5: 类型检出率 + 防护模式
        col1, col2 = st.columns(2)
        with col1:
            render_poison_type_chart(df)
        with col2:
            # 防护模式对比（快速规则引擎，无需 Embedding）
            rule_checker = create_rule_checker()
            poison_samples = load_all_poison_samples()
            render_protection_mode_comparison(poison_samples, rule_checker)

        st.divider()

        # 模块 6 + 7: 领域覆盖 + 风险分布
        col1, col2 = st.columns(2)
        with col1:
            render_domain_coverage_chart(df)
        with col2:
            render_risk_score_distribution(df)

        st.divider()

        # 模块 8: 详细报告
        render_detail_table(df)

    # 导出报告
    if export_button and st.session_state.test_results:
        results = st.session_state.test_results
        df = results["poison_df"]

        report = {
            "timestamp": results["timestamp"],
            "total_poison": results["total_poison"],
            "total_normal": results["total_normal"],
            "elapsed": results["elapsed"],
            "use_llm_review": results["use_llm_review"],
            "block_rate": f"{len(df[df['blocked']==True]) / len(df) * 100:.1f}%",
            "stage_breakdown": {
                "ingest_rule": int(len(df[df["block_stage"].str.contains("rule", na=False)])),
                "ingest_llm": int(len(df[df["block_stage"].str.contains("LLM", na=False)])),
                "retrieval": int(len(df[df["block_stage"] == "retrieval"])),
                "missed": int(len(df[df["blocked"] == False])),
            },
            "normal_false_positive_rate": f"{results['normal_result']['false_positive_rate']*100:.1f}%",
            "results": df.to_dict(orient="records"),
        }

        output_path = _project_root / "docs" / "防毒效果测试报告.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        st.success(f"✅ 报告已导出到 `docs/防毒效果测试报告.json`")


if __name__ == "__main__":
    main()
