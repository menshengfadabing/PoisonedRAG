"""
测试报告生成模块

生成 HTML 格式的测试报告。
"""

from datetime import datetime
from typing import Dict, Any, List
import json


def generate_html_report(metrics_data: Dict[str, Any], output_path: str):
    """生成 HTML 测试报告"""
    html_template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>防护测试报告 - PoisonedRAG</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .header h1 {
            margin: 0;
            font-size: 28px;
        }
        .header .timestamp {
            opacity: 0.8;
            margin-top: 10px;
        }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .card h3 {
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
        }
        .card .value {
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }
        .card.success .value {
            color: #10b981;
        }
        .card.warning .value {
            color: #f59e0b;
        }
        .card.danger .value {
            color: #ef4444;
        }
        .section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .section h2 {
            margin: 0 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #eee;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .progress-bar {
            background: #e5e7eb;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
        }
        .progress-bar .fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        .progress-bar .fill.green {
            background: #10b981;
        }
        .progress-bar .fill.yellow {
            background: #f59e0b;
        }
        .progress-bar .fill.red {
            background: #ef4444;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ PoisonedRAG 防护测试报告</h1>
        <div class="timestamp">生成时间: {timestamp}</div>
    </div>

    <div class="summary">
        <div class="card">
            <h3>总测试样本</h3>
            <div class="value">{total_samples}</div>
        </div>
        <div class="card success">
            <h3>成功检测</h3>
            <div class="value">{detected_samples}</div>
        </div>
        <div class="card danger">
            <h3>攻击成功</h3>
            <div class="value">{attack_success}</div>
        </div>
        <div class="card warning">
            <h3>检测率</h3>
            <div class="value">{detection_rate}</div>
        </div>
    </div>

    <div class="section">
        <h2>各阶段拦截统计</h2>
        <table>
            <tr>
                <th>阶段</th>
                <th>拦截数量</th>
                <th>占比</th>
            </tr>
            {stage_rows}
        </table>
    </div>

    <div class="section">
        <h2>按隐蔽等级统计</h2>
        <table>
            <tr>
                <th>隐蔽等级</th>
                <th>检测数</th>
                <th>总数</th>
                <th>检测率</th>
                <th>进度</th>
            </tr>
            {stealth_rows}
        </table>
    </div>

    <div class="section">
        <h2>按投毒类型统计</h2>
        <table>
            <tr>
                <th>投毒类型</th>
                <th>检测数</th>
                <th>总数</th>
                <th>检测率</th>
            </tr>
            {poison_type_rows}
        </table>
    </div>
</body>
</html>
"""

    # 生成阶段表格行
    stage_names = {
        "ingest": "入库阶段",
        "retrieval": "检索阶段",
        "generation": "生成阶段"
    }
    stage_breakdown = metrics_data.get("stage_breakdown", {})
    total_detected = metrics_data.get("detected_samples", 0)

    stage_rows = []
    for stage, count in stage_breakdown.items():
        name = stage_names.get(stage, stage)
        percent = (count / total_detected * 100) if total_detected > 0 else 0
        stage_rows.append(f"""
            <tr>
                <td>{name}</td>
                <td>{count}</td>
                <td>{percent:.1f}%</td>
            </tr>
        """)

    # 生成隐蔽等级表格行
    stealth_rows = []
    stealth_stats = metrics_data.get("stealth_level_stats", {})
    for level, stats in stealth_stats.items():
        detected = stats.get("detected", 0)
        missed = stats.get("missed", 0)
        total = detected + missed
        rate = (detected / total * 100) if total > 0 else 0

        color = "green" if rate >= 80 else ("yellow" if rate >= 50 else "red")

        stealth_rows.append(f"""
            <tr>
                <td>{level}</td>
                <td>{detected}</td>
                <td>{total}</td>
                <td>{rate:.1f}%</td>
                <td>
                    <div class="progress-bar">
                        <div class="fill {color}" style="width: {rate}%"></div>
                    </div>
                </td>
            </tr>
        """)

    # 生成投毒类型表格行
    poison_type_rows = []
    poison_stats = metrics_data.get("poison_type_stats", {})
    for ptype, stats in poison_stats.items():
        detected = stats.get("detected", 0)
        missed = stats.get("missed", 0)
        total = detected + missed
        rate = (detected / total * 100) if total > 0 else 0

        poison_type_rows.append(f"""
            <tr>
                <td>{ptype}</td>
                <td>{detected}</td>
                <td>{total}</td>
                <td>{rate:.1f}%</td>
            </tr>
        """)

    # 填充模板
    html = html_template.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_samples=metrics_data.get("total_samples", 0),
        detected_samples=metrics_data.get("detected_samples", 0),
        attack_success=metrics_data.get("attack_success", 0),
        detection_rate=metrics_data.get("detection_rate", "0.00%"),
        stage_rows="".join(stage_rows),
        stealth_rows="".join(stealth_rows),
        poison_type_rows="".join(poison_type_rows),
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"报告已生成: {output_path}")


def save_metrics_json(metrics_data: Dict[str, Any], output_path: str):
    """保存 JSON 格式的测试数据"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    print(f"数据已保存: {output_path}")