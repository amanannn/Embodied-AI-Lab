import json
import os
import re
from pathlib import Path

# 假设你使用 ollama 或兼容 OpenAI 接口的本地服务
# 请根据实际情况修改 BASE_URL 和 MODEL_NAME
try:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    MODEL_NAME = "qwen3:32b"
except ImportError:
    print("❌ 缺少 openai 库，请运行: pip install openai")
    exit(1)

PROMPT_DIR = Path("./prompts")

# 8个核心Prompt的测试用例配置
TEST_CASES = {
    "ingest-single.md": {
        "input": {"content": "Qwen3支持128K上下文窗口", "metadata": {"source": "tech_blog", "timestamp": "2026-07-06T10:00:00Z"}},
        "required_keys": ["title", "summary", "tags", "category"] 
    },
    "memory-recall.md": {
        "input": {"user_question": "上次部署是什么时候？", "search_scope": "logs/"},
        "required_keys": ["search_query", "filters", "max_results"]
    },
    "system-agent.md": {
        "input": {"user_request": "记录今日会议纪要", "context_boundaries": ["无"]},
        "required_keys": ["decision", "rationale", "suggested_alternatives"]
    },
    "index-update.md": {
        "input": {"action": "add", "target_path": "docs/tech/qwen3.md", "content_hash": "a1b2c3d4e5f6"},
        "required_keys": ["index_command", "affected_index", "requires_rebuild", "confidence"]
    },
    "ingest-batch.md": {
        "input": {"items": [{"content": "API文档已更新至v2.0", "metadata": {"source": "pr_merge", "timestamp": "2026-07-06T16:00:00Z"}}]},
        "required_keys": ["decisions", "summary"]
    },
    "lint-knowledge.md": {
        "input": {"content": "# 数据库设计\n\n- 使用PostgreSQL 14+\n", "file_path": "wiki/db_design.md"},
        "required_keys": ["is_valid", "errors", "warnings", "suggestions"]
    },
    "memory-log.md": {
        "input": {"raw_log": "[ERROR] API timeout after 30s (service=auth)", "log_type": "system", "timestamp": "2026-07-06T15:00:00Z"},
        "required_keys": ["standardized_log", "category", "extracted_entities", "should_persist"]
    },
    "query-knowledge.md": {
        "input": {"user_query": "如何配置Docker网络？", "context": "运维讨论"},
        "required_keys": ["query_type", "keywords", "required_fields", "time_range", "confidence"]
    }
}

def extract_system_prompt(file_path: Path) -> str:
    """从Markdown文件中提取System Prompt（跳过YAML Frontmatter）"""
    content = file_path.read_text(encoding="utf-8")
    # 移除 YAML frontmatter
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return content.strip()

def test_prompt(filename: str, config: dict) -> bool:
    """执行单个Prompt测试"""
    file_path = PROMPT_DIR / filename
    if not file_path.exists():
        print(f"❌ {filename}: 文件不存在")
        return False

    system_prompt = extract_system_prompt(file_path)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(config["input"], ensure_ascii=False)}
            ],
            temperature=0.15, 
            response_format={"type": "json_object"}
        )
        
        raw_output = response.choices[0].message.content
        
        # 尝试解析JSON
        result = json.loads(raw_output)
        
        # 校验必需字段
        missing_keys = [k for k in config["required_keys"] if k not in result]
        if missing_keys:
            print(f"❌ {filename}: 缺少字段 {missing_keys}")
            print(f"   模型输出: {raw_output[:200]}...")
            return False
            
        print(f"✅ {filename}: 通过!")
        return True
        
    except json.JSONDecodeError:
        print(f"❌ {filename}: JSON解析失败")
        print(f"   原始输出: {raw_output[:300]}...")
        return False
    except Exception as e:
        print(f"❌ {filename}: 运行时错误 - {str(e)}")
        return False

if __name__ == "__main__":
    print(f"🧪 开始测试模型: {MODEL_NAME}\n")
    
    passed = 0
    total = len(TEST_CASES)
    
    for filename, config in TEST_CASES.items():
        print(f"📋 {filename}")
        if test_prompt(filename, config):
            passed += 1
        print("-" * 50)
        
    print(f"\n📊 最终结果: {passed}/{total} 通过")
    if passed == total:
        print("🎉 所有Prompt地基验证完成！可以进入后端编排层开发。")
    else:
        print("⚠️ 存在未通过的Prompt，请检查对应.md文件或调整测试输入。")