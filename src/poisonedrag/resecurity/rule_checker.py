"""
规则检查器模块

使用正则表达式和规则匹配进行快速初步筛查，辅助 LLM 审查。
主要检测：
- typosquatting 攻击（仿冒包名）
- 敏感关键词
- 可疑 URL
"""

import re
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class RuleCheckResult:
    """规则检查结果"""
    has_pip_install: bool = False
    package_names: List[str] = field(default_factory=list)
    suspicious_packages: List[str] = field(default_factory=list)
    has_sensitive_keywords: bool = False
    has_suspicious_urls: bool = False
    risk_indicators: List[str] = field(default_factory=list)

    @property
    def risk_level(self) -> str:
        """综合风险等级"""
        if self.suspicious_packages or self.has_suspicious_urls:
            return "high"
        elif self.has_pip_install or self.has_sensitive_keywords:
            return "medium"
        else:
            return "low"


class RuleChecker:
    """
    规则检查器

    使用预定义规则进行快速风险筛查。
    """

    # 已知的官方包名
    OFFICIAL_PACKAGES = {
        "fastapi", "flask", "django", "requests", "numpy", "pandas",
        "pytorch", "torch", "tensorflow", "scikit-learn", "sklearn",
        "matplotlib", "seaborn", "pillow", "opencv-python", "cv2",
        "sqlalchemy", "alembic", "celery", "redis", "pytest",
        "black", "ruff", "mypy", "pylint", "uvicorn", "gunicorn",
        "python-dateutil", "python-dotenv", "pydantic", "httpx",
        "aiohttp", "tornado", "sanic", "starlette", "bottle",
        "click", "typer", "rich", "colorama", "tqdm",
        "jinja2", "mako", "chameleon", "genshi",
        "werkzeug", "gunicorn", "gevent", "eventlet",
        "pymongo", "mysql-connector-python", "psycopg2", "sqlite3",
        "elasticsearch", "solr", "whoosh", "lucene",
        "celery", "dramatiq", "huey", "rq",
        "sentry-sdk", "loguru", "structlog", "logging",
        "cryptography", "pyjwt", "passlib", "bcrypt",
        "beautifulsoup4", "lxml", "html5lib", "parsel",
        "selenium", "playwright", "puppeteer", "scrapy",
        "openai", "anthropic", "langchain", "llamaindex",
        "transformers", "huggingface-hub", "datasets", "tokenizers",
        "spacy", "nltk", "gensim", "textblob",
    }

    # 可疑包名模式（typosquatting 检测）
    SUSPICIOUS_PATTERNS = [
        r"[\w-]+-plus$",           # xxx-plus
        r"[\w-]+-utils?$",         # xxx-utils / xxx-util
        r"[\w-]+-pro$",            # xxx-pro
        r"[\w-]+-enhanced$",       # xxx-enhanced
        r"[\w-]+-official$",       # xxx-official
        r"[\w-]+-secure$",         # xxx-secure
        r"[\w-]+-safe$",           # xxx-safe
        r"[\w-]+-fast$",           # xxx-fast
        r"^python-[\w-]+$",        # python-xxx（非官方 python- 前缀）
        r"^py[\w-]+$",             # pyxxx（非官方 py 前缀）
    ]

    # 敏感关键词（危险函数和操作）
    SENSITIVE_KEYWORDS = [
        "eval(", "exec(", "compile(",
        "__import__", "globals()", "locals()",
        "pickle.loads", "pickle.load",
        "subprocess.call", "subprocess.run", "os.system",
        "shell=True",
        "base64.b64decode", "base64.decode",
        "marshal.loads",
        "yaml.load(", "yaml.unsafe_load",
    ]

    # pip install 命令模式
    PIP_INSTALL_PATTERN = re.compile(
        r"pip\s+install\s+([-\w\s\.=<>!]+)",
        re.IGNORECASE
    )

    # 可疑 URL 模式（非知名域名）
    SUSPICIOUS_URL_PATTERN = re.compile(
        r"https?://[^\s'\"<>{}\[\]]+",
        re.IGNORECASE
    )

    # 安全域名白名单
    SAFE_DOMAINS = [
        "github.com", "pypi.org", "readthedocs.io", "python.org",
        "stackoverflow.com", "docs.python.org", "realpython.com",
        "medium.com", "dev.to", "stackoverflow.com",
        "numpy.org", "pandas.pydata.org", "scikit-learn.org",
        "tensorflow.org", "pytorch.org", "langchain.com",
        "openai.com", "anthropic.com",
    ]

    def check(self, text: str) -> RuleCheckResult:
        """
        对文本进行规则检查

        Args:
            text: 待检查文本

        Returns:
            RuleCheckResult 检查结果
        """
        result = RuleCheckResult()

        # 1. 检查 pip install 命令
        self._check_pip_install(text, result)

        # 2. 检查敏感关键词
        self._check_sensitive_keywords(text, result)

        # 3. 检查可疑 URL
        self._check_suspicious_urls(text, result)

        return result

    def _check_pip_install(self, text: str, result: RuleCheckResult):
        """检查 pip install 命令"""
        matches = self.PIP_INSTALL_PATTERN.findall(text)

        if matches:
            result.has_pip_install = True

            for match in matches:
                # 提取包名（去掉版本号、选项等）
                packages = re.findall(r"([a-zA-Z][\w-]*)", match)
                for pkg in packages:
                    pkg_lower = pkg.lower()
                    # 避免重复添加
                    if pkg_lower not in result.package_names:
                        result.package_names.append(pkg_lower)

                    # 检查是否可疑
                    if self._is_suspicious_package(pkg_lower):
                        if pkg_lower not in result.suspicious_packages:
                            result.suspicious_packages.append(pkg_lower)
                            result.risk_indicators.append(f"可疑包名: {pkg_lower}")

    def _is_suspicious_package(self, package_name: str) -> bool:
        """判断包名是否可疑（可能是 typosquatting）"""
        # 检查是否是已知的官方包
        if package_name in self.OFFICIAL_PACKAGES:
            return False

        # 检查可疑模式
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, package_name):
                # 提取可能的原始包名
                base_name = self._extract_base_package_name(package_name)
                if base_name and base_name in self.OFFICIAL_PACKAGES:
                    return True

        # 检查是否有类似的官方包（hyphen/underscore 替换）
        for official in self.OFFICIAL_PACKAGES:
            # hyphen/underscore 替换检测
            normalized_input = package_name.replace("-", "_")
            normalized_official = official.replace("-", "_")
            if normalized_input == normalized_official and package_name != official:
                return True

            # 检查是否只差一两个字符（typo 检测）
            if len(package_name) == len(official) and package_name != official:
                diff_count = sum(1 for a, b in zip(package_name, official) if a != b)
                if diff_count <= 2:
                    return True

        return False

    def _extract_base_package_name(self, package_name: str) -> Optional[str]:
        """从可疑包名中提取可能的原始包名"""
        # 移除常见后缀
        suffixes = ["-plus", "-utils", "-util", "-pro", "-enhanced",
                    "-official", "-secure", "-safe", "-fast"]
        for suffix in suffixes:
            if package_name.endswith(suffix):
                return package_name[:-len(suffix)]

        # 移除 python- 前缀
        if package_name.startswith("python-"):
            return package_name[7:]

        # 移除 py 前缀
        if package_name.startswith("py") and len(package_name) > 2:
            return package_name[2:]

        return None

    def _check_sensitive_keywords(self, text: str, result: RuleCheckResult):
        """检查敏感关键词"""
        for keyword in self.SENSITIVE_KEYWORDS:
            if keyword in text:
                result.has_sensitive_keywords = True
                indicator = f"敏感关键词: {keyword}"
                if indicator not in result.risk_indicators:
                    result.risk_indicators.append(indicator)

    def _check_suspicious_urls(self, text: str, result: RuleCheckResult):
        """检查可疑 URL"""
        urls = self.SUSPICIOUS_URL_PATTERN.findall(text)

        suspicious_urls = []
        for url in urls:
            # 检查是否是安全域名
            is_safe = any(safe_domain in url.lower() for safe_domain in self.SAFE_DOMAINS)
            if not is_safe:
                # 排除一些常见的安全模式
                if not any(safe in url.lower() for safe in [
                    "example.com", "localhost", "127.0.0.1",
                    "docs.", "documentation.", "tutorial."
                ]):
                    suspicious_urls.append(url)

        if suspicious_urls:
            result.has_suspicious_urls = True
            for url in suspicious_urls:
                indicator = f"可疑URL: {url[:50]}..." if len(url) > 50 else f"可疑URL: {url}"
                if indicator not in result.risk_indicators:
                    result.risk_indicators.append(indicator)


def create_rule_checker() -> RuleChecker:
    """创建规则检查器实例"""
    return RuleChecker()