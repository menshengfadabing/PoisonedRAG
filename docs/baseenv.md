当前处于wsl2 + ubuntu24.04系统中

当前项目目录结构如下：❯ ls /home/no_story/projects/Python_project/PoisonedRAG
README.md  docs  main.py  pyproject.toml  src  tests  try

当前处于使用uv初始化的python虚拟环境中，安装和管理包统一使用uv指令

完成此毕设指定的框架以及环境如下：

与模型交互采用langgraph+langchain

基础对话模型采用deepseek
DEEPSEEK_API_KEY = <your-deepseek-api-key>

审查模型使用另一个API_KEY:
DEEPSEEK_REVIEW_API_KEY = <your-deepseek-api-key>

嵌入模型采用本地ollama部署的qwen3-embedding:0.6b
本地ollama已经配置好，端口为"http://localhost:11434"，可以通过langchain_ollama包来快速进行本地模型调用，已经安装好的模型如下：
❯ ollama list
NAME                    ID              SIZE      MODIFIED    
qwen3-embedding:0.6b    ac6da0dfba84    639 MB    2 hours ago  

前端界面采用streamlit，跑在默认端口即可

配置langsmith追踪：
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY=<your-langsmith-api-key>
LANGSMITH_PROJECT="chatbot_system"

❯ uv pip list
Package                                  Version
---------------------------------------- ------------
aiohappyeyeballs                         2.6.1
aiohttp                                  3.13.3
aiosignal                                1.4.0
altair                                   6.0.0
annotated-doc                            0.0.4
annotated-types                          0.7.0
anyio                                    4.12.1
attrs                                    25.4.0
backoff                                  2.2.1
bcrypt                                   5.0.0
blinker                                  1.9.0
build                                    1.4.0
cachetools                               7.0.2
certifi                                  2026.2.25
charset-normalizer                       3.4.4
chromadb                                 1.5.2
click                                    8.3.1
dataclasses-json                         0.6.7
distro                                   1.9.0
durationpy                               0.10
filelock                                 3.25.0
flatbuffers                              25.12.19
frozenlist                               1.8.0
fsspec                                   2026.2.0
gitdb                                    4.0.12
gitpython                                3.1.46
googleapis-common-protos                 1.72.0
greenlet                                 3.3.2
grpcio                                   1.78.0
h11                                      0.16.0
hf-xet                                   1.3.2
httpcore                                 1.0.9
httptools                                0.7.1
httpx                                    0.28.1
httpx-sse                                0.4.3
huggingface-hub                          1.5.0
idna                                     3.11
importlib-metadata                       8.7.1
importlib-resources                      6.5.2
jinja2                                   3.1.6
jiter                                    0.13.0
jsonpatch                                1.33
jsonpointer                              3.0.0
jsonschema                               4.26.0
jsonschema-specifications                2025.9.1
kubernetes                               35.0.0
langchain                                1.2.10
langchain-classic                        1.0.1
langchain-community                      0.4.1
langchain-core                           1.2.17
langchain-ollama                         1.0.1
langchain-openai                         1.1.10
langchain-text-splitters                 1.1.1
langgraph                                1.0.10
langgraph-checkpoint                     4.0.1
langgraph-prebuilt                       1.0.8
langgraph-sdk                            0.3.9
langsmith                                0.7.11
markdown-it-py                           4.0.0
markupsafe                               3.0.3
marshmallow                              3.26.2
mdurl                                    0.1.2
mmh3                                     5.2.0
mpmath                                   1.3.0
multidict                                6.7.1
mypy-extensions                          1.1.0
narwhals                                 2.17.0
numpy                                    2.4.2
oauthlib                                 3.3.1
ollama                                   0.6.1
onnxruntime                              1.24.2
openai                                   2.24.0
opentelemetry-api                        1.39.1
opentelemetry-exporter-otlp-proto-common 1.39.1
opentelemetry-exporter-otlp-proto-grpc   1.39.1
opentelemetry-proto                      1.39.1
opentelemetry-sdk                        1.39.1
opentelemetry-semantic-conventions       0.60b1
orjson                                   3.11.7
ormsgpack                                1.12.2
overrides                                7.7.0
packaging                                26.0
pandas                                   2.3.3
pillow                                   12.1.1
posthog                                  5.4.0
propcache                                0.4.1
protobuf                                 6.33.5
pyarrow                                  23.0.1
pybase64                                 1.4.3
pydantic                                 2.12.5
pydantic-core                            2.41.5
pydantic-settings                        2.13.1
pydeck                                   0.9.1
pygments                                 2.19.2
pypika                                   0.51.1
pyproject-hooks                          1.2.0
python-dateutil                          2.9.0.post0
python-dotenv                            1.2.2
pytz                                     2026.1.post1
pyyaml                                   6.0.3
referencing                              0.37.0
regex                                    2026.2.28
requests                                 2.32.5
requests-oauthlib                        2.0.0
requests-toolbelt                        1.0.0
rich                                     14.3.3
rpds-py                                  0.30.0
shellingham                              1.5.4
six                                      1.17.0
smmap                                    5.0.2
sniffio                                  1.3.1
sqlalchemy                               2.0.48
streamlit                                1.55.0
sympy                                    1.14.0
tenacity                                 9.1.4
tiktoken                                 0.12.0
tokenizers                               0.22.2
toml                                     0.10.2
tornado                                  6.5.4
tqdm                                     4.67.3
typer                                    0.24.1
typing-extensions                        4.15.0
typing-inspect                           0.9.0
typing-inspection                        0.4.2
tzdata                                   2025.3
urllib3                                  2.6.3
uuid-utils                               0.14.1
uvicorn                                  0.41.0
uvloop                                   0.22.1
watchdog                                 6.0.0
watchfiles                               1.1.1
websocket-client                         1.9.0
websockets                               16.0
xxhash                                   3.6.0
yarl                                     1.23.0
zipp                                     3.23.0
zstandard                                0.25.0