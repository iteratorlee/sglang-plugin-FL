# GLM-5.3-Flash：Ascend ModelSlim W8A8

本分支在 BF16 适配基础上，增加 SGLang **0.5.11 / CANN 8.5.0 / 910C**
的单机 **TP16 + EP16** 支持。只修改 plugin，不需要修改 SGLang 或 FlagGems。

## 权重与计算路径

- 权重目录必须包含 `quant_model_description.json` 及 ModelSlim safetensors。
  当前适配的是 `W8A8_DYNAMIC`，不是 compressed-tensors。
- MLP 与专家使用 INT8 权重、逐 token 动态 INT8 激活；注意力、KDA、mHC 等
  未量化部分保留浮点计算。DeepEP 传输 BF16 不会将专家 GEMM 退化成 BF16。
- 专家 GEMM1 保留 INT32 累加值，使用 FP32 scale 反量化，按 GLM 语义截断
  gate ≤ 10、up ∈ [-10, 10] 后执行 SiLU 和激活量化。CANN 融合算子的
  gate/up 交织布局在加载时完成转换，显式设置 `glu_alpha=1, glu_bias=0`。
- 专家 GEMM2 同样保留 INT32 累加值，由 plugin 根据设备端专家计数进行
  FP32 反量化后转为 BF16，避免 CANN 8.5 融合 GMM 要求 BF16 scale 的舍入。
- 主模型权重已离线应用 QuaRot，不再次使用 `rot.safetensors` 旋转。
- 加载时检查新旧 checkpoint 名称映射、decoder 参数覆盖率、对称 offset
  及有效 scale；不支持的量化格式或非 DeepEP 路径会明确报错。

## 启动

以下配置针对 16 个逻辑 NPU、单请求最长 128K 的验证资源预算。
模型配置中的更大最大长度不等于已经验证的容量。保持 plugin 的 Ascend
专用 NPU kernel；关闭全部 FlagGems 替换，额外屏蔽 OOT `RMSNorm`。

```bash
export USE_FLAGGEMS=0
export SGLANG_FLAGGEMS_RECORD=0
export SGLANG_FL_OOT_BLACKLIST=RMSNorm
export SGLANG_DEEPEP_BF16_DISPATCH=0
export HCCL_BUFFSIZE=4096
export HCCL_DETERMINISTIC=true

python -m sglang.launch_server \
  --model-path /models/GLM-5.3-Flash-w8a8 \
  --trust-remote-code --quantization modelslim \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --tp-size 16 --ep-size 16 --moe-a2a-backend deepep \
  --mem-fraction-static 0.78 --max-running-requests 1 \
  --max-total-tokens 196608 --max-prefill-tokens 8192 \
  --chunked-prefill-size 4096 --page-size 64 --disable-radix-cache \
  --cuda-graph-bs 16 --cuda-graph-max-bs 16 --disable-piecewise-cuda-graph \
  --disable-custom-all-reduce --weight-loader-disable-mmap \
  --watchdog-timeout 7200 --decode-log-interval 1 \
  --reasoning-parser glm45 --host 0.0.0.0 --port 31024 --skip-server-warmup
```

`--disable-piecewise-cuda-graph` 不会关闭 decode graph；检查各 rank 的
capture 完成日志和实际请求的 `cuda graph: True` 日志。固定 bs=16 用于
补齐图输入，`--max-running-requests 1` 限制实际并发和 KDA 状态容量。
64k / 128k 性能配置均为单并发、1k 输出，避免 chunk prefill 导致第二个请求等待。

官方模板默认推理档为 `max`（不传 `chat_template_kwargs`，或显式传
`{"reasoning_effort":"max"}`），也支持 `high` / `low`。更换档位会改变
实际模型输入，不能当作量化实现问题的修复。

当前验收边界：真实 TP16 + EP16 服务的 16 个 rank 均完成 graph capture，
8K / 32K / 64K / 128K 多位置检索和跨记录关联的答案语义正确，双 64K
并发请求也正常结束。部分重复请求仅 JSON 空格不同，需与语义错误区分。
严格的 64 条长复制仍观察到真实错误：`high` 的 32K 两次中一次遗漏一条；
`max` 的 128K 一次给某条加了多余前缀。正常 stop，不是输入截断或解析
修复后的假通过，尚不能宣称完整长输出精度验收通过。

## 验证

NPU 单元测试：`tests/functional_tests/models/test_glm5_modelslim.py`，覆盖
CPU FP32 参考、EP16 的 18 个本地专家、空专家、变路由、零输入与
graph/eager 逐位一致性；当前专门回归为 18 passed。

HTTP 验证工具在 `tests/functional_tests/models/glm5_long_context/`：
`validate_short_chat.py`、`validate_long_context.py`、`validate_long_output.py`。
传入上述 model path、服务 base URL、`--reasoning-effort max` 和输出路径；
长输入用 checkpoint
tokenizer/template 构造，并检查服务实际 token 数、完整 JSON、正常 stop
及重复请求最终答案一致性。完整请求和原始响应保留供独立审计。

## 推理优化与通信精度

W8A8 专家允许原生 INT8 DeepEP low-latency dispatch，直接使用通信返回的
INT8 激活和逐 token scale，省去对填充接收缓冲区的二次动态量化。
`SGLANG_DEEPEP_BF16_DISPATCH=1` 仍可显式恢复 BF16 传输；浮点权重的专家
始终保留 BF16 通信兼容保护。两种传输方式下，W8A8 专家计算均为
INT8×INT8→INT32，不将权重反量化为 BF16 后做 GEMM。

专用 mHC kernel 融合系数 / 20 次 Sinkhorn 迭代 / 残差混合；专家输出
dequant 仅处理设备端路由计数指定的有效行，并保留填充区为零。
可设置 `SGLANG_FL_GLM53_AUDIT_DIR` 保存每个 worker 的有界 dtype 审计。
新增 NPU 回归见 `test_glm5_mhc_fused.py` 和 `test_glm5_quant_dequant.py`，
与原 W8A8 测试合计 37 项通过。有限语义验收不代表完整质量或逐 token 等价。

本镜像应给生产服务使用独立 Triton cache；不要混用不同
`TRITON_ALL_BLOCKS_PARALLEL` 编译模式产生的缓存。

## NPU MTP 采样语义

SGLang 0.5.11 缺少 NPU tree sampling kernel 时，上游 EAGLE verifier 会将
非贪心请求回退为 argmax。GLM 的 `mtp_sampling.py` 在这个分支为每个
verify 节点按目标 logits 的 temperature/top-k/top-p/min-p 分布抽样，
再复用树匹配：抽样命中 draft 边时继续接受，否则输出该抽样作为 bonus
token。抽样结果从 TP rank 0 广播，确保各 rank 提交相同的 KV/KDA 前缀。

该适配只作用于 GLM、NPU、非贪心请求和缺少原生 tree sampling kernel
的组合；temperature=0 直接调用原 verifier。模型计算、W8A8、graph 和
现有 KV/KDA 提交逻辑保持。当前支持部署使用的 EAGLE v1、topk=1、关闭
overlap；不保证与非 MTP 或不同批次消耗相同随机数，也不扩展上游的
speculative penalty 语义。不要用 do_sample=false 代替 temperature=0。

真实 NPU 回归在 `tests/functional_tests/models/test_glm5_mtp_sampling.py`，
覆盖采样分布、过滤参数、混合批次、接受前缀及 TP 同步接入。分布测试
不能代替整模型质量评测；部署后还应检查非贪心请求不再出现
`Falling back to greedy verification`，并进行同输入、同预算的回放。
