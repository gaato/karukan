# candle-bench

llama.cpp(karukan-engine 経由)と [candle](https://github.com/huggingface/candle) で、同一の jinen GGUF モデル・同一の tokenizer.json・同一の貪欲デコードによる推論速度を公平比較するベンチマークです。あわせて、投機的デコードの効果見積もりに使う draft/target のトークン一致率計測(`agree`)も含みます。

karukan のワークスペースには**意図的に入れていません**。candle は重い依存で、通常ビルドや CI に載せる理由がないためです。このディレクトリ内で単体ビルドして使います。

## 使い方

```bash
cd tools/candle-bench

# candle の量子化SIMDカーネルはコンパイル時ゲートなので native 必須。
# これを付けないと candle 側だけ SIMD なしになり比較になりません。
export RUSTFLAGS="-C target-cpu=native"

# llama.cpp 側(karukan-engine 経由)と candle 側をそれぞれ計測
cargo run --release -- llamacpp evaluation_items.json out_llamacpp.json 50
RAYON_NUM_THREADS=4 cargo run --release -- candle evaluation_items.json out_candle.json 50

# 突き合わせ(出力一致率・レイテンシ比較)
python3 compare.py out_llamacpp.json out_candle.json

# 投機的デコードのドライラン: main の貪欲出力を light が言い当てる率と
# 両モデルのトークン当たりコストを計測
RAYON_NUM_THREADS=4 cargo run --release -- agree evaluation_items.json agree.json 50
```

- 入力は AJIMEE-Bench の `evaluation_items.json`(input / context_text / expected_output)
- モデルは karukan-engine の解決経路(HFキャッシュ)をそのまま使うため、karukan を一度動かした環境ならオフラインで走ります
- llama.cpp 側のスレッド数は karukan と同じデフォルト(n_threads=0)。candle 側は `RAYON_NUM_THREADS` で合わせてください

## 計測スナップショット (2026-08-15, Ryzen AI 7 PRO 350, 両者4スレッド)

jinen-v2-small-Q5_K_M、AJIMEE 50問、貪欲・max_new_tokens=50:

| | mean | median | p95 | tokens/s |
|---|---:|---:|---:|---:|
| llama.cpp (llama-cpp-2 0.1.154) | 42.9ms | 38.5ms | 87.6ms | 250 |
| candle 0.11.0 (`quantized_qwen3`) | 47.3ms | 41.7ms | 101.9ms | 227 |

出力一致率 49/50(不一致は1トークンの量子化誤差)。candle は約10%遅いが品質は実質同一。

`agree`(small の貪欲出力に対する xsmall の teacher-forced 一致率): **96.1%**、トークン当たりコスト比 draft/target ≈ 0.29。
