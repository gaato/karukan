//! Fair inference-speed benchmark: llama.cpp (via karukan-engine) vs candle
//! on the same Qwen3-based kana-kanji GGUF model (jinen-v2-small Q5_K_M).
//!
//! Usage:
//!   candle-bench llamacpp <evaluation_items.json> <out.json> [n_items]
//!   candle-bench candle   <evaluation_items.json> <out.json> [n_items]
//!   candle-bench agree    <evaluation_items.json> <out.json> [n_items]

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Instant;

const VARIANT_ID: &str = "jinen-v2-small-q5";
const MAX_NEW_TOKENS: usize = 50; // karukan-engine ConversionConfig::default()
const EOS_ID: u32 = 3; // </s> — tokenizer.ggml.eos_token_id in the GGUF
/// Added tokens with special=true minus byte-fallback tokens, mirroring
/// LlamaCppModel's special_token_ids: <pad> <unk> <s> </s> + U+EE00..02.
const SPECIAL_IDS: [u32; 7] = [0, 1, 2, 3, 32000, 32001, 32002];
const WARMUP: usize = 3;

#[derive(Deserialize)]
struct BenchItem {
    input: String,
    context_text: Option<String>,
}

#[derive(Serialize)]
struct ItemOut {
    input: String,
    context: String,
    prediction: String,
    latency_ms: f64,
    prompt_tokens: usize,
    /// Token count of the prediction re-encoded with the same tokenizer.
    /// Comparable across engines (the llama.cpp path cannot expose its exact
    /// generated-token count through KanaKanjiConverter::convert).
    gen_tokens_reencoded: usize,
    /// Exact number of kept (non-EOS) sampled tokens; candle path only.
    gen_tokens_exact: Option<usize>,
}

#[derive(Serialize)]
struct RunOut {
    engine: String,
    threads: String,
    versions: String,
    n_items: usize,
    total_ms: f64,
    mean_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    total_gen_tokens_reencoded: usize,
    tokens_per_s: f64,
    items: Vec<ItemOut>,
}

fn load_items(path: &str, n: usize) -> Result<Vec<BenchItem>> {
    let data = std::fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    let items: Vec<BenchItem> = serde_json::from_str(&data)?;
    Ok(items
        .into_iter()
        .filter(|it| !it.input.is_empty())
        .take(n)
        .collect())
}

fn load_tokenizer() -> Result<tokenizers::Tokenizer> {
    let path = karukan_engine::kanji::get_tokenizer_path_by_id(VARIANT_ID)
        .map_err(|e| anyhow!("tokenizer path: {e}"))?;
    let mut tok = tokenizers::Tokenizer::from_file(&path).map_err(|e| anyhow!("{e}"))?;
    tok.with_padding(None);
    tok.with_truncation(None).ok();
    Ok(tok)
}

fn encode_ids(tok: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = tok.encode(text, false).map_err(|e| anyhow!("{e}"))?;
    Ok(enc.get_ids().to_vec())
}

fn stats(engine: &str, threads: String, versions: String, items: Vec<ItemOut>) -> RunOut {
    let mut lat: Vec<f64> = items.iter().map(|i| i.latency_ms).collect();
    lat.sort_by(|a, b| a.total_cmp(b));
    let n = lat.len();
    let total_ms: f64 = lat.iter().sum();
    let median_ms = if n % 2 == 0 {
        (lat[n / 2 - 1] + lat[n / 2]) / 2.0
    } else {
        lat[n / 2]
    };
    let p95_idx = ((n as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    let total_gen: usize = items.iter().map(|i| i.gen_tokens_reencoded).sum();
    RunOut {
        engine: engine.to_string(),
        threads,
        versions,
        n_items: n,
        total_ms,
        mean_ms: total_ms / n as f64,
        median_ms,
        p95_ms: lat[p95_idx],
        total_gen_tokens_reencoded: total_gen,
        tokens_per_s: total_gen as f64 / (total_ms / 1000.0),
        items,
    }
}

// ---------------------------------------------------------------- llama.cpp

fn run_llamacpp(items: &[BenchItem]) -> Result<RunOut> {
    use karukan_engine::kanji::{Backend, KanaKanjiConverter};

    let default_threads = llama_cpp_2::context::params::LlamaContextParams::default().n_threads();
    eprintln!("llama.cpp default n_threads = {default_threads}");

    let t0 = Instant::now();
    let backend = Backend::from_variant_id(VARIANT_ID).map_err(|e| anyhow!("backend: {e}"))?;
    let converter = KanaKanjiConverter::new(backend).map_err(|e| anyhow!("converter: {e}"))?;
    eprintln!("model loaded in {:.1}ms", t0.elapsed().as_secs_f64() * 1e3);

    let tok = load_tokenizer()?; // for token accounting only (untimed)

    let convert = |item: &BenchItem| -> Result<String> {
        let context = item.context_text.as_deref().unwrap_or("");
        let cands = converter
            .convert(&item.input, context, 1)
            .map_err(|e| anyhow!("convert: {e}"))?;
        Ok(cands.into_iter().next().unwrap_or_default())
    };

    for item in items.iter().take(WARMUP) {
        convert(item)?;
    }

    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let start = Instant::now();
        let prediction = convert(item)?;
        let latency_ms = start.elapsed().as_secs_f64() * 1e3;

        let context = item.context_text.as_deref().unwrap_or("");
        let katakana = karukan_engine::kana::hiragana_to_katakana(&item.input);
        let prompt = karukan_engine::kanji::build_jinen_prompt(&katakana, context);
        let prompt_tokens = encode_ids(&tok, &prompt)?.len();
        let gen_tokens_reencoded = encode_ids(&tok, &prediction)?.len();

        eprintln!(
            "[{}/{}] {:.1}ms {}",
            i + 1,
            items.len(),
            latency_ms,
            prediction
        );
        out.push(ItemOut {
            input: item.input.clone(),
            context: context.to_string(),
            prediction,
            latency_ms,
            prompt_tokens,
            gen_tokens_reencoded,
            gen_tokens_exact: None,
        });
    }

    Ok(stats(
        "llamacpp",
        format!("n_threads={default_threads} (llama.cpp default, karukan n_threads=0)"),
        "llama-cpp-2 0.1.154 (karukan-engine path dep), max_new_tokens=50, greedy, n_ctx=256"
            .to_string(),
        out,
    ))
}

// ------------------------------------------------------------------- candle

struct CandleModel {
    model: candle_transformers::models::quantized_qwen3::ModelWeights,
    device: candle_core::Device,
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as u32
}

impl CandleModel {
    fn load(variant_id: &str) -> Result<Self> {
        use candle_core::quantized::gguf_file;
        let gguf_path = karukan_engine::kanji::get_path_by_id(variant_id)
            .map_err(|e| anyhow!("gguf path: {e}"))?;
        let device = candle_core::Device::Cpu;
        let mut file = std::fs::File::open(&gguf_path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let model = candle_transformers::models::quantized_qwen3::ModelWeights::from_gguf(
            content, &mut file, &device,
        )?;
        Ok(Self { model, device })
    }

    /// Mirror of LlamaCppModel::generate + KanaKanjiConverter::convert(_, _, 1):
    /// greedy argmax, stop on EOS before keeping it, at most MAX_NEW_TOKENS kept
    /// tokens, and (like llama.cpp's loop) one final forward after the last
    /// kept token. Returns (prediction, kept_token_count, prompt_token_count).
    fn convert(
        &mut self,
        tok: &tokenizers::Tokenizer,
        reading: &str,
        context: &str,
    ) -> Result<(String, usize, usize)> {
        use candle_core::{DType, Tensor};

        let katakana = karukan_engine::kana::hiragana_to_katakana(reading);
        let prompt = karukan_engine::kanji::build_jinen_prompt(&katakana, context);
        let ids = encode_ids(tok, &prompt)?;

        self.model.clear_kv_cache();
        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut logits = self.model.forward(&input, 0)?; // (1, vocab)

        let mut generated: Vec<u32> = Vec::with_capacity(MAX_NEW_TOKENS);
        let mut pos = ids.len();
        for _ in 0..MAX_NEW_TOKENS {
            let v: Vec<f32> = logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1()?;
            // First max index wins ties, like llama.cpp's greedy sampler.
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &x) in v.iter().enumerate() {
                if x > best_v {
                    best_v = x;
                    best = i;
                }
            }
            let token = best as u32;
            if token == EOS_ID {
                break;
            }
            generated.push(token);
            let inp = Tensor::new(&[token], &self.device)?.unsqueeze(0)?;
            logits = self.model.forward(&inp, pos)?;
            pos += 1;
        }

        // decode(skip_special_tokens=true) equivalent: drop special ids, then
        // ByteFallback-decode the rest; clean_model_output = trim.
        let keep: Vec<u32> = generated
            .iter()
            .copied()
            .filter(|id| !SPECIAL_IDS.contains(id))
            .collect();
        let text = tok.decode(&keep, false).map_err(|e| anyhow!("{e}"))?;
        let clean = text.trim().to_string();
        let prediction = if clean.is_empty() {
            reading.to_string()
        } else {
            clean
        };
        Ok((prediction, generated.len(), ids.len()))
    }

    /// Greedy generation returning (prompt_ids, generated_ids, gen_loop_ms).
    fn greedy_tokens(
        &mut self,
        tok: &tokenizers::Tokenizer,
        reading: &str,
        context: &str,
    ) -> Result<(Vec<u32>, Vec<u32>, f64)> {
        use candle_core::{DType, Tensor};
        let katakana = karukan_engine::kana::hiragana_to_katakana(reading);
        let prompt = karukan_engine::kanji::build_jinen_prompt(&katakana, context);
        let ids = encode_ids(tok, &prompt)?;

        self.model.clear_kv_cache();
        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut logits = self.model.forward(&input, 0)?;
        let t0 = Instant::now();
        let mut generated: Vec<u32> = Vec::new();
        let mut pos = ids.len();
        for _ in 0..MAX_NEW_TOKENS {
            let v: Vec<f32> = logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1()?;
            let token = argmax(&v);
            if token == EOS_ID {
                break;
            }
            generated.push(token);
            let inp = Tensor::new(&[token], &self.device)?.unsqueeze(0)?;
            logits = self.model.forward(&inp, pos)?;
            pos += 1;
        }
        Ok((ids, generated, t0.elapsed().as_secs_f64() * 1e3))
    }

    /// Teacher-forced agreement: after the prompt, does this model's argmax
    /// match y[i] given the true prefix y[..i]? Returns the bitmap and the
    /// per-token loop time in ms.
    fn agree_bitmap(&mut self, prompt_ids: &[u32], y: &[u32]) -> Result<(Vec<bool>, f64)> {
        use candle_core::{DType, Tensor};
        self.model.clear_kv_cache();
        let input = Tensor::new(prompt_ids, &self.device)?.unsqueeze(0)?;
        let mut logits = self.model.forward(&input, 0)?;
        let t0 = Instant::now();
        let mut agree = Vec::with_capacity(y.len());
        let mut pos = prompt_ids.len();
        for &yi in y {
            let v: Vec<f32> = logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1()?;
            agree.push(argmax(&v) == yi);
            let inp = Tensor::new(&[yi], &self.device)?.unsqueeze(0)?;
            logits = self.model.forward(&inp, pos)?;
            pos += 1;
        }
        Ok((agree, t0.elapsed().as_secs_f64() * 1e3))
    }
}

/// Speculative-decoding dry run: teacher-forced draft agreement against the
/// target's greedy output, plus per-token costs of both models.
fn run_agree(items: &[BenchItem], out_path: &str) -> Result<()> {
    let tok = load_tokenizer()?;
    let mut target = CandleModel::load("jinen-v2-small-q5")?;
    let mut draft = CandleModel::load("jinen-v2-xsmall-q5")?;

    #[derive(Serialize)]
    struct AgreeItem {
        n: usize,
        agree: Vec<bool>,
    }
    let mut out_items = Vec::new();
    let (mut t_ms, mut t_tok, mut d_ms, mut d_tok) = (0.0f64, 0usize, 0.0f64, 0usize);
    for (i, item) in items.iter().enumerate() {
        let context = item.context_text.as_deref().unwrap_or("");
        let (prompt_ids, y, gen_ms) = target.greedy_tokens(&tok, &item.input, context)?;
        if y.is_empty() {
            continue;
        }
        let (agree, tf_ms) = draft.agree_bitmap(&prompt_ids, &y)?;
        t_ms += gen_ms;
        t_tok += y.len();
        d_ms += tf_ms;
        d_tok += y.len();
        eprintln!(
            "[{}/{}] {} agree {}/{}",
            i + 1,
            items.len(),
            item.input.chars().take(12).collect::<String>(),
            agree.iter().filter(|&&a| a).count(),
            y.len()
        );
        out_items.push(AgreeItem { n: y.len(), agree });
    }
    let out = serde_json::json!({
        "target_ms_per_tok": t_ms / t_tok as f64,
        "draft_ms_per_tok": d_ms / d_tok as f64,
        "items": out_items,
    });
    std::fs::write(out_path, serde_json::to_string_pretty(&out)?)?;
    eprintln!("wrote {out_path}");
    Ok(())
}

fn run_candle(items: &[BenchItem]) -> Result<RunOut> {
    let n_rayon = rayon::current_num_threads();
    eprintln!("candle rayon threads = {n_rayon}");

    let t0 = Instant::now();
    let mut cm = CandleModel::load(VARIANT_ID)?;
    eprintln!("model loaded in {:.1}ms", t0.elapsed().as_secs_f64() * 1e3);

    let tok = load_tokenizer()?;

    for item in items.iter().take(WARMUP) {
        let context = item.context_text.as_deref().unwrap_or("");
        cm.convert(&tok, &item.input, context)?;
    }

    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let context = item.context_text.as_deref().unwrap_or("");
        let start = Instant::now();
        let (prediction, gen_exact, prompt_tokens) = cm.convert(&tok, &item.input, context)?;
        let latency_ms = start.elapsed().as_secs_f64() * 1e3;

        let gen_tokens_reencoded = encode_ids(&tok, &prediction)?.len();
        eprintln!(
            "[{}/{}] {:.1}ms {}",
            i + 1,
            items.len(),
            latency_ms,
            prediction
        );
        out.push(ItemOut {
            input: item.input.clone(),
            context: context.to_string(),
            prediction,
            latency_ms,
            prompt_tokens,
            gen_tokens_reencoded,
            gen_tokens_exact: Some(gen_exact),
        });
    }

    Ok(stats(
        "candle",
        format!("rayon threads={n_rayon} (RAYON_NUM_THREADS)"),
        "candle-core/candle-transformers 0.11.0 quantized_qwen3, max_new_tokens=50, greedy"
            .to_string(),
        out,
    ))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: candle-bench <llamacpp|candle|agree> <items.json> <out.json> [n_items]");
        std::process::exit(2);
    }
    let mode = &args[1];
    let items_path = &args[2];
    let out_path = &args[3];
    let n: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(50);

    let items = load_items(items_path, n)?;
    eprintln!("{} items, warmup {}", items.len(), WARMUP);

    let run = match mode.as_str() {
        "llamacpp" => run_llamacpp(&items)?,
        "candle" => run_candle(&items)?,
        "agree" => return run_agree(&items, out_path),
        m => return Err(anyhow!("unknown mode {m}")),
    };

    println!(
        "\n== {} ==\nitems: {}  total: {:.1}ms  mean: {:.1}ms  median: {:.1}ms  p95: {:.1}ms",
        run.engine, run.n_items, run.total_ms, run.mean_ms, run.median_ms, run.p95_ms
    );
    println!(
        "gen tokens (reencoded): {}  tokens/s: {:.1}\nthreads: {}",
        run.total_gen_tokens_reencoded, run.tokens_per_s, run.threads
    );

    std::fs::write(out_path, serde_json::to_string_pretty(&run)?)?;
    eprintln!("wrote {out_path}");
    Ok(())
}
