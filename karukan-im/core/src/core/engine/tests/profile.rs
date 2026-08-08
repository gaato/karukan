//! Tests for the conversion profile (config `profile`): a fixed prefix on
//! the lctx sent to the model.
//!
//! These run without a loaded model: the conversion cache is seeded with the
//! lctx the engine is expected to build, and a hit proves the profile was
//! injected into the model call and the cache key.

use super::*;
use crate::core::engine::EngineConfig;
use crate::core::engine::cache::ConversionCacheKey;

fn profile_engine(profile: &str) -> InputMethodEngine {
    let config = EngineConfig {
        profile: profile.to_string(),
        ..EngineConfig::default()
    };
    InputMethodEngine::with_config(config)
}

/// Seed the conversion cache as if the model had converted `katakana` with
/// `lctx` to `converted`.
fn seed_cache(engine: &mut InputMethodEngine, katakana: &str, lctx: &str, converted: &str) {
    engine.conversion_cache.insert(
        ConversionCacheKey {
            katakana: katakana.to_string(),
            lctx: lctx.to_string(),
            strategy: ConversionStrategy::MainModelOnly,
        },
        vec![converted.to_string()],
    );
}

#[test]
fn test_profile_prefixes_model_lctx() {
    // With a profile configured, the model lctx (and thus the cache key) is
    // 「プロフィール:{p}・発言:{ctx}」 — the seeded entry is only reachable
    // through that exact prefix.
    let mut engine = profile_engine("田中太郎/エンジニア");
    seed_cache(
        &mut engine,
        "アイ",
        "プロフィール:田中太郎/エンジニア・発言:",
        "HIT",
    );
    engine.process_key(&press('a'));
    engine.process_key(&press('i'));
    assert_eq!(engine.chunks[0].converted, "HIT");
}

#[test]
fn test_empty_profile_leaves_lctx_unchanged() {
    let mut engine = profile_engine("");
    seed_cache(&mut engine, "アイ", "", "HIT");
    engine.process_key(&press('a'));
    engine.process_key(&press('i'));
    assert_eq!(engine.chunks[0].converted, "HIT");
}

#[test]
fn test_long_profile_keeps_its_tail() {
    // Only the last 25 chars of an over-long profile reach the lctx.
    let mut engine = profile_engine(&"あ".repeat(30));
    let lctx = format!("プロフィール:{}・発言:", "あ".repeat(25));
    seed_cache(&mut engine, "アイ", &lctx, "HIT");
    engine.process_key(&press('a'));
    engine.process_key(&press('i'));
    assert_eq!(engine.chunks[0].converted, "HIT");
}

#[test]
fn test_profile_applies_to_every_chunk_lctx() {
    // Chunked live conversion: each chunk's lctx gets the same profile
    // prefix, with the preceding chunks' converted text after it.
    let config = EngineConfig {
        profile: "太郎".to_string(),
        composing_chunk_len: 2,
        ..EngineConfig::default()
    };
    let mut engine = InputMethodEngine::with_config(config);
    seed_cache(&mut engine, "アイ", "プロフィール:太郎・発言:", "壱");
    seed_cache(&mut engine, "ウエ", "プロフィール:太郎・発言:壱", "弐");
    for k in ['a', 'i', 'u', 'e'] {
        engine.process_key(&press(k));
    }
    let converted: Vec<&str> = engine.chunks.iter().map(|c| c.converted.as_str()).collect();
    assert_eq!(converted, vec!["壱", "弐"]);
}
