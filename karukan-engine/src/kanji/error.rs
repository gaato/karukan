//! Error types for kanji conversion

/// Errors that can occur during kanji conversion operations.
#[derive(Debug, thiserror::Error)]
pub enum KanjiError {
    #[error("unknown model variant: '{0}'")]
    UnknownVariant(String),

    #[error("invalid model spec: '{0}' (expected \"hf:owner/repo/filename.gguf\")")]
    InvalidSpec(String),

    #[error(
        "tokenizer.json not found: put tokenizer.json in the same directory as the GGUF (expected {0})"
    )]
    TokenizerNotFound(String),

    #[error("download failed")]
    Download(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("model load failed")]
    ModelLoad(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("tokenizer load failed")]
    TokenizerLoad(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[error("inference failed")]
    Inference(#[source] Box<dyn std::error::Error + Send + Sync>),
}

pub type Result<T> = std::result::Result<T, KanjiError>;
