// SPDX-License-Identifier: GPL-3.0-only
//! Mistral speech-to-text backend targeting the `super-stt:realtime` world.
//!
//! The component exports both halves of that world:
//!
//! * `wasi:http/incoming-handler` — the Super STT `/v1` batch contract
//!   (`docs/protocol/backend/contract.md`), dispatching on method + path. This
//!   path is stateless: the daemon injects the API key as the
//!   `x-stt-secret-mistral_api_key` request header and the model in the
//!   transcribe body, and the component forwards audio to the Mistral
//!   transcription API over `wasi:http/outgoing-handler`.
//! * `super-stt:realtime/ws-server` — the realtime WebSocket session handler.
//!   It bridges a consumer WebSocket to Mistral's realtime transcription API
//!   over the daemon-implemented `super-stt:realtime/ws` import.
//!
//! The `wit-bindgen` / `wasi:http` handler is **wasm-only**, so it lives behind
//! `#[cfg(target_arch = "wasm32")]` in [`mod@component`]. The pure audio,
//! request-shaping, and realtime-payload helpers stay host-compiled here and
//! are unit-tested natively (a pure-wasm crate could not test them).

// Casts are intentional in audio/WAV encoding; doc lint trips on brand names.
#![allow(clippy::cast_possible_truncation, clippy::doc_markdown)]

use base64::Engine as _;
use serde_json::{Value, json};

#[cfg(target_arch = "wasm32")]
mod component;

// ── batch helpers (pure; host-compiled + unit-tested) ───────────────────────

/// Case-insensitive header lookup over a `(name, value)` list.
#[must_use]
pub fn header(entries: &[(String, Vec<u8>)], want: &str) -> Option<String> {
    entries
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(want))
        .map(|(_, v)| String::from_utf8_lossy(v).into_owned())
}

/// Split a base URL into `(is_https, authority)`, where authority is
/// `host[:port]`. A bare host (no scheme) is treated as HTTPS.
#[must_use]
pub fn parse_base(base: &str) -> (bool, String) {
    if let Some(rest) = base.strip_prefix("https://") {
        (true, rest.trim_end_matches('/').to_string())
    } else if let Some(rest) = base.strip_prefix("http://") {
        (false, rest.trim_end_matches('/').to_string())
    } else {
        (true, base.trim_end_matches('/').to_string())
    }
}

/// Encode f32 samples as a 16-bit PCM mono WAV (mirrors the daemon's
/// `encode_wav_in_memory`).
#[must_use]
pub fn encode_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let bytes_per_sample: u32 = 2;
    let data_len = samples.len() as u32 * bytes_per_sample;
    let mut buf = Vec::with_capacity(44 + data_len as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // channels = mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * bytes_per_sample).to_le_bytes()); // byte rate
    buf.extend_from_slice(&(bytes_per_sample as u16).to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Build a `multipart/form-data` body with `model`, an optional `language`, and
/// `file` (audio.wav).
#[must_use]
pub fn build_multipart(boundary: &str, model: &str, language: Option<&str>, wav: &[u8]) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\n");
    body.extend_from_slice(model.as_bytes());
    body.extend_from_slice(b"\r\n");
    if let Some(lang) = language {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(b"Content-Disposition: form-data; name=\"language\"\r\n\r\n");
        body.extend_from_slice(lang.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n",
    );
    body.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
    body.extend_from_slice(wav);
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    body
}

/// Extract the transcription text from a Mistral batch-API response body.
///
/// # Errors
/// Returns an error string if `bytes` is not JSON or lacks a string `text`
/// field.
pub fn parse_transcript(bytes: &[u8]) -> Result<String, String> {
    serde_json::from_slice::<Value>(bytes)
        .map_err(|e| format!("parse: {e}"))?
        .get("text")
        .and_then(|v| v.as_str())
        .map(String::from)
        .ok_or_else(|| "no_text_field".to_string())
}

// ── realtime helpers (pure; host-compiled + unit-tested) ────────────────────

/// Convert an `http(s)://host` base URL into the realtime `ws(s)://` endpoint.
/// Defaults to `wss://` when no scheme is present.
#[must_use]
pub fn ws_url(base_url: &str, model: &str) -> String {
    let host = if let Some(rest) = base_url.strip_prefix("https://") {
        rest
    } else if let Some(rest) = base_url.strip_prefix("http://") {
        rest
    } else {
        base_url
    };
    let scheme = if base_url.starts_with("http://") {
        "ws"
    } else {
        "wss"
    };
    let host = host.trim_end_matches('/');
    format!("{scheme}://{host}/v1/audio/transcriptions/realtime?model={model}")
}

/// `input_audio.append` payload carrying base64-standard PCM (s16le mono 16 kHz,
/// the format Mistral's realtime transcription expects).
#[must_use]
pub fn audio_append_json(pcm: &[u8]) -> String {
    let audio = base64::engine::general_purpose::STANDARD.encode(pcm);
    json!({ "type": "input_audio.append", "audio": audio }).to_string()
}

/// Parse the consumer's `start` frame: `{"type":"start","sample_rate":N,
/// "language":"xx"}`. Requires `type == "start"`; `sample_rate` defaults to
/// 16000; `language` is optional.
///
/// # Errors
/// Returns an error string when the JSON is invalid or `type != "start"`.
pub fn parse_start(s: &str) -> Result<(u32, Option<String>), String> {
    let v: Value = serde_json::from_str(s).map_err(|_| "invalid start frame".to_string())?;
    if v.get("type").and_then(Value::as_str) != Some("start") {
        return Err("invalid start frame".to_string());
    }
    let sample_rate = v
        .get("sample_rate")
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(16000);
    let language = v
        .get("language")
        .and_then(Value::as_str)
        .map(str::to_string);
    Ok((sample_rate, language))
}

/// `true` if `s` is a JSON object with `type == "stop"`.
#[must_use]
pub fn is_stop(s: &str) -> bool {
    serde_json::from_str::<Value>(s)
        .ok()
        .and_then(|v| v.get("type").and_then(Value::as_str).map(|t| t == "stop"))
        .unwrap_or(false)
}

/// Consumer `preview` frame (incremental transcript).
#[must_use]
pub fn preview_json(text: &str) -> String {
    json!({ "type": "preview", "text": text }).to_string()
}

/// Consumer `done` frame (final transcript).
#[must_use]
pub fn done_json(text: &str) -> String {
    json!({ "type": "done", "transcription": text }).to_string()
}

/// Consumer `error` frame.
#[must_use]
pub fn error_json(msg: &str) -> String {
    json!({ "type": "error", "message": msg }).to_string()
}

/// What one upstream event from Mistral's realtime API means to the session.
///
/// Pure classification, kept here rather than in the wasm-only bridge so the
/// protocol reading is unit-tested on the host. The bridge decides what to
/// send the consumer; this only says what arrived.
#[derive(Debug, PartialEq, Eq)]
pub enum UpstreamEvent {
    /// An incremental transcript fragment to append and preview.
    Delta(String),
    /// The transcript is complete. Carries the upstream's own final text when
    /// it sent one, otherwise the session falls back to what it accumulated.
    Done(Option<String>),
    /// The upstream reported a failure, with the best message it gave.
    Error(String),
    /// Handshake chatter, an unknown event type, or a frame that is not JSON.
    /// Mistral may add event types, and an older backend must keep working, so
    /// anything unrecognized is skipped rather than treated as a failure.
    Ignore,
}

/// Classify one upstream JSON event.
#[must_use]
pub fn classify_upstream_event(s: &str) -> UpstreamEvent {
    let Ok(v) = serde_json::from_str::<Value>(s) else {
        return UpstreamEvent::Ignore;
    };
    match v.get("type").and_then(Value::as_str).unwrap_or("") {
        "error" => {
            let msg = v
                .get("error")
                .and_then(|e| e.get("message"))
                .or_else(|| v.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("upstream error");
            UpstreamEvent::Error(msg.to_string())
        }
        "transcription.done" => {
            UpstreamEvent::Done(v.get("text").and_then(Value::as_str).map(str::to_string))
        }
        // A delta carrying no text says nothing to append, so it is skipped
        // rather than emitting a preview identical to the last one.
        "transcription.text.delta" => v
            .get("text")
            .and_then(Value::as_str)
            .map_or(UpstreamEvent::Ignore, |t| {
                UpstreamEvent::Delta(t.to_string())
            }),
        _ => UpstreamEvent::Ignore,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        UpstreamEvent, build_multipart, classify_upstream_event, encode_wav, is_stop, parse_base,
        parse_start, parse_transcript, ws_url,
    };

    #[test]
    fn encode_wav_header_and_clamping() {
        let wav = encode_wav(&[0.0, 1.0, -1.0, 2.0, -2.0], 16000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[36..40], b"data");
        // 44-byte header + 5 samples × 2 bytes.
        assert_eq!(wav.len(), 44 + 10);
        // sample_rate little-endian at offset 24.
        assert_eq!(&wav[24..28], &16000u32.to_le_bytes());
        // Out-of-range samples clamp to ±i16::MAX.
        let s3 = i16::from_le_bytes([wav[44 + 6], wav[44 + 7]]); // 4th sample (2.0)
        let s4 = i16::from_le_bytes([wav[44 + 8], wav[44 + 9]]); // 5th sample (-2.0)
        assert_eq!(s3, i16::MAX);
        assert_eq!(s4, -i16::MAX);
    }

    #[test]
    fn parse_base_scheme_and_authority() {
        assert_eq!(
            parse_base("https://api.mistral.ai"),
            (true, "api.mistral.ai".into())
        );
        assert_eq!(
            parse_base("http://localhost:8080/"),
            (false, "localhost:8080".into())
        );
        assert_eq!(
            parse_base("api.mistral.ai"),
            (true, "api.mistral.ai".into())
        );
    }

    #[test]
    fn parse_transcript_happy_and_missing() {
        assert_eq!(parse_transcript(br#"{"text":"hi"}"#).unwrap(), "hi");
        assert!(parse_transcript(br#"{"other":"x"}"#).is_err());
        assert!(parse_transcript(b"not json").is_err());
    }

    #[test]
    fn build_multipart_has_model_and_file_parts() {
        let body = build_multipart("BOUND", "voxtral-mini-latest", None, b"WAVDATA");
        let s = String::from_utf8_lossy(&body);
        assert!(s.starts_with("--BOUND\r\n"));
        assert!(s.contains("name=\"model\""));
        assert!(s.contains("voxtral-mini-latest"));
        assert!(s.contains("name=\"file\"; filename=\"audio.wav\""));
        assert!(s.contains("Content-Type: audio/wav"));
        assert!(s.contains("WAVDATA"));
        assert!(!s.contains("name=\"language\""));
        assert!(s.ends_with("--BOUND--\r\n"));
    }

    #[test]
    fn build_multipart_includes_language_when_present() {
        let body = build_multipart("BOUND", "voxtral-mini-latest", Some("es"), b"WAVDATA");
        let s = String::from_utf8_lossy(&body);
        assert!(s.contains("name=\"language\""));
        assert!(s.contains("\r\n\r\nes\r\n"));
    }

    #[test]
    fn ws_url_scheme_mapping() {
        assert_eq!(
            ws_url("https://api.mistral.ai", "m"),
            "wss://api.mistral.ai/v1/audio/transcriptions/realtime?model=m"
        );
        assert_eq!(
            ws_url("http://127.0.0.1:9/", "m"),
            "ws://127.0.0.1:9/v1/audio/transcriptions/realtime?model=m"
        );
        assert_eq!(
            ws_url("api.mistral.ai", "m"),
            "wss://api.mistral.ai/v1/audio/transcriptions/realtime?model=m"
        );
    }

    #[test]
    fn parse_start_valid_default_and_invalid() {
        assert_eq!(
            parse_start(r#"{"type":"start","sample_rate":24000,"language":"fr"}"#).unwrap(),
            (24000, Some("fr".to_string()))
        );
        // Defaults sample_rate to 16000; language optional.
        assert_eq!(parse_start(r#"{"type":"start"}"#).unwrap(), (16000, None));
        assert!(parse_start(r#"{"type":"stop"}"#).is_err());
        assert!(parse_start("not json").is_err());
    }

    #[test]
    fn is_stop_detects_stop_frames() {
        assert!(is_stop(r#"{"type":"stop"}"#));
        assert!(!is_stop(r#"{"type":"start"}"#));
        assert!(!is_stop("garbage"));
    }

    /// The four events the bridge acts on, in the spellings Mistral actually
    /// sends. Getting these wrong is what the realtime protocol fix was for,
    /// so they are pinned here rather than only exercised through a mock.
    #[test]
    fn upstream_events_classify_by_type() {
        assert_eq!(
            classify_upstream_event(r#"{"type":"transcription.text.delta","text":"hello "}"#),
            UpstreamEvent::Delta("hello ".to_string())
        );
        assert_eq!(
            classify_upstream_event(r#"{"type":"transcription.done","text":"hello world"}"#),
            UpstreamEvent::Done(Some("hello world".to_string()))
        );
        assert_eq!(
            classify_upstream_event(r#"{"type":"transcription.done"}"#),
            UpstreamEvent::Done(None),
            "no final text means fall back to what was accumulated"
        );
        assert_eq!(
            classify_upstream_event(r#"{"type":"session.created"}"#),
            UpstreamEvent::Ignore,
            "the handshake is read before the session loop"
        );
    }

    /// An error may name its message at the top level or nested under `error`,
    /// and must still be an error when it names neither.
    #[test]
    fn upstream_errors_surface_the_best_message_available() {
        assert_eq!(
            classify_upstream_event(r#"{"type":"error","error":{"message":"bad key"}}"#),
            UpstreamEvent::Error("bad key".to_string())
        );
        assert_eq!(
            classify_upstream_event(r#"{"type":"error","message":"rate limited"}"#),
            UpstreamEvent::Error("rate limited".to_string())
        );
        assert_eq!(
            classify_upstream_event(r#"{"type":"error"}"#),
            UpstreamEvent::Error("upstream error".to_string())
        );
    }

    /// Mistral may add event types, and a frame may not be JSON at all. Either
    /// way the session keeps going: treating the unknown as terminal would end
    /// a working transcription.
    #[test]
    fn unknown_and_malformed_events_are_skipped() {
        for raw in [
            r#"{"type":"transcription.text.delta"}"#,
            r#"{"type":"something.new"}"#,
            r#"{"no_type":true}"#,
            "not json at all",
            "",
        ] {
            assert_eq!(
                classify_upstream_event(raw),
                UpstreamEvent::Ignore,
                "should be ignored: {raw}"
            );
        }
    }
}
