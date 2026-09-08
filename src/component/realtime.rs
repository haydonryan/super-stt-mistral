// SPDX-License-Identifier: GPL-3.0-only
//! Mistral realtime WebSocket transcription bridge.
//!
//! Bridges a consumer WebSocket session to Mistral's realtime transcription
//! API (`wss://api.mistral.ai/v1/audio/transcriptions/realtime`).
//!
//! ## Full duplex
//! The session waits on the consumer and the upstream at the same time, via
//! `subscribe` on both and one `wasi:io/poll`. A `transcription.text.delta`
//! therefore reaches the consumer as it arrives, while audio is still going
//! up, which is what makes the previews live.
//!
//! This used to run half-duplex, forwarding all the audio first and only then
//! draining the upstream, because the host's `subscribe` was a stub that
//! trapped. It is implemented now, and readiness is non-destructive: the frame
//! that made a pollable ready is still returned by the next `recv` on that
//! resource, so polling never costs a frame.
//!
//! Once the consumer stops, only the upstream can still speak, so the loop
//! drops the consumer from the poll set and reads the upstream directly to
//! completion.
//!
//! The pure frame-parsing and payload-building helpers (`parse_start`,
//! `is_stop`, `ws_url`, `audio_append_json`, `classify_upstream_event`,
//! `preview_json`/`done_json`/`error_json`, `header`) live in the crate root so
//! they compile and unit-test on the host; this module wires them to the
//! wasm-only `super-stt:realtime` resources.

use serde_json::Value;

use super::exports::super_stt::realtime::ws_server::Guest as WsServerGuest;
use super::super_stt::realtime::ws::{self, ConsumerStream, WsError, WsFrame, WsStream};
use super::wasi::io::poll;
use crate::UpstreamEvent;

/// Index of the consumer's pollable in the poll set, and of the upstream's.
/// `poll` answers with indices into the slice it was given.
const CONSUMER: u32 = 0;
const UPSTREAM: u32 = 1;

const DEFAULT_BASE_URL: &str = "https://api.mistral.ai";
const DEFAULT_MODEL: &str = "voxtral-mini-transcribe-realtime-2602";
const INPUT_AUDIO_FLUSH: &str = r#"{"type":"input_audio.flush"}"#;
const INPUT_AUDIO_END: &str = r#"{"type":"input_audio.end"}"#;

impl WsServerGuest for super::Component {
    fn handle(headers: Vec<(String, Vec<u8>)>, consumer: ConsumerStream) -> Result<(), WsError> {
        run(&headers, &consumer)
    }
}

fn run(headers: &[(String, Vec<u8>)], consumer: &ConsumerStream) -> Result<(), WsError> {
    let Some(api_key) = crate::header(headers, "x-stt-secret-mistral_api_key") else {
        let _ = consumer.send_text(&crate::error_json("missing mistral api key"));
        return Ok(());
    };
    // Internal test seam: `base_url` is not a declared manifest option, so the
    // daemon never sends this header in production — the default is always used.
    // The realtime round-trip test injects it to reach a mock upstream.
    let base_url = crate::header(headers, "x-stt-option-base_url")
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
    let model = crate::header(headers, "x-stt-model").unwrap_or_else(|| DEFAULT_MODEL.to_string());

    // 1. Read and validate the consumer's `start` frame. Mistral needs no
    //    session config — the model is in the URL and PCM s16le/16 kHz is
    //    assumed — so the frame's fields are not forwarded upstream.
    match consumer.recv()? {
        WsFrame::Text(s) if crate::parse_start(&s).is_ok() => {}
        WsFrame::Text(_) => {
            let _ = consumer.send_text(&crate::error_json("invalid start frame"));
            return Ok(());
        }
        WsFrame::Close(_) => return Ok(()), // consumer hung up before starting
        WsFrame::Binary(_) => {
            let _ = consumer.send_text(&crate::error_json("audio before start frame"));
            return Ok(());
        }
    }

    // 2. Open the upstream WS and wait for `session.created` before streaming.
    let url = crate::ws_url(&base_url, &model);
    let upstream = match ws::connect(
        &url,
        &[(
            "authorization".to_string(),
            format!("Bearer {api_key}").into_bytes(),
        )],
    ) {
        Ok(u) => u,
        Err(e) => {
            let _ = consumer.send_text(&crate::error_json(&format!(
                "upstream connect failed: {e:?}"
            )));
            return Ok(());
        }
    };
    if !await_session_created(&upstream, consumer) {
        return Ok(());
    }

    // 3. Pump both directions at once for as long as the consumer is sending.
    //    Audio goes up as it arrives and transcripts come down as they arrive,
    //    rather than one after the other.
    let mut accumulated = String::new();
    let mut input_open = true;
    while input_open {
        let consumer_ready = consumer.subscribe();
        let upstream_ready = upstream.subscribe();
        let ready = poll::poll(&[&consumer_ready, &upstream_ready]);

        // Both can be ready at once, and each is handled on its own: the frame
        // that made a pollable ready is still waiting on that resource.
        if ready.contains(&CONSUMER) {
            match forward_consumer_frame(consumer, &upstream) {
                Input::Open => {}
                Input::Ended => {
                    if !end_input(&upstream, consumer) {
                        return Ok(());
                    }
                    input_open = false;
                }
                Input::Failed => return Ok(()),
            }
        }
        if ready.contains(&UPSTREAM) && read_upstream(&upstream, consumer, &mut accumulated) {
            // The upstream finished or failed while the consumer was still
            // sending. Nothing further can change the outcome.
            let _ = consumer.close();
            return Ok(());
        }
    }

    // 4. The input is closed, so only the upstream can still speak. Read it to
    //    completion without polling a consumer that has nothing left to say.
    while !read_upstream(&upstream, consumer, &mut accumulated) {}
    let _ = consumer.close();
    Ok(())
}

/// What the consumer's latest frame means for the session.
enum Input {
    /// More audio may follow.
    Open,
    /// The consumer said `stop` or hung up; finalize the upstream input.
    Ended,
    /// The upstream could not be written to; the session is over.
    Failed,
}

/// Read one consumer frame and forward it upstream.
///
/// A consumer that has gone away is treated as a `stop` rather than an error:
/// the audio it did send is still worth transcribing, and the upstream owes a
/// final transcript for it.
fn forward_consumer_frame(consumer: &ConsumerStream, upstream: &WsStream) -> Input {
    match consumer.recv() {
        Ok(WsFrame::Binary(pcm)) => {
            if let Err(e) = upstream.send_text(&crate::audio_append_json(&pcm)) {
                let _ =
                    consumer.send_text(&crate::error_json(&format!("upstream send failed: {e:?}")));
                return Input::Failed;
            }
            Input::Open
        }
        Ok(WsFrame::Text(s)) if crate::is_stop(&s) => Input::Ended,
        Ok(WsFrame::Text(_)) => Input::Open, // ignore unknown control frames
        Ok(WsFrame::Close(_)) | Err(WsError::Closed) => Input::Ended,
        Err(e) => {
            let _ = consumer.send_text(&crate::error_json(&format!("consumer recv failed: {e:?}")));
            Input::Failed
        }
    }
}

/// Tell the upstream no more audio is coming. Returns `false` (after notifying
/// the consumer) when the upstream could not be written to.
fn end_input(upstream: &WsStream, consumer: &ConsumerStream) -> bool {
    for msg in [INPUT_AUDIO_FLUSH, INPUT_AUDIO_END] {
        if let Err(e) = upstream.send_text(msg) {
            let _ = consumer.send_text(&crate::error_json(&format!("flush/end failed: {e:?}")));
            return false;
        }
    }
    true
}

/// Read upstream until Mistral's `session.created` handshake event. Returns
/// `false` (after notifying the consumer) if the upstream errors or closes
/// before the session is ready.
fn await_session_created(upstream: &WsStream, consumer: &ConsumerStream) -> bool {
    loop {
        match upstream.recv() {
            Ok(WsFrame::Text(s)) => match event_type(&s).as_deref() {
                Some("session.created") => return true,
                Some("error") => {
                    let _ = consumer.send_text(&crate::error_json(&format!("upstream error: {s}")));
                    return false;
                }
                _ => {} // ignore other handshake chatter
            },
            Ok(WsFrame::Binary(_)) => {}
            Ok(WsFrame::Close(_)) | Err(_) => {
                let _ = consumer.send_text(&crate::error_json("upstream closed during handshake"));
                return false;
            }
        }
    }
}

/// The `type` field of a JSON event, if present.
fn event_type(s: &str) -> Option<String> {
    serde_json::from_str::<Value>(s)
        .ok()
        .and_then(|v| v.get("type").and_then(Value::as_str).map(str::to_string))
}

/// Read one upstream frame and relay what it means to the consumer. Returns
/// `true` when the session is over (the transcript completed, the upstream
/// failed, or it closed), `false` to keep reading.
fn read_upstream(upstream: &WsStream, consumer: &ConsumerStream, accumulated: &mut String) -> bool {
    match upstream.recv() {
        Ok(WsFrame::Text(s)) => handle_upstream_event(&s, consumer, accumulated),
        Ok(WsFrame::Binary(_)) => false, // Mistral sends JSON text; ignore binary
        Ok(WsFrame::Close(_)) | Err(WsError::Closed) => {
            // Upstream closed without a completed event: emit what we have.
            let _ = consumer.send_text(&crate::done_json(accumulated.trim()));
            true
        }
        Err(e) => {
            let _ = consumer.send_text(&crate::error_json(&format!("upstream recv failed: {e:?}")));
            true
        }
    }
}

/// Handle one upstream JSON event. Returns `true` when the session is complete
/// (a done or error event), `false` to keep reading.
fn handle_upstream_event(s: &str, consumer: &ConsumerStream, accumulated: &mut String) -> bool {
    match crate::classify_upstream_event(s) {
        UpstreamEvent::Delta(delta) => {
            accumulated.push_str(&delta);
            let _ = consumer.send_text(&crate::preview_json(accumulated.trim()));
            false
        }
        UpstreamEvent::Done(text) => {
            let transcript = text.unwrap_or_else(|| accumulated.trim().to_string());
            let _ = consumer.send_text(&crate::done_json(&transcript));
            true
        }
        UpstreamEvent::Error(msg) => {
            let _ = consumer.send_text(&crate::error_json(&msg));
            true
        }
        UpstreamEvent::Ignore => false,
    }
}
