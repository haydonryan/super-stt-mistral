// SPDX-License-Identifier: GPL-3.0-only
//! Realtime `ws-server` session against a `tokio-tungstenite` mock of the
//! Mistral realtime WebSocket upstream. Drives a consumer transport through
//! `WasmBackend::realtime_session` and asserts preview + done frames return —
//! exercising the component's full upstream bridge over the host `ws` import.
//!
//! Three sessions are covered:
//! 1. The upstream speaks only after the input ends. The transcript still
//!    arrives, which is the finalize path.
//! 2. The upstream speaks *during* the audio. The preview must reach the
//!    consumer before it says `stop`, which is only possible if the bridge
//!    waits on both streams at once. A half-duplex bridge deadlocks here.
//! 3. The upstream fails during the audio. The session ends on that, without
//!    waiting for a `stop` that a half-duplex bridge would still be blocked on.
#![allow(clippy::doc_markdown)]

mod common;

use std::time::Duration;

use common::{ConsumerStreamTransport, WasmBackend, WsFrame};
use futures::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message as WsMessage;

macro_rules! component_or_skip {
    () => {
        match common::component_path() {
            Some(p) => p,
            None => {
                eprintln!("skipping: component not built (run `just build-component`)");
                return;
            }
        }
    };
}

/// Mock Mistral realtime upstream. Accepts the WS upgrade, sends
/// `session.created`, consumes `input_audio.append*`, and on `input_audio.end`
/// replies with two `transcription.text.delta` events then `transcription.done`.
/// Returns the bound authority (host:port) and the accept task handle.
async fn start_mock_upstream() -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let authority = listener.local_addr().unwrap().to_string(); // "127.0.0.1:PORT"
    let handle = tokio::spawn(async move {
        // Accept exactly one upstream connection (the guest's).
        let Ok((tcp, _)) = listener.accept().await else {
            return;
        };
        let Ok(mut ws) = accept_async(tcp).await else {
            return;
        };
        // Handshake: the guest waits for `session.created` before streaming.
        let _ = ws
            .send(WsMessage::Text(r#"{"type":"session.created"}"#.into()))
            .await;
        while let Some(Ok(msg)) = ws.next().await {
            // input_audio.append*: consume silently; react once the input ends.
            if let WsMessage::Text(t) = msg
                && t.as_str().contains("input_audio.end")
            {
                let _ = ws
                    .send(WsMessage::Text(
                        r#"{"type":"transcription.text.delta","text":"hello "}"#.into(),
                    ))
                    .await;
                let _ = ws
                    .send(WsMessage::Text(
                        r#"{"type":"transcription.text.delta","text":"world"}"#.into(),
                    ))
                    .await;
                let _ = ws
                    .send(WsMessage::Text(
                        r#"{"type":"transcription.done","text":"hello world"}"#.into(),
                    ))
                    .await;
                // Done; the guest closes after `transcription.done`.
                break;
            }
        }
    });
    (authority, handle)
}

#[tokio::test]
async fn realtime_round_trip() {
    let path = component_or_skip!();
    let (authority, _mock) = start_mock_upstream().await;

    // The guest builds the upstream URL from x-stt-option-base_url. Point it at
    // the mock over plaintext ws:// (http:// -> ws:// in the guest). The mock is
    // on loopback, which the SSRF guard blocks for untrusted backends, so the
    // test opts in via `permit_loopback_egress` below.
    let backend = WasmBackend::new_realtime(
        &path,
        vec![authority.clone()],
        "voxtral-mini-transcribe-realtime-2602".to_string(),
        vec![
            (
                "x-stt-secret-mistral_api_key".to_string(),
                "test-key".to_string(),
            ),
            (
                "x-stt-option-base_url".to_string(),
                format!("http://{authority}"),
            ),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    // Channels: consumer_tx -> guest (incoming); guest -> guest_rx (outgoing).
    let (consumer_tx, consumer_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let (guest_tx, mut guest_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let transport = ConsumerStreamTransport {
        incoming: consumer_rx,
        outgoing: guest_tx,
    };

    // Drive the consumer side concurrently with the session.
    let driver = tokio::spawn(async move {
        consumer_tx
            .send(WsFrame::Text(
                r#"{"type":"start","sample_rate":16000}"#.to_string(),
            ))
            .unwrap();
        // A couple of PCM chunks (silence is fine for the mock).
        consumer_tx.send(WsFrame::Binary(vec![0u8; 3200])).unwrap();
        consumer_tx.send(WsFrame::Binary(vec![0u8; 3200])).unwrap();
        consumer_tx
            .send(WsFrame::Text(r#"{"type":"stop"}"#.to_string()))
            .unwrap();
        // Keep consumer_tx alive briefly so the guest doesn't see an early close
        // before reading `stop`. Dropping it after is harmless.
        consumer_tx
    });

    // Run the session with a timeout so a hang fails loudly.
    let session =
        tokio::time::timeout(Duration::from_secs(30), backend.realtime_session(transport));
    let result = session.await.expect("session timed out");
    let _held = driver.await.unwrap(); // keep consumer_tx alive until session ends
    result.expect("session returned an error");

    // Collect everything the guest sent to the consumer.
    let mut texts = Vec::new();
    while let Ok(frame) = guest_rx.try_recv() {
        if let WsFrame::Text(s) = frame {
            texts.push(s);
        }
    }

    assert!(
        texts.iter().any(|t| t.contains(r#""type":"preview""#)),
        "expected at least one preview frame; got {texts:?}"
    );
    let done = texts
        .iter()
        .find(|t| t.contains(r#""type":"done""#))
        .unwrap_or_else(|| panic!("expected a done frame; got {texts:?}"));
    assert!(
        done.contains("hello world"),
        "done frame should contain the transcript; got {done}"
    );
}

/// A mock that speaks first: it sends a `transcription.text.delta` as soon as
/// the session is up, while the consumer is still sending audio and long
/// before `input_audio.end`. It finishes the transcript once the input ends.
async fn start_streaming_mock_upstream() -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let authority = listener.local_addr().unwrap().to_string();
    let handle = tokio::spawn(async move {
        let Ok((tcp, _)) = listener.accept().await else {
            return;
        };
        let Ok(mut ws) = accept_async(tcp).await else {
            return;
        };
        let _ = ws
            .send(WsMessage::Text(r#"{"type":"session.created"}"#.into()))
            .await;
        // The early delta. Nothing has been flushed or ended yet.
        let _ = ws
            .send(WsMessage::Text(
                r#"{"type":"transcription.text.delta","text":"hello "}"#.into(),
            ))
            .await;
        while let Some(Ok(msg)) = ws.next().await {
            if let WsMessage::Text(t) = msg
                && t.as_str().contains("input_audio.end")
            {
                let _ = ws
                    .send(WsMessage::Text(
                        r#"{"type":"transcription.text.delta","text":"world"}"#.into(),
                    ))
                    .await;
                let _ = ws
                    .send(WsMessage::Text(
                        r#"{"type":"transcription.done","text":"hello world"}"#.into(),
                    ))
                    .await;
                break;
            }
        }
    });
    (authority, handle)
}

/// A mock that fails mid-stream: it reports an error while the consumer is
/// still sending audio, and never acknowledges the input at all.
async fn start_failing_mock_upstream() -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let authority = listener.local_addr().unwrap().to_string();
    let handle = tokio::spawn(async move {
        let Ok((tcp, _)) = listener.accept().await else {
            return;
        };
        let Ok(mut ws) = accept_async(tcp).await else {
            return;
        };
        let _ = ws
            .send(WsMessage::Text(r#"{"type":"session.created"}"#.into()))
            .await;
        let _ = ws
            .send(WsMessage::Text(
                r#"{"type":"error","error":{"message":"upstream exploded"}}"#.into(),
            ))
            .await;
        // Stay up so the guest's decision comes from the event, not the close.
        while ws.next().await.is_some() {}
    });
    (authority, handle)
}

fn realtime_backend(authority: &str) -> WasmBackend {
    let path = common::component_path().expect("component built");
    WasmBackend::new_realtime(
        &path,
        vec![authority.to_string()],
        "voxtral-mini-transcribe-realtime-2602".to_string(),
        vec![
            (
                "x-stt-secret-mistral_api_key".to_string(),
                "test-key".to_string(),
            ),
            (
                "x-stt-option-base_url".to_string(),
                format!("http://{authority}"),
            ),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress()
}

/// Collect the text frames still queued from the guest.
fn drain_texts(rx: &mut tokio::sync::mpsc::UnboundedReceiver<WsFrame>) -> Vec<String> {
    let mut texts = Vec::new();
    while let Ok(frame) = rx.try_recv() {
        if let WsFrame::Text(s) = frame {
            texts.push(s);
        }
    }
    texts
}

/// The regression test for the half-duplex bridge.
///
/// The consumer withholds `stop` until it has seen a preview. The old bridge
/// forwarded every audio frame before reading the upstream even once, so it
/// could not produce that preview until after `stop` — and `stop` never comes.
/// It deadlocks, and the driver's timeout says so. Full duplex satisfies both
/// sides at once.
#[tokio::test]
async fn a_preview_arrives_while_the_consumer_is_still_sending() {
    let _path = component_or_skip!();
    let (authority, _mock) = start_streaming_mock_upstream().await;
    let backend = realtime_backend(&authority);

    let (consumer_tx, consumer_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let (guest_tx, guest_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let transport = ConsumerStreamTransport {
        incoming: consumer_rx,
        outgoing: guest_tx,
    };

    let driver = tokio::spawn(async move {
        let mut guest_rx = guest_rx;
        consumer_tx
            .send(WsFrame::Text(
                r#"{"type":"start","sample_rate":16000}"#.to_string(),
            ))
            .unwrap();
        consumer_tx.send(WsFrame::Binary(vec![0u8; 3200])).unwrap();

        // Block on the preview before conceding `stop`.
        let preview = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                match guest_rx.recv().await {
                    Some(WsFrame::Text(t)) if t.contains(r#""type":"preview""#) => {
                        break Some(t);
                    }
                    Some(_) => {}
                    None => break None,
                }
            }
        })
        .await;

        consumer_tx
            .send(WsFrame::Text(r#"{"type":"stop"}"#.to_string()))
            .unwrap();
        (preview, consumer_tx, guest_rx)
    });

    let session =
        tokio::time::timeout(Duration::from_secs(30), backend.realtime_session(transport));
    let result = session.await.expect("session timed out");
    let (preview, _held, mut guest_rx) = driver.await.unwrap();
    result.expect("session returned an error");

    let preview = preview
        .expect("no preview before stop: the bridge is not full duplex")
        .expect("the guest closed the consumer channel without previewing");
    assert!(
        preview.contains("hello"),
        "the early delta should be previewed; got {preview}"
    );

    // And the session still finalizes normally afterwards.
    let texts = drain_texts(&mut guest_rx);
    let done = texts
        .iter()
        .find(|t| t.contains(r#""type":"done""#))
        .unwrap_or_else(|| panic!("expected a done frame; got {texts:?}"));
    assert!(
        done.contains("hello world"),
        "done frame should carry the full transcript; got {done}"
    );
}

/// An upstream failure during the audio ends the session there and then.
///
/// The consumer never sends `stop`. A bridge that only reads the upstream
/// after the input ends would sit waiting for one forever; reading both means
/// the error is seen and relayed while the input is still open.
#[tokio::test]
async fn an_upstream_error_ends_the_session_without_a_stop() {
    let _path = component_or_skip!();
    let (authority, _mock) = start_failing_mock_upstream().await;
    let backend = realtime_backend(&authority);

    let (consumer_tx, consumer_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let (guest_tx, mut guest_rx) = tokio::sync::mpsc::unbounded_channel::<WsFrame>();
    let transport = ConsumerStreamTransport {
        incoming: consumer_rx,
        outgoing: guest_tx,
    };

    consumer_tx
        .send(WsFrame::Text(
            r#"{"type":"start","sample_rate":16000}"#.to_string(),
        ))
        .unwrap();
    consumer_tx.send(WsFrame::Binary(vec![0u8; 3200])).unwrap();

    let session =
        tokio::time::timeout(Duration::from_secs(30), backend.realtime_session(transport));
    let result = session.await.expect(
        "session timed out: the bridge was still waiting for a stop the consumer never sends",
    );
    result.expect("session returned an error");
    drop(consumer_tx);

    let texts = drain_texts(&mut guest_rx);
    let error = texts
        .iter()
        .find(|t| t.contains(r#""type":"error""#))
        .unwrap_or_else(|| panic!("expected an error frame; got {texts:?}"));
    assert!(
        error.contains("upstream exploded"),
        "the upstream's own message should reach the consumer; got {error}"
    );
}
