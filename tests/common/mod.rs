// SPDX-License-Identifier: GPL-3.0-only
//! Self-contained wasmtime host harness that "plays the daemon" for the Mistral
//! *realtime* backend: it loads the prebuilt component and drives both halves of
//! the `realtime-backend` world in-process, exactly as Super STT's daemon does —
//! the batch `/v1` contract over `wasi:http`, and the `super-stt:realtime`
//! WebSocket session — while confining the component's outbound egress to an
//! allowlist with the same SSRF guard. This repo shares no code with super-stt;
//! the harness is a minimal, independent reimplementation of the daemon's
//! `WasmBackend` (`stt_models/wasm/{mod,host,ws_host}.rs`).
//!
//! Tests pair it with mocks of the *Mistral upstream*: `wiremock` for the batch
//! HTTP API and a `tokio-tungstenite` server for the realtime WebSocket — so
//! every side of the contract the component speaks (daemon ⇄ component ⇄
//! upstream) is mocked, and the component itself is real.

#![allow(dead_code)] // not every test uses every helper

use std::net::{IpAddr, Ipv4Addr, ToSocketAddrs};
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use futures::{SinkExt, StreamExt};
use http_body_util::BodyExt;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::Uri;
use tokio_tungstenite::tungstenite::protocol::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};
use wasmtime::component::{Component, Linker, Resource, ResourceTable};
use wasmtime::{Config, Engine, Store};
use wasmtime_wasi::{WasiCtx, WasiCtxView, WasiView};
use wasmtime_wasi_http::WasiHttpCtx;
use wasmtime_wasi_http::p2::bindings::http::types::{ErrorCode, Scheme};
use wasmtime_wasi_http::p2::body::HyperOutgoingBody;
use wasmtime_wasi_http::p2::types::{HostFutureIncomingResponse, OutgoingRequestConfig};
use wasmtime_wasi_http::p2::{
    HttpResult, WasiHttpCtxView, WasiHttpHooks, WasiHttpView, default_send_request,
};

// Host-side bindings for the realtime world. Only the outgoing `ws` interface is
// implemented here; the wasi:* deps are aliased to wasmtime's existing generated
// bindings so the resource/type definitions unify with `add_to_linker_async`.
wasmtime::component::bindgen!({
    path: "wit",
    world: "realtime-backend",
    imports: { default: async | trappable },
    exports: { default: async },
    with: {
        "wasi:io": wasmtime_wasi::p2::bindings::io,
        "wasi:clocks": wasmtime_wasi::p2::bindings::clocks,
        "wasi:http": wasmtime_wasi_http::p2::bindings::http,
        "super-stt:realtime/ws.ws-stream": WsStreamResource,
        "super-stt:realtime/ws.consumer-stream": ConsumerStreamResource,
    },
});

pub use self::super_stt::realtime::ws::{CloseFrame, WsError, WsFrame};

/// Path to the prebuilt component (`just build-component`), or `None` if it
/// isn't built — tests skip gracefully in that case so a partial `cargo test`
/// (no wasm build) still passes; CI builds it first. Pair with `let-else`:
///
/// ```ignore
/// let Some(path) = common::component_path() else {
///     eprintln!("skipping: component not built (run `just build-component`)");
///     return;
/// };
/// ```
#[must_use]
pub fn component_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/wasm32-wasip2/release/super_stt_backend_mistral.wasm");
    p.exists().then_some(p)
}

// ── host state + egress allowlist (mirrors the daemon's wasm host) ──────────

struct Host {
    table: ResourceTable,
    wasi: WasiCtx,
    http: WasiHttpCtx,
    hooks: AllowlistHooks,
}

impl WasiView for Host {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi,
            table: &mut self.table,
        }
    }
}

impl WasiHttpView for Host {
    fn http(&mut self) -> WasiHttpCtxView<'_> {
        WasiHttpCtxView {
            ctx: &mut self.http,
            table: &mut self.table,
            hooks: &mut self.hooks,
        }
    }
}

struct AllowlistHooks {
    allowed_hosts: Vec<String>,
    allow_loopback: bool,
}

impl WasiHttpHooks for AllowlistHooks {
    fn send_request(
        &mut self,
        request: hyper::Request<HyperOutgoingBody>,
        config: OutgoingRequestConfig,
    ) -> HttpResult<HostFutureIncomingResponse> {
        let authority = request.uri().authority().map(|a| a.as_str().to_string());
        let host = request.uri().host().map(str::to_string);
        let allowed = self.allowed_hosts.iter().any(|a| {
            Some(a.as_str()) == authority.as_deref() || Some(a.as_str()) == host.as_deref()
        });
        if !allowed {
            return Err(ErrorCode::InternalError(Some(format!(
                "outbound host not allowed: {}",
                authority.or(host).unwrap_or_default()
            )))
            .into());
        }
        if let Some(h) = host.as_deref() {
            let port =
                request
                    .uri()
                    .port_u16()
                    .unwrap_or(if request.uri().scheme_str() == Some("http") {
                        80
                    } else {
                        443
                    });
            if let Err(msg) = guard_egress_host(h, port, self.allow_loopback) {
                return Err(ErrorCode::InternalError(Some(msg)).into());
            }
        }
        Ok(default_send_request(request, config))
    }
}

/// Confine an outbound connection to `allowed` + run the SSRF resolver guard.
/// Shared by the HTTP hook and the `ws` host so both transports enforce
/// identical egress rules (mirrors the daemon's `host::check_host_allowed`).
fn check_host_allowed(
    allowed: &[String],
    host: &str,
    port: u16,
    allow_loopback: bool,
) -> Result<(), String> {
    let authority = format!("{host}:{port}");
    let on_allowlist = allowed
        .iter()
        .any(|a| a.as_str() == host || a.as_str() == authority);
    if !on_allowlist {
        return Err(format!("outbound host not allowed: {host}"));
    }
    guard_egress_host(host, port, allow_loopback)
}

fn guard_egress_host(host: &str, port: u16, allow_loopback: bool) -> Result<(), String> {
    if let Ok(ip) = host.parse::<IpAddr>() {
        if allow_loopback && ip.is_loopback() {
            return Ok(());
        }
        if is_disallowed_ip(&ip) {
            return Err(format!("host {host} is a disallowed address {ip}"));
        }
        return Ok(());
    }
    match (host, port).to_socket_addrs() {
        Ok(addrs) => {
            for addr in addrs {
                let ip = addr.ip();
                if allow_loopback && ip.is_loopback() {
                    continue;
                }
                if is_disallowed_ip(&ip) {
                    return Err(format!("host {host} resolves to a disallowed address {ip}"));
                }
            }
            Ok(())
        }
        Err(_) => Err(format!("cannot resolve host {host}")),
    }
}

fn is_disallowed_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_disallowed_v4(*v4),
        IpAddr::V6(v6) => {
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return is_disallowed_v4(mapped);
            }
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_unique_local()
                || v6.is_unicast_link_local()
        }
    }
}

fn is_disallowed_v4(v4: Ipv4Addr) -> bool {
    v4.is_loopback()
        || v4.is_private()
        || v4.is_link_local()
        || v4.is_unspecified()
        || v4.is_broadcast()
}

// ── outgoing `super-stt:realtime/ws` host (mirrors the daemon's ws_host.rs) ──

/// A live outgoing WebSocket owned by the host. `None` once closed.
pub struct WsStreamResource {
    stream: Option<WebSocketStream<MaybeTlsStream<TcpStream>>>,
    /// One-frame lookahead. `wasi:io/poll` asks "is this readable?" without
    /// taking anything, but a `WebSocketStream` only answers by consuming, so
    /// [`Pollable::ready`] reads one frame and parks it here for the `recv`
    /// that follows. Readiness stays non-destructive from the guest's side:
    /// every frame `ready` observes is still delivered by `recv`.
    pending: Option<std::result::Result<WsFrame, WsError>>,
}

impl WsStreamResource {
    fn new(stream: WebSocketStream<MaybeTlsStream<TcpStream>>) -> Self {
        Self {
            stream: Some(stream),
            pending: None,
        }
    }

    /// The next frame the protocol exposes, skipping tungstenite's ping/pong.
    ///
    /// Cancel-safe, which `poll` requires: it races both `ready` futures and
    /// drops the loser.
    async fn next_frame(&mut self) -> std::result::Result<WsFrame, WsError> {
        let Some(stream) = self.stream.as_mut() else {
            return Err(WsError::Closed);
        };
        loop {
            match stream.next().await {
                Some(Ok(Message::Text(text))) => {
                    return Ok(WsFrame::Text(text.as_str().to_string()));
                }
                Some(Ok(Message::Binary(data))) => return Ok(WsFrame::Binary(data.into())),
                Some(Ok(Message::Close(frame))) => {
                    self.stream = None;
                    let close = frame.map_or(
                        CloseFrame {
                            code: 1005,
                            reason: String::new(),
                        },
                        |f| CloseFrame {
                            code: f.code.into(),
                            reason: f.reason.as_str().to_string(),
                        },
                    );
                    return Ok(WsFrame::Close(close));
                }
                Some(Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_))) => {}
                Some(Err(e)) => {
                    self.stream = None;
                    return Err(WsError::RecvFailed(format!("recv failed: {e}")));
                }
                None => {
                    self.stream = None;
                    return Err(WsError::Closed);
                }
            }
        }
    }
}

#[async_trait::async_trait]
impl wasmtime_wasi::p2::Pollable for WsStreamResource {
    async fn ready(&mut self) {
        if self.pending.is_none() {
            self.pending = Some(self.next_frame().await);
        }
    }
}

/// The host-side bridge between the test's consumer channels and the guest's
/// realtime session. The test owns the opposite ends: it sends frames into
/// `incoming` and reads frames the guest emits off `outgoing`.
pub struct ConsumerStreamTransport {
    /// Frames arriving FROM the consumer TO the guest.
    pub incoming: tokio::sync::mpsc::UnboundedReceiver<WsFrame>,
    /// Frames the guest sends OUT to the consumer.
    pub outgoing: tokio::sync::mpsc::UnboundedSender<WsFrame>,
}

/// A live consumer WebSocket the harness owns and hands to the guest's
/// `ws-server.handle`. `None` once the session is closed.
pub struct ConsumerStreamResource {
    transport: Option<ConsumerStreamTransport>,
    /// One-frame lookahead, for the same reason as [`WsStreamResource::pending`].
    pending: Option<std::result::Result<WsFrame, WsError>>,
}

impl ConsumerStreamResource {
    /// Wrap a live transport for handoff to the guest.
    pub fn new(t: ConsumerStreamTransport) -> Self {
        Self {
            transport: Some(t),
            pending: None,
        }
    }

    /// The next frame from the consumer, or `Closed` once the test drops its
    /// sender. Cancel-safe: `mpsc::Receiver::recv` leaves the queue untouched
    /// when its future is dropped before completing.
    async fn next_frame(&mut self) -> std::result::Result<WsFrame, WsError> {
        let Some(transport) = self.transport.as_mut() else {
            return Err(WsError::Closed);
        };
        let Some(frame) = transport.incoming.recv().await else {
            self.transport = None;
            return Err(WsError::Closed);
        };
        Ok(frame)
    }
}

#[async_trait::async_trait]
impl wasmtime_wasi::p2::Pollable for ConsumerStreamResource {
    async fn ready(&mut self) {
        if self.pending.is_none() {
            self.pending = Some(self.next_frame().await);
        }
    }
}

/// WebSocket handshake headers the host owns; a guest-supplied header with one
/// of these names is dropped so it cannot corrupt the upgrade request.
const RESERVED_HEADERS: &[&str] = &[
    "host",
    "connection",
    "upgrade",
    "sec-websocket-key",
    "sec-websocket-version",
];

impl self::super_stt::realtime::ws::Host for Host {
    async fn connect(
        &mut self,
        url: String,
        headers: Vec<(String, Vec<u8>)>,
    ) -> wasmtime::Result<std::result::Result<Resource<WsStreamResource>, WsError>> {
        let uri: Uri = match url.parse() {
            Ok(u) => u,
            Err(e) => return Ok(Err(WsError::InvalidUrl(format!("invalid url: {e}")))),
        };

        match uri.scheme_str() {
            Some("ws" | "wss") => {}
            other => {
                return Ok(Err(WsError::InvalidUrl(format!(
                    "scheme must be ws or wss, got {}",
                    other.unwrap_or("<none>")
                ))));
            }
        }

        let Some(host) = uri.host() else {
            return Ok(Err(WsError::InvalidUrl("url has no host".to_string())));
        };
        let port = uri.port_u16().unwrap_or(match uri.scheme_str() {
            Some("ws") => 80,
            _ => 443,
        });

        if let Err(msg) = check_host_allowed(
            &self.hooks.allowed_hosts,
            host,
            port,
            self.hooks.allow_loopback,
        ) {
            return Ok(Err(WsError::HostNotAllowed(msg)));
        }

        let mut request = match uri.into_client_request() {
            Ok(r) => r,
            Err(e) => return Ok(Err(WsError::InvalidUrl(format!("invalid url: {e}")))),
        };
        for (name, value) in headers {
            if RESERVED_HEADERS.contains(&name.to_ascii_lowercase().as_str()) {
                continue;
            }
            let header_name = match name.parse::<tokio_tungstenite::tungstenite::http::HeaderName>()
            {
                Ok(n) => n,
                Err(e) => {
                    return Ok(Err(WsError::ConnectFailed(format!(
                        "invalid header name {name}: {e}"
                    ))));
                }
            };
            let header_value =
                match tokio_tungstenite::tungstenite::http::HeaderValue::from_bytes(&value) {
                    Ok(v) => v,
                    Err(e) => {
                        return Ok(Err(WsError::ConnectFailed(format!(
                            "invalid header value for {name}: {e}"
                        ))));
                    }
                };
            request.headers_mut().append(header_name, header_value);
        }

        match connect_async(request).await {
            Ok((stream, _response)) => {
                let resource = self.table.push(WsStreamResource::new(stream))?;
                Ok(Ok(resource))
            }
            Err(e) => Ok(Err(WsError::ConnectFailed(format!("connect failed: {e}")))),
        }
    }
}

// Signatures here are dictated by wasmtime's generated trait, so the `async`
// cannot be dropped from the members that never await. `unknown_lints` keeps
// the attribute harmless on toolchains predating the lint.
#[allow(unknown_lints, clippy::unused_async_trait_impl)]
impl self::super_stt::realtime::ws::HostWsStream for Host {
    async fn send_text(
        &mut self,
        self_: Resource<WsStreamResource>,
        text: String,
    ) -> wasmtime::Result<std::result::Result<(), WsError>> {
        let res = self.table.get_mut(&self_)?;
        let Some(stream) = res.stream.as_mut() else {
            return Ok(Err(WsError::Closed));
        };
        match stream.send(Message::Text(text.into())).await {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(WsError::SendFailed(format!("send failed: {e}")))),
        }
    }

    async fn send_binary(
        &mut self,
        self_: Resource<WsStreamResource>,
        data: Vec<u8>,
    ) -> wasmtime::Result<std::result::Result<(), WsError>> {
        let res = self.table.get_mut(&self_)?;
        let Some(stream) = res.stream.as_mut() else {
            return Ok(Err(WsError::Closed));
        };
        match stream.send(Message::Binary(data.into())).await {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(WsError::SendFailed(format!("send failed: {e}")))),
        }
    }

    async fn recv(
        &mut self,
        self_: Resource<WsStreamResource>,
    ) -> wasmtime::Result<std::result::Result<WsFrame, WsError>> {
        let res = self.table.get_mut(&self_)?;
        // A frame `subscribe`/`ready` already pulled off the socket is handed
        // over before touching the stream again.
        if let Some(frame) = res.pending.take() {
            return Ok(frame);
        }
        Ok(res.next_frame().await)
    }

    async fn subscribe(
        &mut self,
        self_: Resource<WsStreamResource>,
    ) -> wasmtime::Result<Resource<wasmtime_wasi::p2::bindings::io::poll::Pollable>> {
        // Backed by the one-frame lookahead, exactly as the daemon's host does
        // it: the pollable resolves once a frame has been read, and `recv`
        // hands that same frame over. This is what lets the guest wait on its
        // consumer and its upstream at once.
        wasmtime_wasi::p2::subscribe(&mut self.table, self_)
    }

    async fn close(
        &mut self,
        self_: Resource<WsStreamResource>,
    ) -> wasmtime::Result<std::result::Result<(), WsError>> {
        let res = self.table.get_mut(&self_)?;
        if let Some(mut stream) = res.stream.take() {
            let _ = stream.close(None).await;
        }
        Ok(Ok(()))
    }

    async fn drop(&mut self, rep: Resource<WsStreamResource>) -> wasmtime::Result<()> {
        let _ = self.table.delete(rep)?;
        Ok(())
    }
}

// Signatures here are dictated by wasmtime's generated trait, so the `async`
// cannot be dropped from the members that never await. `unknown_lints` keeps
// the attribute harmless on toolchains predating the lint.
#[allow(unknown_lints, clippy::unused_async_trait_impl)]
impl self::super_stt::realtime::ws::HostConsumerStream for Host {
    async fn send_text(
        &mut self,
        self_: Resource<ConsumerStreamResource>,
        text: String,
    ) -> wasmtime::Result<std::result::Result<(), WsError>> {
        let res = self.table.get_mut(&self_)?;
        let Some(transport) = res.transport.as_mut() else {
            return Ok(Err(WsError::Closed));
        };
        match transport.outgoing.send(WsFrame::Text(text)) {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(WsError::SendFailed(format!("consumer gone: {e}")))),
        }
    }

    async fn send_binary(
        &mut self,
        self_: Resource<ConsumerStreamResource>,
        data: Vec<u8>,
    ) -> wasmtime::Result<std::result::Result<(), WsError>> {
        let res = self.table.get_mut(&self_)?;
        let Some(transport) = res.transport.as_mut() else {
            return Ok(Err(WsError::Closed));
        };
        match transport.outgoing.send(WsFrame::Binary(data)) {
            Ok(()) => Ok(Ok(())),
            Err(e) => Ok(Err(WsError::SendFailed(format!("consumer gone: {e}")))),
        }
    }

    async fn recv(
        &mut self,
        self_: Resource<ConsumerStreamResource>,
    ) -> wasmtime::Result<std::result::Result<WsFrame, WsError>> {
        let res = self.table.get_mut(&self_)?;
        // A frame `subscribe`/`ready` already took off the channel is handed
        // over before waiting on it again.
        if let Some(frame) = res.pending.take() {
            return Ok(frame);
        }
        Ok(res.next_frame().await)
    }

    async fn subscribe(
        &mut self,
        self_: Resource<ConsumerStreamResource>,
    ) -> wasmtime::Result<Resource<wasmtime_wasi::p2::bindings::io::poll::Pollable>> {
        // Backed by the same one-frame lookahead as `ws-stream::subscribe`.
        wasmtime_wasi::p2::subscribe(&mut self.table, self_)
    }

    async fn close(
        &mut self,
        self_: Resource<ConsumerStreamResource>,
    ) -> wasmtime::Result<std::result::Result<(), WsError>> {
        let res = self.table.get_mut(&self_)?;
        res.transport = None;
        Ok(Ok(()))
    }

    async fn drop(&mut self, rep: Resource<ConsumerStreamResource>) -> wasmtime::Result<()> {
        let _ = self.table.delete(rep)?;
        Ok(())
    }
}

// ── the backend driver ──────────────────────────────────────────────────────

/// A loaded Mistral realtime component, driven over the batch `/v1` contract
/// and the realtime `ws-server` session.
pub struct WasmBackend {
    engine: Engine,
    pre: RealtimeBackendPre<Host>,
    allowed_hosts: Vec<String>,
    allow_loopback: bool,
    transcribe_headers: Vec<(String, String)>,
    model_id: String,
}

impl WasmBackend {
    /// Load a realtime component the way the daemon does: secrets/options are
    /// the already-formed `x-stt-secret-*` / `x-stt-option-*` header pairs.
    ///
    /// # Errors
    /// Returns an error if the component cannot be loaded or linked.
    pub fn new_realtime(
        component_path: &std::path::Path,
        allowed_hosts: Vec<String>,
        model_id: String,
        transcribe_headers: Vec<(String, String)>,
    ) -> Result<Self> {
        let mut config = Config::new();
        config.wasm_component_model(true);
        let engine = Engine::new(&config)?;
        let component = Component::from_file(&engine, component_path)
            .map_err(|e| anyhow!("loading component {}: {e}", component_path.display()))?;
        let mut linker: Linker<Host> = Linker::new(&engine);
        wasmtime_wasi::p2::add_to_linker_async(&mut linker)?;
        wasmtime_wasi_http::p2::add_only_http_to_linker_async(&mut linker)?;
        self::super_stt::realtime::ws::add_to_linker::<Host, wasmtime::component::HasSelf<Host>>(
            &mut linker,
            |h| h,
        )?;
        let pre = RealtimeBackendPre::new(linker.instantiate_pre(&component)?)?;
        Ok(Self {
            engine,
            pre,
            allowed_hosts,
            allow_loopback: false,
            transcribe_headers,
            model_id,
        })
    }

    /// Permit egress to loopback (for a mock upstream bound to 127.0.0.1). Off
    /// by default — the SSRF guard blocks loopback for untrusted backends.
    #[must_use]
    pub fn permit_loopback_egress(mut self) -> Self {
        self.allow_loopback = true;
        self
    }

    fn new_host(&self) -> Host {
        Host {
            table: ResourceTable::new(),
            wasi: WasiCtx::builder().build(),
            http: WasiHttpCtx::new(),
            hooks: AllowlistHooks {
                allowed_hosts: self.allowed_hosts.clone(),
                allow_loopback: self.allow_loopback,
            },
        }
    }

    async fn invoke(
        &self,
        method: &str,
        path: &str,
        headers: &[(String, String)],
        body: Vec<u8>,
    ) -> Result<(u16, Vec<u8>)> {
        let mut store = Store::new(&self.engine, self.new_host());

        let mut builder = hyper::Request::builder()
            .method(method)
            .uri(format!("http://backend.local{path}"));
        for (key, value) in headers {
            builder = builder.header(key.as_str(), value.as_str());
        }
        let request = builder
            .body(
                http_body_util::Full::new(bytes::Bytes::from(body))
                    .map_err(|never: std::convert::Infallible| -> ErrorCode { match never {} }),
            )
            .context("building backend request")?;

        let (tx, rx) = tokio::sync::oneshot::channel();
        let req = store
            .data_mut()
            .http()
            .new_incoming_request(Scheme::Http, request)?;
        let out = store.data_mut().http().new_response_outparam(tx)?;
        // Both worlds export `wasi:http/incoming-handler`, so the batch `/v1`
        // path works for a realtime backend's non-realtime models too.
        let inst = self.pre.instantiate_async(&mut store).await?;
        inst.wasi_http_incoming_handler()
            .call_handle(&mut store, req, out)
            .await?;

        let response = rx
            .await
            .context("backend produced no response")?
            .map_err(|e| anyhow!("backend transport error: {e:?}"))?;
        let status = response.status().as_u16();
        let collected = response.into_body().collect().await?.to_bytes();
        Ok((status, collected.to_vec()))
    }

    /// `GET /v1/ping`.
    ///
    /// # Errors
    /// Returns an error if the component cannot be invoked or returns non-JSON.
    pub async fn ping(&self) -> Result<serde_json::Value> {
        let (_, body) = self.invoke("GET", "/v1/ping", &[], Vec::new()).await?;
        Ok(serde_json::from_slice(&body)?)
    }

    /// `GET /v1/status`.
    ///
    /// # Errors
    /// Returns an error if the component cannot be invoked or returns non-JSON.
    pub async fn status(&self) -> Result<serde_json::Value> {
        let (_, body) = self.invoke("GET", "/v1/status", &[], Vec::new()).await?;
        Ok(serde_json::from_slice(&body)?)
    }

    /// `POST /v1/transcribe`. Returns the transcription, or the backend's own
    /// error message on a non-200 response (mirrors the daemon).
    ///
    /// # Errors
    /// Returns an error if the component cannot be invoked or the backend
    /// reports a non-success status.
    pub async fn transcribe_audio(&mut self, audio: &[f32], sample_rate: u32) -> Result<String> {
        self.transcribe_with_language(audio, sample_rate, None)
            .await
    }

    /// Like [`Self::transcribe_audio`], with an explicit `language` in the
    /// request body — a BCP-47 tag or the reserved `auto`.
    ///
    /// # Errors
    /// Returns an error if the component cannot be invoked or the backend
    /// reports a non-success status.
    pub async fn transcribe_with_language(
        &mut self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String> {
        let mut payload = serde_json::json!({
            "audio_data": audio,
            "sample_rate": sample_rate,
        });
        if let Some(lang) = language {
            payload["language"] = serde_json::Value::String(lang.to_string());
        }
        let body = serde_json::to_vec(&payload)?;
        let mut headers = self.transcribe_headers.clone();
        headers.push(("x-stt-model".to_string(), self.model_id.clone()));
        let (status, resp) = self
            .invoke("POST", "/v1/transcribe", &headers, body)
            .await?;
        let json: serde_json::Value =
            serde_json::from_slice(&resp).context("parsing backend transcribe response")?;
        if status == 200 {
            json["transcription"]
                .as_str()
                .map(String::from)
                .ok_or_else(|| anyhow!("backend response missing transcription"))
        } else {
            let msg = json
                .get("detail")
                .and_then(|v| v.as_str())
                .or_else(|| json.get("message").and_then(|v| v.as_str()))
                .unwrap_or("transcription failed");
            bail!("{msg}");
        }
    }

    /// Run one consumer realtime session: instantiate the component and invoke
    /// its `super-stt:realtime/ws-server.handle` export with the injected
    /// headers and a host-owned consumer stream. Returns when the guest's
    /// handler returns.
    ///
    /// # Errors
    /// Returns an error if instantiation fails or the guest's handler returns a
    /// `ws-error`.
    pub async fn realtime_session(&self, transport: ConsumerStreamTransport) -> Result<()> {
        let mut store = Store::new(&self.engine, self.new_host());
        let mut headers: Vec<(String, Vec<u8>)> = self
            .transcribe_headers
            .iter()
            .map(|(k, v)| (k.clone(), v.clone().into_bytes()))
            .collect();
        headers.push((
            "x-stt-model".to_string(),
            self.model_id.clone().into_bytes(),
        ));
        let consumer = store
            .data_mut()
            .table
            .push(ConsumerStreamResource::new(transport))?;
        let inst = self.pre.instantiate_async(&mut store).await?;
        inst.super_stt_realtime_ws_server()
            .call_handle(&mut store, &headers, consumer)
            .await?
            .map_err(|e| anyhow!("ws-server.handle returned error: {e:?}"))
    }
}
