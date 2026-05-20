# SPDX-License-Identifier: Apache-2.0
"""
Local PD (Prefill-Decode) Separation Proxy for DyCP.

This proxy sits in front of a single vLLM instance with DyCP enabled.
For each incoming request, it:
1. Sends a prefill-only request (max_tokens=1) to vLLM with
   kv_transfer_params indicating KV cache should be saved.
2. Waits for prefill completion and extracts kv_transfer_params from the
   response.
3. Sends a decode request to the same vLLM instance with the original
   max_tokens and kv_transfer_params from the prefill response.
4. Streams the decode response back to the client.

Usage:
    python -m dycp.proxy.local_pd_proxy \
        --vllm-url http://localhost:8000 \
        --port 9000
"""

import argparse
import copy
import json
import logging
import os
import sys
import uuid

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# Use vllm logger when available, but add a dedicated handler so proxy
# logs are always visible regardless of how uvicorn / vllm configure the
# global logging state.
try:
    from vllm.logger import init_logger

    logger = init_logger("vllm.dycp.proxy.local_pd_proxy")
except ImportError:
    logger = logging.getLogger(__name__)

# Ensure proxy logs go to stderr with a predictable format.  vLLM's
# dictConfig only touches the "vllm" logger and uvicorn may reconfigure
# root handlers later, so we attach our own handler unconditionally.
if not logger.handlers:
    _proxy_handler = logging.StreamHandler()
    _proxy_handler.setFormatter(logging.Formatter(
        fmt="%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%m-%d %H:%M:%S",
    ))
    _proxy_handler.setLevel(logging.DEBUG)
    logger.addHandler(_proxy_handler)
    logger.setLevel(logging.DEBUG)
    # Prevent messages from being discarded by a no-handler root.
    logger.propagate = False

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)  # 6 hours

app = FastAPI(title="DyCP Local PD Proxy")


# ---------------------------------------------------------------------------
# Global state (set from CLI args at startup)
# ---------------------------------------------------------------------------
class ProxyConfig:
    vllm_url: str = "http://localhost:8000"
    api_key: str | None = None
    _session: aiohttp.ClientSession | None = None

    async def get_session(self) -> aiohttp.ClientSession:
        """Reuse a single aiohttp session for connection pooling."""
        if self._session is None or self._session.closed:
            conn = aiohttp.TCPConnector(limit=100, keepalive_timeout=30)
            self._session = aiohttp.ClientSession(
                timeout=AIOHTTP_TIMEOUT, connector=conn
            )
        return self._session


proxy_config = ProxyConfig()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_request_prefix() -> str:
    """Generate a shared prefix for correlating prefill/decode requests."""
    return f"pd-{uuid.uuid4().hex[:12]}"


def _auth_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if proxy_config.api_key:
        headers["Authorization"] = f"Bearer {proxy_config.api_key}"
    return headers


def _estimate_token_count(body: dict) -> int:
    """
    Estimate the number of prompt tokens from the request body.

    For /v1/completions: body["prompt"] is a string or list of token IDs.
    For /v1/chat/completions: body["messages"] is a list of message dicts.

    Uses a simple heuristic: character count / 4 for strings, or len() for
    token ID lists.
    """
    prompt = body.get("prompt")
    if prompt is not None:
        if isinstance(prompt, str):
            return max(len(prompt) // 4, 1)
        if isinstance(prompt, list):
            # List of token IDs or list of strings
            if prompt and isinstance(prompt[0], int):
                return len(prompt)
            # List of strings — sum character counts
            return max(sum(len(s) for s in prompt) // 4, 1)

    messages = body.get("messages")
    if messages is not None:
        total_chars = sum(
            len(msg.get("content", ""))
            for msg in messages
            if isinstance(msg.get("content"), str)
        )
        return max(total_chars // 4, 1)

    return 0


def _get_long_request_threshold() -> int:
    """Read the token threshold from env var VLLM_LONG_REQUEST_THRESHOLD."""
    return int(os.environ.get("VLLM_LONG_REQUEST_THRESHOLD", "100"))


# ---------------------------------------------------------------------------
# Direct forwarding (short requests)
# ---------------------------------------------------------------------------
async def _forward_direct(
    endpoint: str,
    body: dict,
    request_id: str,
) -> StreamingResponse | JSONResponse:
    """
    Forward a short request directly to vLLM without PD separation.
    Supports both streaming and non-streaming modes.
    """
    session = await proxy_config.get_session()
    headers = _auth_headers()
    headers["X-Request-Id"] = request_id
    is_streaming = body.get("stream", False)

    if is_streaming:
        async def stream_generator():
            try:
                async with session.post(
                    f"{proxy_config.vllm_url}{endpoint}",
                    json=body,
                    headers=headers,
                ) as resp:
                    if resp.status >= 400:
                        error_body = await resp.read()
                        logger.error(
                            "Direct forward failed [%s] status=%d: %s",
                            request_id, resp.status,
                            error_body.decode(errors="replace"),
                        )
                        yield error_body
                        return
                    async for chunk in resp.content.iter_chunked(1024):
                        yield chunk
            except aiohttp.ClientError as exc:
                logger.error(
                    "Direct forward connection error [%s]: %s",
                    request_id, exc,
                )
                error_payload = json.dumps({
                    "error": {
                        "message": f"Direct forward connection error: {exc}",
                        "type": "proxy_error",
                        "code": 502,
                    }
                }).encode()
                yield error_payload

        return StreamingResponse(
            stream_generator(), media_type="text/event-stream",
        )
    else:
        try:
            async with session.post(
                f"{proxy_config.vllm_url}{endpoint}",
                json=body,
                headers=headers,
            ) as resp:
                resp_body = await resp.read()
                if resp.status >= 400:
                    logger.error(
                        "Direct forward failed [%s] status=%d: %s",
                        request_id, resp.status,
                        resp_body.decode(errors="replace"),
                    )
                return JSONResponse(
                    status_code=resp.status,
                    content=json.loads(resp_body),
                )
        except aiohttp.ClientError as exc:
            logger.error(
                "Direct forward connection error [%s]: %s",
                request_id, exc,
            )
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"Direct forward connection error: {exc}",
                        "type": "proxy_error",
                        "code": 502,
                    }
                },
            )


def _build_prefill_request(
    original: dict,
    prefix: str,
) -> dict:
    """
    Build the prefill request from the original client request.

    * max_tokens is forced to 1 (we only need the KV cache, not generation).
    * stream is forced to False so we can read the full response.
    * kv_transfer_params tells vLLM to save the KV cache for later decode.
    """
    req = copy.deepcopy(original)
    req["max_tokens"] = 1
    if "max_completion_tokens" in req:
        req["max_completion_tokens"] = 1
    req["stream"] = False
    # Remove stream_options to avoid errors when stream=False
    req.pop("stream_options", None)
    req["kv_transfer_params"] = {
        "do_remote_prefill": False,
        "do_remote_decode": True,
        "pd_request_prefix": prefix,
    }
    return req


def _build_decode_request(
    original: dict,
    prefix: str,
    prefill_kv_params: dict | None,
) -> dict:
    """
    Build the decode request from the original client request.

    * Keeps the original max_tokens / stream settings.
    * kv_transfer_params tells vLLM to load KV cache from the prefill step.
    * Any extra params returned by prefill (e.g. block ids, engine id) are
      merged into kv_transfer_params.
    """
    req = copy.deepcopy(original)
    kv_params: dict = {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "pd_request_prefix": prefix,
    }
    # Merge any params that the prefill response returned (block ids, etc.)
    if prefill_kv_params:
        for key, value in prefill_kv_params.items():
            if key not in kv_params:
                kv_params[key] = value

        # Optimization: use token IDs from prefill to skip re-tokenization.
        # Store token IDs for endpoint switching in _handle_pd_request.
        prompt_token_ids = prefill_kv_params.get("prompt_token_ids")
        if prompt_token_ids:
            req["_use_token_ids"] = True
            req["prompt"] = prompt_token_ids
            req.pop("messages", None)

    req["kv_transfer_params"] = kv_params
    return req


# ---------------------------------------------------------------------------
# Response format conversion (completion → chat)
# ---------------------------------------------------------------------------
def _convert_completion_to_chat(resp: dict) -> dict:
    """Convert a /v1/completions response to /v1/chat/completions format."""
    resp["object"] = "chat.completion"
    for choice in resp.get("choices", []):
        text = choice.pop("text", "")
        choice["message"] = {"role": "assistant", "content": text}
    return resp


def _convert_completion_chunk_to_chat(chunk: dict, is_first: bool) -> dict:
    """Convert a /v1/completions streaming chunk to chat format."""
    chunk["object"] = "chat.completion.chunk"
    for choice in chunk.get("choices", []):
        text = choice.pop("text", "")
        delta = {}
        if is_first:
            delta["role"] = "assistant"
        if text:
            delta["content"] = text
        choice["delta"] = delta
    return chunk


async def _convert_streaming_completion_to_chat(raw_chunks):
    """Convert a streaming completion SSE response to chat SSE format.

    Parses data lines from the SSE stream, converts each completion chunk
    to chat completion chunk format, and re-emits as SSE.
    """
    line_buffer = b""
    first_chunk = True

    async for raw in raw_chunks:
        line_buffer += raw
        while b"\n" in line_buffer:
            line, line_buffer = line_buffer.split(b"\n", 1)
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(b"data: "):
                payload = stripped[6:]
                if payload == b"[DONE]":
                    yield b"data: [DONE]\n\n"
                    continue
                try:
                    chunk_data = json.loads(payload)
                    _convert_completion_chunk_to_chat(chunk_data, first_chunk)
                    first_chunk = False
                    yield f"data: {json.dumps(chunk_data)}\n\n".encode()
                except (json.JSONDecodeError, KeyError):
                    yield line + b"\n"
            else:
                yield line + b"\n"

    remaining = line_buffer.strip()
    if remaining:
        if remaining.startswith(b"data: "):
            payload = remaining[6:]
            if payload == b"[DONE]":
                yield b"data: [DONE]\n\n"
            else:
                try:
                    chunk_data = json.loads(payload)
                    _convert_completion_chunk_to_chat(chunk_data, first_chunk)
                    yield f"data: {json.dumps(chunk_data)}\n\n".encode()
                except (json.JSONDecodeError, KeyError):
                    yield remaining + b"\n"
        else:
            yield remaining + b"\n"


# ---------------------------------------------------------------------------
# Core PD flow
# ---------------------------------------------------------------------------
async def _do_prefill(
    session: aiohttp.ClientSession,
    endpoint: str,
    prefill_req: dict,
    request_id: str,
) -> dict:
    """
    Send the prefill request and return the parsed JSON response.
    Raises on HTTP or connection errors.
    """
    headers = _auth_headers()
    headers["X-Request-Id"] = request_id

    async with session.post(
        f"{proxy_config.vllm_url}{endpoint}",
        json=prefill_req,
        headers=headers,
    ) as resp:
        body = await resp.read()
        if resp.status >= 400:
            detail = body.decode(errors="replace")
            logger.error(
                "Prefill failed [%s] status=%d: %s",
                request_id, resp.status, detail,
            )
            raise aiohttp.ClientResponseError(
                request_info=resp.request_info,
                history=resp.history,
                status=resp.status,
                message=detail,
            )
        return json.loads(body)


async def _stream_decode(
    session: aiohttp.ClientSession,
    endpoint: str,
    decode_req: dict,
    request_id: str,
):
    """
    Send the decode request and yield response chunks.
    Works for both streaming (SSE) and non-streaming responses.
    """
    headers = _auth_headers()
    headers["X-Request-Id"] = request_id

    async with session.post(
        f"{proxy_config.vllm_url}{endpoint}",
        json=decode_req,
        headers=headers,
    ) as resp:
        if resp.status >= 400:
            error_body = await resp.read()
            detail = error_body.decode(errors="replace")
            logger.error(
                "Decode failed [%s] status=%d: %s",
                request_id, resp.status, detail,
            )
            # Yield the error so the client sees it
            yield error_body
            return
        async for chunk in resp.content.iter_chunked(1024):
            yield chunk


async def _handle_pd_request(endpoint: str, request: Request):
    """
    Unified handler for both /v1/completions and /v1/chat/completions.
    Orchestrates the prefill -> decode flow.
    """
    original_body = await request.json()
    prefix = _make_request_prefix()
    prefill_request_id = f"prefill-{prefix}"
    decode_request_id = f"decode-{prefix}"

    logger.info(
        "New PD request [%s] endpoint=%s model=%s",
        prefix, endpoint, original_body.get("model", "unknown"),
    )

    # --- Phase 1: Prefill ---
    prefill_req = _build_prefill_request(original_body, prefix)

    try:
        session = await proxy_config.get_session()
        prefill_resp = await _do_prefill(
            session, endpoint, prefill_req, prefill_request_id,
        )
    except aiohttp.ClientResponseError as exc:
        logger.error("Prefill request failed [%s]: %s", prefix, exc.message)
        return JSONResponse(
            status_code=exc.status,
            content={
                "error": {
                    "message": f"Prefill failed: {exc.message}",
                    "type": "proxy_error",
                    "code": exc.status,
                }
            },
        )
    except aiohttp.ClientError as exc:
        logger.error("Prefill connection error [%s]: %s", prefix, exc)
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"Prefill connection error: {exc}",
                    "type": "proxy_error",
                    "code": 502,
                }
            },
        )

    # Extract kv_transfer_params from prefill response
    prefill_kv_params = prefill_resp.get("kv_transfer_params")
    logger.info(
        "Prefill completed [%s] kv_transfer_params=%s",
        prefix, prefill_kv_params,
    )

    # --- Phase 2: Decode ---
    decode_req = _build_decode_request(original_body, prefix, prefill_kv_params)
    is_streaming = original_body.get("stream", False)

    # Use /v1/completions for decode when token IDs available (skip tokenization)
    decode_endpoint = endpoint
    if decode_req.pop("_use_token_ids", False):
        decode_endpoint = "/v1/completions"

    needs_chat_conversion = (
        endpoint == "/v1/chat/completions"
        and decode_endpoint != endpoint
    )

    logger.info(
        "Starting decode [%s] endpoint=%s streaming=%s chat_conversion=%s",
        prefix, decode_endpoint, is_streaming, needs_chat_conversion,
    )

    async def generate():
        try:
            session = await proxy_config.get_session()
            raw_chunks = _stream_decode(
                session, decode_endpoint, decode_req, decode_request_id,
            )
            if needs_chat_conversion and not is_streaming:
                body_parts = []
                async for chunk in raw_chunks:
                    body_parts.append(chunk)
                full_body = b"".join(body_parts)
                resp_data = json.loads(full_body)
                _convert_completion_to_chat(resp_data)
                yield json.dumps(resp_data).encode()
            elif needs_chat_conversion and is_streaming:
                async for converted in _convert_streaming_completion_to_chat(
                    raw_chunks,
                ):
                    yield converted
            else:
                async for chunk in raw_chunks:
                    yield chunk
        except aiohttp.ClientError as exc:
            logger.error("Decode connection error [%s]: %s", prefix, exc)
            error_payload = json.dumps({
                "error": {
                    "message": f"Decode connection error: {exc}",
                    "type": "proxy_error",
                    "code": 502,
                }
            }).encode()
            yield error_payload

    media_type = "text/event-stream" if is_streaming else "application/json"
    return StreamingResponse(generate(), media_type=media_type)


# ---------------------------------------------------------------------------
# Request dispatcher (short vs long)
# ---------------------------------------------------------------------------
async def _dispatch_request(endpoint: str, request: Request):
    """
    Route incoming requests based on estimated prompt length.

    Short requests (estimated tokens < threshold) are forwarded directly to
    vLLM without PD separation.  Long requests go through the two-phase
    prefill-then-decode flow.
    """
    body = await request.json()
    estimated_tokens = _estimate_token_count(body)
    threshold = _get_long_request_threshold()

    if estimated_tokens < threshold:
        request_id = f"direct-{uuid.uuid4().hex[:12]}"
        logger.info(
            "Short request [%s] endpoint=%s model=%s "
            "estimated_tokens=%d (< threshold=%d) -> direct forward",
            request_id, endpoint, body.get("model", "unknown"),
            estimated_tokens, threshold,
        )
        return await _forward_direct(endpoint, body, request_id)
    else:
        logger.info(
            "Long request endpoint=%s model=%s "
            "estimated_tokens=%d (>= threshold=%d) -> PD two-phase",
            endpoint, body.get("model", "unknown"),
            estimated_tokens, threshold,
        )
        return await _handle_pd_request_with_body(endpoint, body)


async def _handle_pd_request_with_body(endpoint: str, original_body: dict):
    """
    Two-phase PD flow, accepting a pre-parsed request body.
    Extracted from _handle_pd_request to allow _dispatch_request to parse
    the body once and pass it in.
    """
    prefix = _make_request_prefix()
    prefill_request_id = f"prefill-{prefix}"
    decode_request_id = f"decode-{prefix}"

    logger.info(
        "New PD request [%s] endpoint=%s model=%s",
        prefix, endpoint, original_body.get("model", "unknown"),
    )

    # --- Phase 1: Prefill ---
    prefill_req = _build_prefill_request(original_body, prefix)

    try:
        session = await proxy_config.get_session()
        prefill_resp = await _do_prefill(
            session, endpoint, prefill_req, prefill_request_id,
        )
    except aiohttp.ClientResponseError as exc:
        logger.error("Prefill request failed [%s]: %s", prefix, exc.message)
        return JSONResponse(
            status_code=exc.status,
            content={
                "error": {
                    "message": f"Prefill failed: {exc.message}",
                    "type": "proxy_error",
                    "code": exc.status,
                }
            },
        )
    except aiohttp.ClientError as exc:
        logger.error("Prefill connection error [%s]: %s", prefix, exc)
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"Prefill connection error: {exc}",
                    "type": "proxy_error",
                    "code": 502,
                }
            },
        )

    # Extract kv_transfer_params from prefill response
    prefill_kv_params = prefill_resp.get("kv_transfer_params")
    logger.info(
        "Prefill completed [%s] kv_transfer_params=%s",
        prefix, prefill_kv_params,
    )

    # --- Phase 2: Decode ---
    decode_req = _build_decode_request(
        original_body, prefix, prefill_kv_params,
    )
    is_streaming = original_body.get("stream", False)

    # Use /v1/completions for decode when token IDs available
    decode_endpoint = endpoint
    if decode_req.pop("_use_token_ids", False):
        decode_endpoint = "/v1/completions"

    # When original request was chat but decode uses completions endpoint,
    # we need to convert the response format back.
    needs_chat_conversion = (
        endpoint == "/v1/chat/completions"
        and decode_endpoint != endpoint
    )

    logger.info(
        "Starting decode [%s] endpoint=%s streaming=%s chat_conversion=%s",
        prefix, decode_endpoint, is_streaming, needs_chat_conversion,
    )

    async def generate():
        try:
            session = await proxy_config.get_session()
            raw_chunks = _stream_decode(
                session, decode_endpoint, decode_req, decode_request_id,
            )
            if needs_chat_conversion and not is_streaming:
                # Non-streaming: collect full response, convert, yield
                body_parts = []
                async for chunk in raw_chunks:
                    body_parts.append(chunk)
                full_body = b"".join(body_parts)
                resp_data = json.loads(full_body)
                _convert_completion_to_chat(resp_data)
                yield json.dumps(resp_data).encode()
            elif needs_chat_conversion and is_streaming:
                # Streaming: convert SSE chunks
                async for converted in _convert_streaming_completion_to_chat(
                    raw_chunks,
                ):
                    yield converted
            else:
                async for chunk in raw_chunks:
                    yield chunk
        except aiohttp.ClientError as exc:
            logger.error("Decode connection error [%s]: %s", prefix, exc)
            error_payload = json.dumps({
                "error": {
                    "message": f"Decode connection error: {exc}",
                    "type": "proxy_error",
                    "code": 502,
                }
            }).encode()
            yield error_payload

    media_type = "text/event-stream" if is_streaming else "application/json"
    return StreamingResponse(generate(), media_type=media_type)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/v1/completions")
async def create_completion(request: Request):
    return await _dispatch_request("/v1/completions", request)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    return await _dispatch_request("/v1/chat/completions", request)


@app.get("/health")
async def health():
    return JSONResponse(content={"status": "ok"})


@app.get("/v1/models")
async def proxy_models():
    """Forward /v1/models to vLLM so clients can discover the served model."""
    try:
        session = await proxy_config.get_session()
        headers = _auth_headers()
        async with session.get(
            f"{proxy_config.vllm_url}/v1/models", headers=headers,
        ) as resp:
                body = await resp.read()
                return JSONResponse(
                    status_code=resp.status,
                    content=json.loads(body),
                )
    except aiohttp.ClientError as exc:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(exc), "type": "proxy_error"}},
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="DyCP Local PD (Prefill-Decode) Separation Proxy",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the vLLM instance (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Port for the proxy server (default: 9000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the proxy server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key for authenticating with vLLM",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    proxy_config.vllm_url = args.vllm_url.rstrip("/")
    proxy_config.api_key = args.api_key

    logger.info(
        "Starting DyCP Local PD Proxy: proxy=%s:%d -> vllm=%s",
        args.host, args.port, proxy_config.vllm_url,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
