#!/usr/bin/env python3
"""
toolcall_parser.py — Incremental deterministic parser for the Qwen-native
<tool_call> format used by Ornith-1.5 in the agent loop.

Format (from TOOL_DESCRIPTIONS):
    <tool_call>
    <function=NAME>
    <parameter=KEY>VALUE</parameter>
    ...
    </function>
    </tool_call>

Classification at every feed() position:
  VALID_SO_FAR  — prefix is a complete, schema-valid call
  RECOVERABLE   — incomplete but consistent with a well-formed call in progress
  UNRECOVERABLE — structurally broken beyond repair (deterministic)

Zero-false-reject property: any string that the reference parse_tool_call()
accepts must end in VALID_SO_FAR. Validated by replay over the full recorded
corpus before any live use.
"""

import re
from enum import Enum

KNOWN_TOOLS = {"shell", "read", "write", "test"}
# tools with required parameters (test takes none)
REQUIRED_PARAMS = {"shell": {"command"}, "read": {"path"}, "write": {"path", "content"}, "test": set()}


class State(Enum):
    VALID_SO_FAR = "VALID_SO_FAR"
    RECOVERABLE = "RECOVERABLE"
    UNRECOVERABLE = "UNRECOVERABLE"


class ParseError(Exception):
    def __init__(self, position: int, reason: str):
        self.position = position
        self.reason = reason
        super().__init__(f"pos {position}: {reason}")


class IncrementalToolCallParser:
    """Feed text incrementally; classify after each feed."""

    def __init__(self):
        self.buf = ""
        self.pos = 0
        self.state = State.RECOVERABLE
        self.error = None
        self.finished = False  # True once a complete call has been consumed

    # ---- public API -------------------------------------------------------
    def feed(self, text: str) -> State:
        if self.finished:
            return self.state
        self.buf += text
        try:
            self._scan()
        except ParseError as e:
            self.state = State.UNRECOVERABLE
            self.error = {"position": e.position, "reason": e.reason}
            self.finished = True
        return self.state

    def result(self) -> dict:
        return {
            "state": self.state.value,
            "error": self.error,
            "call": self._parsed_call(),
            "consumed_len": self.pos,
        }

    # ---- internals --------------------------------------------------------
    def _fail(self, reason: str):
        raise ParseError(self.pos, reason)

    @staticmethod
    def _strip_prologue(buf: str):
        """Return (index_after_tool_call_tag, None) or (None, partial_ok)."""
        idx = buf.find("<tool_call>")
        if idx != -1:
            return idx + len("<tool_call>"), None
        # could the tail be a partial "<tool_call>"?
        for k in range(1, len("<tool_call>")):
            if buf.endswith("<tool_call>"[:k]):
                return None, True
        return None, False

    def _scan(self):
        buf = self.buf
        start, partial = self._strip_prologue(buf)
        if start is None:
            # No <tool_call> yet. Prose before the call is allowed; only fail on
            # an impossible construct: another tag family opening a call.
            if re.search(r"<function=", buf) or re.search(r"</tool_call>", buf):
                self._fail("function/closing tag before <tool_call>")
            self.pos = max(0, len(buf) - (len("<tool_call>") - 1 if partial else 0))
            self.state = State.RECOVERABLE
            return

        body = buf[start:]
        # function open tag
        m = re.match(r"\s*<function=([A-Za-z_][A-Za-z0-9_]*)>", body)
        if m:
            name = m.group(1)
            if name not in KNOWN_TOOLS:
                self._fail(f"unknown function name '{name}'")
            self._scan_params(body[m.end():], name, offset=start + m.end())
            return
        # partial function tag?
        m_part = re.match(r"\s*<function=([A-Za-z_][A-Za-z0-9_]*)?$", body.strip("\n"))
        if m_part or re.search(r"<function=[A-Za-z0-9_]*$", body):
            nm = re.search(r"<function=([A-Za-z0-9_]*)$", body)
            if nm and nm.group(1) and nm.group(1) not in KNOWN_TOOLS and len(nm.group(1)) >= 3:
                # long prefix that can no longer match a known tool
                known_prefixes = [t[:len(nm.group(1))] for t in KNOWN_TOOLS]
                if nm.group(1) not in known_prefixes:
                    self._fail(f"unknown function name prefix '{nm.group(1)}'")
            self.pos = len(buf)
            self.state = State.RECOVERABLE
            return
        # closing tag without function open?
        if re.match(r"\s*</(function|tool_call)>", body):
            self._fail("closing tag without complete function block")
        self.pos = len(buf)
        self.state = State.RECOVERABLE

    def _scan_params(self, body: str, name: str, offset: int):
        pos = 0
        params = {}
        n = len(body)
        while pos < n:
            # closing function tag?
            m_close = re.match(r"\s*</function>", body[pos:])
            if m_close:
                end = pos + m_close.end()
                # optional whitespace then </tool_call>
                m_tc = re.match(r"\s*</tool_call>", body[end:])
                # Schema mismatches (unknown/missing params) are recorded as
                # warnings, NOT parse failures — the reference parser accepts
                # them and the tool layer owns argument errors. Replay proved
                # schema-strict rejection produces false rejects.
                missing = REQUIRED_PARAMS[name] - set(params)
                warnings = [f"missing_required_params:{sorted(missing)}"] if missing else []
                self._final_call = {"name": name, "arguments": params,
                                    "schema_warnings": warnings}
                if m_tc:
                    self.pos = offset + end + m_tc.end()
                    self.state = State.VALID_SO_FAR
                    self.finished = True
                    return
                # function closed but tool_call not yet closed
                rest = body[end:]
                stripped_rest = rest.lstrip()
                LEGAL_TAIL = ("</tool_call>",)
                if (stripped_rest == ""
                        or len(rest) < len("</tool_call>")
                        or any(tok.startswith(stripped_rest) for tok in LEGAL_TAIL)):
                    self.pos = offset + len(body)
                    self.state = State.RECOVERABLE
                    return
                self._fail("garbage between </function> and </tool_call>")
            # parameter open?
            m_param = re.match(r"\s*<parameter=([^>]+)>", body[pos:])
            if m_param:
                key = m_param.group(1).strip()
                vstart = pos + m_param.end()
                vend = body.find("</parameter>", vstart)
                if vend == -1:
                    # value still streaming — recoverable
                    self.pos = offset + len(body)
                    self.state = State.RECOVERABLE
                    return
                params[key] = body[vstart:vend]
                pos = vend + len("</parameter>")
                continue
            # partial parameter tag at end?
            tail = body[pos:]
            stripped = tail.lstrip()
            LEGAL_CONTINUATIONS = ("<parameter=", "</parameter>", "</function>", "</tool_call>")
            if stripped == "" or any(tok.startswith(stripped) for tok in LEGAL_CONTINUATIONS):
                # pure whitespace or a prefix of a legal next token: keep waiting
                self.pos = offset + len(body)
                self.state = State.RECOVERABLE
                return
            if stripped.startswith("<parameter="):
                # key still streaming (no '>' yet): always recoverable
                self.pos = offset + len(body)
                self.state = State.RECOVERABLE
                return
            # whitespace between tags is tolerated
            if re.match(r"\s+$", tail) or tail == "":
                self.pos = offset + len(body)
                self.state = State.RECOVERABLE
                return
            self._fail(f"unexpected content in function body: {tail[:40]!r}")
        # ran out of input mid-structure
        self.pos = offset + len(body)
        self.state = State.RECOVERABLE

    def _parsed_call(self):
        return getattr(self, "_final_call", None)


def classify_stream(chunks) -> dict:
    """Convenience: feed an iterable of chunks, return final classification."""
    p = IncrementalToolCallParser()
    first_unrecoverable = None
    for i, ch in enumerate(chunks):
        st = p.feed(ch)
        if st == State.UNRECOVERABLE and first_unrecoverable is None:
            first_unrecoverable = i
    res = p.result()
    res["first_unrecoverable_chunk"] = first_unrecoverable
    return res
