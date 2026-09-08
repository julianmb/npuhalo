#!/usr/bin/env python3
"""
grammar_compiler.py — Compile the agent tool set into a GBNF grammar that
forces every generation to end in a well-formed Qwen-native <tool_call>.

Structure: bounded tag-free prose, then mandatory tool-call structure.
The sampler cannot emit EOS inside the prose region, which eliminates the
missing-call failure mode (model rambles until max_tokens without acting).
"""

KNOWN_TOOLS = {
    "shell": {"required": ["command"]},
    "read": {"required": ["path"]},
    "write": {"required": ["path", "content"]},
    "test": {"required": []},
}

# value: anything except "</" (so </parameter> terminates); "<" alone allowed
_VALUE_RULE = 'value ::= ( [^<] | "<" [^/] )*\n'
_PROSE_BOUND = 400  # chars of tag-free reasoning allowed before the call


def compile_tool_call_grammar(tools=None, prose_bound: int = _PROSE_BOUND) -> str:
    """Build a GBNF grammar forcing: bounded prose, then one complete tool call."""
    if tools is None:
        tools = list(KNOWN_TOOLS)
    unknown = [t for t in tools if t not in KNOWN_TOOLS]
    if unknown:
        raise ValueError(f"unknown tools: {unknown}")
    names = ' | '.join(f'"{t}"' for t in tools)
    return f'''root ::= prose toolcall
prose ::= [^<]{{0,{prose_bound}}}
toolcall ::= "<tool_call>\\n<function=" name ">\\n" params "</function>\\n</tool_call>"
name ::= {names}
params ::= param*
param ::= "<parameter=" key ">" value "</parameter>\\n"
key ::= [a-zA-Z_]+
{_VALUE_RULE}'''


if __name__ == "__main__":
    print(compile_tool_call_grammar())
