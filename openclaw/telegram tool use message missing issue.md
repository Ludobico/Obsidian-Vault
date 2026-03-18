This is a regression in Telegram streaming with verbose mode - text disappears during tool calls.

**Root cause:**

- When tool call starts, previously streamed text is cleared/overwritten
- Text reappears later but out of context
- Related to [[Telegram] Streaming with extended thinking overwrites previous message text #17935](https://github.com/openclaw/openclaw/issues/17935) which was fixed in v2026.2.15

**The issue:**

1. Agent streams text
2. Tool call starts → text disappears
3. Tool output appears
4. More text → visible briefly
5. Next tool call → text disappears again
6. After all tools, text reappears but out of order

**Fix:**

- Preserve streamed text when tool calls occur
- Maintain chronological order of text + tool outputs
- Don't clear/replace text during tool execution

**Workaround noted:**

- Use `/verbose off`
- Disable streaming: `"streaming": "off"`

This breaks the user experience of following the agent's workflow in real-time. The text should stay visible in its original position.