# Query Tips

The agent embeds the query text and retrieves the closest prototypes and memories. Simple keyword searches work well, but when your notes follow a structured format you can bias retrieval by embedding a similarly structured string.

## Query-time templating

If memories use fields like `WHO`, `WHAT` and `WHEN`, render the question in the same layout before embedding:

```
WHO: ?; WHAT: budget; WHEN: May 2025; CONTENT: ?
```

Embedding this template aligns the tokens with the stored memories, improving ranking without explicit filters. For short free-text searches such as "budget meeting" you can simply embed the raw text.
