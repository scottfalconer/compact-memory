# Querying Playbook

This short guide collects practical tips for crafting search queries when working with the Gist Memory agent. The agent simply embeds the raw query text and finds the closest prototypes and memories. For simple keyword style queries that behaviour is fine. However, when your notes follow a structured format, you can bias retrieval by embedding a similarly structured query string.

## 1. Query‑time templating

If your memories store fields such as `WHO`, `WHAT`, and `WHEN`, you can replicate that layout when embedding a question. Re‑render the query as a partial template so the slot tokens line up with those used in the memories. For example, when a user asks:

```
Who talked about budget in May?
```

Render the query for embedding like this:

```
WHO: ?; WHAT: budget; WHEN: May 2025; CONTENT: ?
```

Feeding the templated string to `embed_text` makes the query vector more similar to memories that share those `WHO`, `WHAT`, and `WHEN` tokens even without explicit filtering.

If a user only types a few keywords like `"budget meeting"`, skip the template and embed the raw text – the default ranking still works well for free text searches.

