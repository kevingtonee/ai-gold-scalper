# Security Rotation Required

A real OpenAI API key was present in `config/config.json` and has been removed from the working tree.
Rotate/revoke that key immediately in the OpenAI dashboard. Do not reuse it.

Use `OPENAI_API_KEY` from the environment or a local, gitignored `config/config.json`.
Do not commit API keys, tunnel credentials, or account credentials.
