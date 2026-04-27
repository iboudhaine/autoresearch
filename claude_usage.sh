#!/bin/bash

TOKEN=$(jq -r '.claudeAiOauth.accessToken' "$HOME/.claude/.credentials.json")

curl -sS -X GET "https://api.anthropic.com/api/oauth/usage" \
     -H "Accept: application/json, text/plain, */*" \
     -H "Accept-Encoding: gzip, compress, deflate, br" \
     -H "Authorization: Bearer ${TOKEN}" \
     -H "Content-Type: application/json" \
     -H "User-Agent: claude-code/2.1.110" \
     -H "anthropic-beta: oauth-2025-04-20" \
     -H "Host: api.anthropic.com" \
     --compressed
