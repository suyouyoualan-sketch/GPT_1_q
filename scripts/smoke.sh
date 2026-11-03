#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
smoke_dir="$(mktemp -d /tmp/quant-harbor-smoke.XXXXXX)"
server_pid=""

cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid"
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf "$smoke_dir"
}
trap cleanup EXIT

export DATABASE_URL="sqlite:///$smoke_dir/trader.db"
export ADMIN_TOKEN="smoke-test-token"
export TRADING_MASTER_KEY="smoke-test-encryption-key"
export ALLOW_LIVE_TRADING="false"

.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8765 >"$smoke_dir/server.log" 2>&1 &
server_pid=$!

ready="false"
for _ in {1..100}; do
  if curl -fsS http://127.0.0.1:8765/api/health >"$smoke_dir/health.json" 2>/dev/null; then
    ready="true"
    break
  fi
  sleep 0.2
done

if [[ "$ready" != "true" ]]; then
  cat "$smoke_dir/server.log" >&2
  exit 1
fi

curl -fsS http://127.0.0.1:8765/ | grep -q "Quant Harbor"
curl -fsS http://127.0.0.1:8765/ | grep -q "市场行情"
curl -fsS -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://127.0.0.1:8765/api/overview >"$smoke_dir/overview.json"
curl -fsS -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://127.0.0.1:8765/api/library >"$smoke_dir/library.json"
curl -fsS -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"烟雾测试策略","description":"临时验证","category":"我的策略","entry_summary":"价格高于 100","exit_summary":"价格低于 90","rule":{"entry":{"left":{"field":"close"},"op":">","right":{"value":100}},"exit":{"left":{"field":"close"},"op":"<","right":{"value":90}}}}' \
  http://127.0.0.1:8765/api/library >"$smoke_dir/custom-template.json"

unauthorized_status="$(curl -sS -o /dev/null -w '%{http_code}' http://127.0.0.1:8765/api/meta)"
if [[ "$unauthorized_status" != "401" ]]; then
  echo "Expected unauthenticated /api/meta to return 401; got $unauthorized_status" >&2
  exit 1
fi

grep -q '"ok":true' "$smoke_dir/health.json"
grep -q '"accounts":0' "$smoke_dir/overview.json"
grep -q '经典双均线' "$smoke_dir/library.json"
grep -q '动量 Alpha' "$smoke_dir/library.json"
grep -q '烟雾测试策略' "$smoke_dir/custom-template.json"
echo "HTTP smoke test passed"
