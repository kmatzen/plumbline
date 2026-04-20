#!/usr/bin/env bash
# Mint a short-lived STS session token for use on a rental GPU box.
#
# Usage:
#   scripts/gpu_box_session_token.sh
#
# Prerequisites (on the LAPTOP, once):
#   aws configure --profile plumbline-gpu-cache
#     AWS Access Key ID:     <the long-lived key from IAM create-access-key>
#     AWS Secret Access Key: <the long-lived secret>
#     Default region name:   us-west-2
#     Default output format: json
#
# Output: three `export` lines + `AWS_DEFAULT_REGION`. Paste on the
# rental GPU box to give it 12h of access scoped to the
# plumbline-bench S3 cache.
#
# Why session tokens, not long-lived keys on the rental box:
#   - If the rental box is compromised (shared, spot-reclaimed, image
#     leaked), the exposure window is bounded to DURATION.
#   - The laptop-resident long-lived key stays on the laptop.
#
# Env overrides:
#   PLUMBLINE_S3_PROFILE  — AWS CLI profile name (default: plumbline-gpu-cache)
#   PLUMBLINE_S3_DURATION — seconds (default: 43200 = 12h; max 129600 = 36h)

set -euo pipefail

PROFILE="${PLUMBLINE_S3_PROFILE:-plumbline-gpu-cache}"
DURATION="${PLUMBLINE_S3_DURATION:-43200}"

if ! aws --profile "$PROFILE" sts get-caller-identity >/dev/null 2>&1; then
    cat >&2 <<ERR
error: AWS profile '$PROFILE' is not configured or the credentials are invalid.

Configure it once with:
    aws configure --profile $PROFILE

using the access-key pair created for the IAM user 'plumbline-gpu-cache'.
ERR
    exit 1
fi

read -r AKID SECRET SESSION EXPIRES < <(
    aws --profile "$PROFILE" sts get-session-token \
        --duration-seconds "$DURATION" \
        --query 'Credentials.[AccessKeyId,SecretAccessKey,SessionToken,Expiration]' \
        --output text
)

cat <<EOF
# Session token for plumbline S3 cache (bucket: plumbline-bench, us-west-2).
# Expires: ${EXPIRES}
# Paste on the rental GPU box:

export AWS_ACCESS_KEY_ID='${AKID}'
export AWS_SECRET_ACCESS_KEY='${SECRET}'
export AWS_SESSION_TOKEN='${SESSION}'
export AWS_DEFAULT_REGION=us-west-2

# Quick smoke test:
#   aws s3 ls s3://plumbline-bench/
EOF
