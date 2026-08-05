SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SETUP_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Missing $ENV_FILE. Copy .env.example to .env and fill in your Onshape API credentials." >&2
    return 1 2>/dev/null || exit 1
fi

set -a
source "$ENV_FILE"
set +a
