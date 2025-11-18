#!/usr/bin/env bash
set -euo pipefail

# Bootstrap helper to fetch large assets kept outside of Git.
# Supply Google Drive (or any HTTP-accessible) URLs via environment variables:
#   HOPE_AGENT_DATA_URL  -> archive with data/clean/*.csv
#   HOPE_AGENT_DB_URL    -> DuckDB database file or archive
#   HOPE_AGENT_VEC_URL   -> Chroma collection archive

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
DB_DIR="${ROOT_DIR}/db"
VEC_DIR="${ROOT_DIR}/vec"

mkdir -p "${DATA_DIR}" "${DB_DIR}" "${VEC_DIR}"

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

download_file() {
    local url="$1"
    local output="$2"

    if command_exists curl; then
        curl -L "$url" -o "$output"
    elif command_exists wget; then
        wget -O "$output" "$url"
    elif command_exists gdown; then
        gdown --fuzzy "$url" -O "$output"
    else
        echo "✖ Neither curl, wget, nor gdown is installed. Install one to enable automatic downloads."
        return 1
    fi
}

unpack_if_archive() {
    local file="$1"
    local destination="$2"

    case "$file" in
        *.zip)
            unzip -o "$file" -d "$destination"
            ;;
        *.tar.gz|*.tgz)
            tar -xzf "$file" -C "$destination"
            ;;
        *.tar)
            tar -xf "$file" -C "$destination"
            ;;
        *)
            # Assume it's already the expected artifact; move into place.
            mv "$file" "$destination"
            ;;
    esac
}

fetch_asset() {
    local url="$1"
    local destination="$2"
    local label="$3"

    if [[ -z "$url" ]]; then
        echo "• Skipping ${label}: no URL provided (set HOPE_AGENT_${label}_URL)."
        return
    fi

    local tmpfile
    tmpfile="$(mktemp)"
    echo "→ Downloading ${label} from ${url}"
    if ! download_file "$url" "$tmpfile"; then
        echo "✖ Failed to download ${label}. Retrieve it manually (see data/README.md and vec/README.md)."
        rm -f "$tmpfile"
        return
    fi

    echo "→ Extracting ${label} to ${destination}"
    mkdir -p "$destination"
    if ! unpack_if_archive "$tmpfile" "$destination"; then
        echo "✖ Could not unpack ${label}. File left at ${tmpfile} for inspection."
        return
    fi

    rm -f "$tmpfile"
    echo "✓ ${label} ready."
}

fetch_asset "${HOPE_AGENT_DATA_URL:-}" "${DATA_DIR}" "DATA"
fetch_asset "${HOPE_AGENT_DB_URL:-}" "${DB_DIR}" "DB"
fetch_asset "${HOPE_AGENT_VEC_URL:-}" "${VEC_DIR}" "VEC"

echo ""
echo "Bootstrap complete. Review the README files in data/ and vec/ for expected layouts."
