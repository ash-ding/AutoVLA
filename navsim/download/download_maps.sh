#!/bin/bash

set -euo pipefail

DEST="./data"
MAPS_URL="https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip"
MAPS_ARCHIVE="nuplan-maps-v1.1.zip"

mkdir -p "$DEST"

if ! command -v aria2c >/dev/null 2>&1; then
    echo "Error: aria2c is required but not installed." >&2
    exit 1
fi

aria2c --dir="$DEST" --out="$MAPS_ARCHIVE" "$MAPS_URL"
unzip "$DEST/$MAPS_ARCHIVE" -d "$DEST"
rm "$DEST/$MAPS_ARCHIVE"

if [ -d "$DEST/nuplan-maps-v1.0" ]; then
    mv "$DEST/nuplan-maps-v1.0" "$DEST/maps"
    echo "Success: Maps moved to $DEST/maps"
else
    echo "Error: Directory $DEST/nuplan-maps-v1.0 not found."
fi
