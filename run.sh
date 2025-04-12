#!/bin/bash

PROJECT_NAME="trusted-micros"

download_and_extract_model() {
    MODEL_URL="https://drive.usercontent.google.com/download?id=1YIuWQfQhqunpwu4r9pKMBHM65-V8KXA8&export=download&authuser=0&confirm=t&uuid=e85b0ec4-e3c5-454c-b4c1-cc62659577a2&at=APcmpoxHCVQgZucCp7FonzLt2miU%3A1744026522347"
    MODEL_DEST_DIR="app/raw/model"
    MODEL_DEST_FILE="$MODEL_DEST_DIR/ml_classifier.zip"
    MODEL_EXTRACTED_DIR="$MODEL_DEST_DIR/saved_model"

    mkdir -p "$MODEL_DEST_DIR"

    if [ -d "$MODEL_EXTRACTED_DIR" ]; then
        echo "Extracted folder already exists."
        return
    fi

    for attempt in 1 2; do
        [ "$attempt" -eq 2 ] && echo "Retrying download and extraction..." && rm -f "$MODEL_DEST_FILE"

        [ ! -f "$MODEL_DEST_FILE" ] && curl -# -L "$MODEL_URL" -o "$MODEL_DEST_FILE"
        if unzip -o "$MODEL_DEST_FILE" -d "$MODEL_DEST_DIR"; then
            echo "Model downloaded and extracted successfully."
            return
        fi

        [ "$attempt" -eq 2 ] && echo "Failed to extract model after retrying. Exiting." && exit 1
    done

}


stop_docker() {
    docker compose stop
    docker compose -p $PROJECT_NAME exec -T redis redis-cli FLUSHALL || true
}

stop_docker
if [ "$1" == "stop" ]; then
    echo "crawler service stopped"
else
    if [ "$1" == "build" ]; then
        download_and_extract_model
        docker compose -p $PROJECT_NAME build
    fi

    docker compose -p $PROJECT_NAME up -d
    echo "crawler service started"
fi


