#!/bin/bash

# This script downloads the images/pokemon directory from the pokerogue-assets repository
# It uses sparse-checkout to minimize download size.

TARGET_DIR="pokerogue_sprites"
REPO_URL="https://github.com/pagefaultgames/pokerogue-assets.git"
TEMP_DIR=".tmp_assets"

# Check if target already exists
if [ -d "$TARGET_DIR" ]; then
    echo "Directory '$TARGET_DIR' already exists. Skipping download."
    exit 0
fi

# Ensure git is installed
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed. Please install git and try again."
    exit 1
fi

echo "Downloading Pokemon assets from $REPO_URL..."

# Create temporary directory for sparse checkout
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR" || exit 1

# Initialize a temporary git repository
git init --quiet
git remote add origin "$REPO_URL"
git config core.sparseCheckout true
echo "images/pokemon" >> .git/info/sparse-checkout

# Pull only the required folder from the primary branches (main, master, or beta)
echo "Pulling images/pokemon folder..."
if git pull --depth 1 origin main --quiet 2>/dev/null || \
   git pull --depth 1 origin master --quiet 2>/dev/null || \
   git pull --depth 1 origin beta --quiet 2>/dev/null; then
    
    cd ..
    
    # Move the assets to the project root
    if [ -d "$TEMP_DIR/images/pokemon" ]; then
        mv "$TEMP_DIR/images/pokemon" "$TARGET_DIR"
        rm -rf "$TEMP_DIR"
        echo "Successfully imported $(ls "$TARGET_DIR" | wc -l) files into '$TARGET_DIR'."
    else
        echo "Error: Asset folder not found after pull."
        rm -rf "$TEMP_DIR"
        exit 1
    fi
else
    echo "Error: Failed to pull assets from $REPO_URL."
    cd ..
    rm -rf "$TEMP_DIR"
    exit 1
fi
