#!/bin/bash
# Run once after cloning to install the pre-push safety check.
# Usage: bash install_hooks.sh

HOOK_SRC=".git/hooks/pre-push"

cat > "$HOOK_SRC" << 'EOF'
#!/bin/bash
BANNED_PATTERNS=(
    "pokeapi_data/"
    "split_sprites/"
    "pokerogue_sprites/"
    "Classification/checkpoints/"
    "Classification/results/"
    "Classification/.index_cache.pkl"
    ".conda/"
)

FOUND=0
for pattern in "${BANNED_PATTERNS[@]}"; do
    matches=$(git ls-files | grep "$pattern")
    if [ -n "$matches" ]; then
        echo "ERROR: The following files match '$pattern' and should not be pushed:"
        echo "$matches" | sed 's/^/  /'
        FOUND=1
    fi
done

if [ "$FOUND" -eq 1 ]; then
    echo ""
    echo "Push aborted. Run 'git rm -r --cached <file>' to untrack them, then commit."
    exit 1
fi

echo "Pre-push check passed."
exit 0
EOF

chmod +x "$HOOK_SRC"
echo "Pre-push hook installed."
