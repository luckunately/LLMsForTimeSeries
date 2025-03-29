#!/bin/bash

# Directory containing the scripts
SCRIPT_DIR="./script"

# Loop through all .sh files in the directory and execute them
for script in "$SCRIPT_DIR"/*.sh; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        bash "$script"
    else
        echo "No scripts found in $SCRIPT_DIR"
    fi
done