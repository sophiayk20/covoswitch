#!/bin/bash

# Check if the Common Voice file exists
if [ ! -f "./validation-ds/en_ar/en_ar-usedfiles.txt" ]; then
    echo "Error: input file not found."
    exit 1
fi

# Name of the output tar file
output_tar="validation-files.tar"

# Ensure the output tar file doesn't already exist
if [ -f "$output_tar" ]; then
    echo "Error: $output_tar already exists. Please remove or rename it."
    exit 1
fi

# Create a tar archive containing files listed in used-en_de.txt
tar -cvzf "$output_tar" -T "./validation-ds/en_ar/en_ar-usedfiles.txt"

echo "Tarring completed. All files are archived into $output_tar."