#!/bin/bash

# Define paths to the environment files
WITH_BUILD_PATH="conda/environment_with_build.yml"
WITHOUT_BUILD_PATH="conda/environment_without_build.yml"

# Function to create and activate conda environment
create_and_activate_env() {
    local env_file=$1

    echo "Creating the conda environment from $env_file..."
    conda env create -f "$env_file"

    # Check if the environment was created successfully
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create the conda environment using $env_file!"
        return 1
    fi

    # Extract environment name from the YAML file
    ENV_NAME=$(head -n 1 "$env_file" | cut -d ' ' -f 2)
    echo "Activating the environment: $ENV_NAME"
    conda activate "$ENV_NAME"

    # Confirm activation
    if [ $? -eq 0 ]; then
        echo "Environment $ENV_NAME activated successfully."
    else
        echo "Error: Failed to activate the environment!"
        return 1
    fi

    return 0
}

# Try creating the environment with build specifications
if [ -f "$WITH_BUILD_PATH" ]; then
    create_and_activate_env "$WITH_BUILD_PATH"
    if [ $? -eq 0 ]; then
        echo "Environment created successfully with build specifications."
        exit 0
    fi
else
    echo "Warning: $WITH_BUILD_PATH not found!"
fi

# Fallback to creating the environment without build specifications
if [ -f "$WITHOUT_BUILD_PATH" ]; then
    create_and_activate_env "$WITHOUT_BUILD_PATH"
    if [ $? -eq 0 ]; then
        echo "Environment created successfully without build specifications."
        exit 0
    fi
else
    echo "Error: $WITHOUT_BUILD_PATH not found!"
    exit 1
fi

echo "All done!"
