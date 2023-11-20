#!/bin/bash

# Variables
relativePath="2023/05/"

# De naam van het bestand
filename="todo_$(echo "$relativePath" | tr -d '/')"

# Combining relativePath and filename
file_path="${filename}.txt"

# Krijg de huidige datum en tijd
current_datetime=$(date)

# Schrijf de huidige datum en tijd naar het bestand
echo "$current_datetime" > "$file_path"

# Optioneel: bevestiging dat het bestand is aangemaakt
if [ -e "$file_path" ]; then
    echo "Bestand $file_path is aangemaakt met de huidige datum en tijd: '$current_datetime'."
else
    echo "Er is een probleem opgetreden bij het maken van $file_path."
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --key)
            key="$2"
            shift
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Check if the key is not empty
if [ -z "$key" ]; then
    echo "Usage: bash copy_from_cloudvps.sh --key cloudvps_password"
    exit 1
fi

# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Add connection to CloudVPS
rclone config create raw_objectstore_rclone swift auth https://identity.stack.cloudvps.com/v2.0 auth_version 2 tenant 3206eec333a04cc980799f75a593505a user CVT_read_pano key "$key"

echo "Counting files in $relativePath..."
countOutput=$(rclone size raw_objectstore_rclone:intermediate/"$relativePath")
echo "rclone size output: $countOutput"

# Copy from CloudVps to Azure Compute to the current working directory.
# It creates the "intermediate" in the current working directory.
rclone copy raw_objectstore_rclone:intermediate/"$relativePath" intermediate/"$relativePath"

# De naam van het bestand
filename="done_$(echo "$relativePath" | tr -d '/')"

# Combining relativePath and filename
file_path="${filename}.txt"

# Krijg de huidige datum en tijd
current_datetime=$(date)

# Schrijf de huidige datum en tijd naar het bestand
echo "$current_datetime" > "$file_path"

# Optioneel: bevestiging dat het bestand is aangemaakt
if [ -e "$file_path" ]; then
    echo "Bestand $file_path is aangemaakt met de huidige datum en tijd: '$current_datetime'."
else
    echo "Er is een probleem opgetreden bij het maken van $file_path."
fi

# Copy from CloudVps to Azure Compute to the current working directory.
# It creates the "intermediate" in the current working directory.
# NOTE we are not able to do rclone config create for the blob container, so we use AZ copy later
# rclone copy raw_objectstore_rclone:intermediate/"$relativePath" intermediate/"$relativePath"

# Copy from Azure Compute to Storage Account
# NOTE make sure to run "azcopy login" before this line
# echo "!!!!NOTE make sure to run "azcopy login" before this line!!!"
# azcopy copy intermediate/ "$storageAccountUrl" --recursive

# Useful command in case the azcopy does not finish successfully:azcopy jobs show <job-id> --with-status=Failed