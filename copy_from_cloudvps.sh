# Variables
#relativePath="2016/03/17/"
relativePath="2016/03/21/"
storageAccountUrl="https://cvodataweupgwapeg4pyiw5e.blob.core.windows.net/panorama"

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
# NOTE we are not able to do rclone config create for the blob container, so we use AZ copy later
rclone copy raw_objectstore_rclone:intermediate/"$relativePath" intermediate/"$relativePath"

# Copy from Azure Compute to Storage Account
# NOTE make sure to run "azcopy login" before this line
echo "!!!!NOTE make sure to run "azcopy login" before this line!!!"
azcopy copy intermediate/ "$storageAccountUrl" --recursive

# Useful command in case the azcopy does not finish successfully:azcopy jobs show <job-id> --with-status=Failed