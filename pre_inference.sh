#!/bin/bash

# Set execution_time variable
execution_time=$(date +'%Y-%m-%d_%H_%M_%S')

# Set customer, storage_account, and batch_size variables
customer="data-office"
storage_account="todo"
number_of_batches=5
subscription="todo"
year="todo"


# Define source and destination URLs
destination_url="https://${storage_account}.blob.core.windows.net/${customer}-input-structured/"
if [ "$customer" = "hist" ]; then
    source_url="https://${storage_account}.blob.core.windows.net/${customer}-input/${year}/*"
    # AzCopy command to copy contents from source to destination with the execution_time folder
    azcopy copy "${source_url}" "${destination_url}${execution_time}/${year}/" --overwrite=ifSourceNewer --recursive
else
    source_url="https://${storage_account}.blob.core.windows.net/${customer}-input/*"
    azcopy copy "${source_url}" "${destination_url}${execution_time}/" --overwrite=ifSourceNewer --recursive
fi

# Check if the AzCopy operation was successful
if [ $? -eq 0 ]; then
    echo "Files moved successfully from ${source_url} to ${destination_url}${execution_time}/"

    # Image file extensions to filter
    IMG_FORMATS=('bmp' 'dng' 'jpeg' 'jpg' 'mpo' 'png' 'tif' 'tiff' 'webp' 'pfm')

    # Constructing query for all specified extensions
    query=""
    for extension in "${IMG_FORMATS[@]}"
    do
        query+="ends_with(name, '.${extension}') || "
    done
    query=${query%|| }  # Remove the trailing " || "

    # Use Azure CLI to list files in the specific folder of the destination container with specified extensions
    file_list=$(az storage blob list --container-name "${customer}-input-structured" --account-name "${storage_account}" --subscription ${subscription} --prefix "${execution_time}/" --query "[?(${query})].name" --num-results 9999999 -o tsv)

    # Count the number of files in file_list
    file_count=$(echo "$file_list" | wc -l)

    # Validate number_of_batches if less than the number of files, otherwise default to 1
    if [ $file_count -lt $number_of_batches ]; then
        number_of_batches=1
        echo "Number of files is less than the number of batches"
    fi

    # Calculate files per batch and remaining files
    files_per_batch=$(( file_count / number_of_batches ))
    remaining_files=$(( file_count % number_of_batches ))

    # Create temporary batch files
    start_index=1
    for ((i = 0; i < $number_of_batches; i++)); do
        batch_size=$files_per_batch

        if [ $i -lt $remaining_files ]; then
            batch_size=$(( batch_size + 1 ))
        fi

        end_index=$((start_index + batch_size - 1))
        echo "$file_list" | sed -n "${start_index},${end_index}p" > "/tmp/${execution_time}_batch_${i}.txt"
        start_index=$((end_index + 1))
    done

    # Upload temporary batch files to the Azure Blob Storage destination
    for ((i = 0; i < $number_of_batches; i++)); do
        azcopy copy "/tmp/${execution_time}_batch_${i}.txt" "${destination_url}inference_queue/${execution_time}_batch_${i}.txt"
    done

    echo "Files split into batches and uploaded to inference_queue"

#    # Delete files inside source_url
#    az storage blob delete-batch --source "${source_url}"
#    echo "Files inside ${source_url} deleted"
else
    echo "Files failed to move from ${source_url} to ${destination_url}inference_queue/"
fi
