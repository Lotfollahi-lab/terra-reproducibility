#!/bin/bash

source_dir=$(pwd)
config_file=$1
file_type=$2

target_dir=/lustre/scratch126/cellgen/team361/DATASETS/bronze/$1

mkdir -p $target_dir

echo "Copying data... from $source_dir to $target_dir"
echo "FROM: $source_dir"
echo "TO:   $target_dir"
case "$file_type" in
    zip)
        cp -r "$source_dir"/*.zip "$target_dir"
        ;;
    folder)
        cp -r "$source_dir"/* "$target_dir"
        ;;
    *)
        echo "Unsupported file type: $file_type"
        exit 1
        ;;
esac

echo $(ls -l $target_dir)
