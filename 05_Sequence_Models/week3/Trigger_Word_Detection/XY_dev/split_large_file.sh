# File: split_large_file.sh
# Author: Bhishan Poudel
# Date: Aug 17, 2019
# Purpose: split large files into smaller file
#
# Note: we use bash commands split to split and cat to join them.
# split -b1m large.txt large gives largea largeb largec etc files.
#
# we can join them later using: cat largea? > large.txt
#
# to see disk space use: du -sh large.txt
#============================================================
#
# this will break files into chunks of 40MB
fname="$1"
head="${fname%.*}"


split -b40m "$1" "$head"_

# remove large file
#rm X_dev.npy

# to join
# cat X_dev_a? > X_dev.npy
