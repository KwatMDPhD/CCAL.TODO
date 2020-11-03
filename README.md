## Install

```sh
git clone https://github.com/kwatme/kraft

pip install --editable kraft/

ls --size kraft/kraft/data/ # Make sure that the file sizes are good

# If file sizes are small, then they are pointers instead of the files themselves; so git large file system to fetch them

# Install git-lfs

# Fetch

git lfs fetch

ls --size kraft/kraft/data/ # Check the file sizes again

```
