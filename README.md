## Download

```sh
git clone https://github.com/kwatme/kraft
```

Check that the data file sizes are good.

```sh
ls --size kraft/kraft/data/
```

If the data file sizes are small (they are pointers to the files in the GitLFS server), fetch them with GitLFS.

Install git-lfs

```sh
cd kraft/

git lfs fetch

cd ..
```

Confirm that the data file sizes are good.

```sh
ls --size kraft/kraft/data/
```

## Install

```sh
pip install --editable kraft/
```
