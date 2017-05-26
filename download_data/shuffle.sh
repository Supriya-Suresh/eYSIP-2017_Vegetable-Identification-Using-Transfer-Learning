mkdir ./test
shuf -n $1 -e * | xargs -i mv {} ./test/