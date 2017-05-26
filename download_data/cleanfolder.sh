# Delete all corrupted jpeg's
find . -type f -iname "*.jpg" -o -iname "*.jpeg"| xargs jpeginfo -c | grep -E "WARNING|ERROR" | cut -d " " -f 1 | xargs rm -r
# Delete all files which dont end in .jpg
find . -type f ! -name '*.jpg' -delete
# Convert to inception required format (300x300) pixels with padding (parallel 6 images at a time)
ls -1 *.jpg | parallel -j 6 --eta convert '{}' -resize 300x300 -gravity center -background '"rgb(255,255,255)"' -extent 300x300 '{.}.jpg'