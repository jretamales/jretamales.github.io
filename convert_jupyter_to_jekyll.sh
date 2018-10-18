#!/bin/bash
#
# Convert Jupyter notebook files (in _jupyter folder) to markdown files (in _posts folder).
#
# Arguments: 
# $1 filename (excluding extension)
# 

# Generate a filename with today's date.
filename=$(date +%Y-%m-%d)-$1

# Jupyter will put all the assets associated with the notebook in a folder with this naming convention.
# The folder will be in the same output folder as the generated markdown file.
foldername=$filename"_files"

# Do the conversion.
jupyter nbconvert ./_jupyter/$1.ipynb --template my_jekyll.tpl --to markdown --output-dir=./_posts --output=$filename

# Move the images from the jupyter-generated folder to the images folder.
echo "Moving images..."
mv ./_posts/$foldername/* ./assets/images

# Remove the now empty folder.
rmdir ./_posts/$foldername

# Go through the markdown file and rewrite image paths.
# NB: this sed command works on OSX, it might need to be tweaked for other platforms.
echo "Rewriting image paths..."
sed -i.tmp -e "s/$foldername/\/assets\/images/g" ./_posts/$filename.md

# Remove backup file created by sed command.
rm ./_posts/$filename.md.tmp

# Check if the conversion has left a blank line at the top of the file. 
# (Sometimes it does and this messes up formatting.)
firstline=$(head -n 1 ./_posts/$filename.md)
if [ "$firstline" = "" ]; then
  # If it has, get rid of it.
  tail -n +2 "./_posts/$filename.md" > "./_posts/$filename.tmp" && mv "./_posts/$filename.tmp" "./_posts/$filename.md"
fi


sed -i '1s/^/---\n/' ./_posts/$filename.md 
sed -i '1s/^/categories: data\n/' ./_posts/$filename.md  
sed -i '1s/^/date: '"$(date +%Y-%m-%d) $(date +%H:%M) +0000"'\n/' ./_posts/$filename.md 
sed -i '1s/^/title: Titulo\n/' ./_posts/$filename.md 
sed -i '1s/^/mathjax: true\n/' ./_posts/$filename.md 
sed -i '1s/^/layout: post\n/' ./_posts/$filename.md 
sed -i '1s/^/---\n/' ./_posts/$filename.md 

echo "Done converting."
