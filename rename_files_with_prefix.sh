directory="./data/RTK/GT/asphalt/bad"
prefix="3_"

for file in "$directory"/*; do
  if [ -f "$file" ]; then
    filename=$(basename -- "$file")
    mv "$file" "$directory/$prefix$filename"
  fi
done