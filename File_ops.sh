#!/bin/bash

echo "File Operations:"
echo "1. Create file"
echo "2. Delete file"
echo "3. Rename file"
echo "4. Search word in file"
read -p "Enter your choice: " choice

case $choice in
#  1)
    read -p "Enter filename to create: " fname
    touch "$fname"
    echo "File $fname created."
    ;;
  #2)
    read -p "Enter filename to delete: " fname
    rm -i "$fname"
    ;;
 # 3)
    read -p "Enter current filename: " oldname
    read -p "Enter new filename: " newname
    mv "$oldname" "$newname"
    echo "File renamed to $newname"
    ;;
  #4)
    read -p "Enter filename to search in: " fname
    read -p "Enter word to search: " word
    grep --color "$word" "$fname"
    ;;
  *)
    echo "Invalid choice"
    ;;
esac

#2. Shell Script for Process Creation
#!/bin/bash

echo "Starting a background process (sleep 60)..."
sleep 60 &
pid=$!
echo "Process started with PID $pid"

3. Shell Script for Process Monitoring
#!/bin/bash

read -p "Enter PID to monitor: " pid

if ps -p $pid > /dev/null; then
  echo "Monitoring PID: $pid"
  top -b -n1 | grep "$pid" | awk '{print "PID: "$1", CPU: "$9"%, MEM: "$10"%"}'
else
  echo "Process with PID $pid not running."
fi
