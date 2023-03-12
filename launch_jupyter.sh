sudo apt-get update 
sudo apt-get install chromium-browser
chromium-browser --no-sandbox &
jupyter notebook --port 3002 --ip=0.0.0.0 --allow-root
