# Install all dependecies
pip install -r requirements.txt

# Downloading Datasets
cd src/data

# Downloading dataset - Food-101
gdown 1YpJ1bLFuFjAlhTyttZNlrUXP4TLMPaJb

# Downloading dataset - StanfordCars
gdown 1tQdRrhfjXfHnUuRJiIsP-HNSgV3CbtN3

# unzip the dataset
unzip food-101.zip
unzip stanfordcars.zip

# remove the zip files
rm food-101.zip
rm stanfordcars.zip

# remove macosx folder
rm -rf __MACOSX/

cd ../..
