# Install all dependecies
pip install -r requirements.txt

# Downloading Datasets
cd src/data

# Downloading dataset - Food-101
echo "Downloading dataset - Food-101"
gdown 1YpJ1bLFuFjAlhTyttZNlrUXP4TLMPaJb

# Downloading dataset - StanfordCars
echo "Downloading dataset - StanfordCars"
gdown 1tQdRrhfjXfHnUuRJiIsP-HNSgV3CbtN3

# Downloading dataset - StanfordDogs
echo "Downloading dataset - StanfordDogs"

# Downloading dataset - StanfordFlowers
echo "Downloading dataset - OxfordFlowers"
gdown 1SBXiscR21AMygh4dPdIGUg6XE617oGP1

# Downloading dataset - CUB-200-2011
echo "Downloading dataset - CUB-200-2011"
gdown 1b-lwpTtRHkgd1bMLIYZ4gzBDj4bQOLIN

# Downloading dataset - nabirds
echo "Downloading dataset - nabirds"
gdown 1BNrjVuuvE21GAIziVslbZWIIiXW0sSBc


# unzip the dataset
unzip food-101.zip
unzip stanfordcars.zip
unzip stanford-dogs.zip
unzip oxford-flowers.zip
unzip CUB_200_2011.zip
unzip nabirds.zip

# remove the zip files
rm *.zip

# remove macosx folder
rm -rf __MACOSX/

cd ../..
