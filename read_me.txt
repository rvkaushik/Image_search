Steps to setup and run search Engine

I have used the dataset provided by :
http://tamaraberg.com/street2shop/wheretobuyit/photos.tar

1. Data Preparation Step:
    a. Under data folder run wget http://tamaraberg.com/street2shop/wheretobuyit/photos.tar
    b. tar -xvf photos.tar
    c. run wget  http://tamaraberg.com/street2shop/wheretobuyit/meta.zip for meta info
    d. unzip meta.zip

2. Run image_downloader.py - This will create a tran and test dataset for 200 images in each category.
3. If you want to train a predefined model run train.py, ignore this if you already have the trained model
4. Run Feature Extractor , it will save extracted feature in features directory
5. Create a test folder and put jpg images to test in that directory.
6. Run query_images.py with test folder as argument. Example : python query_images.py -ti data/test
7. For each test Image in folder _output.jpg is created, which is combination of test image and top 4 search results.

