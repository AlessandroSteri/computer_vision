from gte.fex import batch_extractor

def main():
    img_path = './DATA/flickr30k_images/'
    # img_path = './img'
    pickle_name = "features.pck"
    batch_extractor(img_path, pickled_db_path=pickle_name)




if __name__ == '__main__':
    main()
