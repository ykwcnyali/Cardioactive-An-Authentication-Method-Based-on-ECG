import timefrom loguru import logger#import Recognizesystemimport FeatureExtractorimport ModelTrainnerimport osdef get_filenames_using_os(folder_path):    filenames = os.listdir(folder_path)    return filenamesdef DataProcess():    namelist = get_filenames_using_os(os.getcwd() + '\data')    for name in namelist:        if name[0:4] != 'tesr':            FeatureExtractor.main(name)            print(f'Subject {name} data processed.')            print('------------------------------------')    print('-------------------------------------------------------')    print('All Exist Data Extracted.')    print('-------------------------------------------------------')def ModelTrain():    ModelTrainner.Train()    print('-------------------------------------------------------')    print('Model Trained Successfully.')    print('-------------------------------------------------------')#DataProcess()ModelTrainner.main()