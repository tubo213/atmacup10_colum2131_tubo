import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import create_arai_feature,create_bert_feature,create_universal_feature,create_calor_img,create_calor_feature

def main():
    create_universal_feature.main()
    create_calor_img.main()
    create_calor_feature.main()
    create_arai_feature.main()
    create_bert_feature.main()
    
if __name__ == '__main__':
    main()