# Configuration dictionary
class ExpConfig:
    def __init__(self):
        self.data_folder = 'C:/Users/Seo/Downloads/cFlow-master/data/our_LIDC-IDRI_dataset/'
        self.transform = {
            'train': None,  # 여기에 적절한 transform을 설정하세요
            'val': None,    # 여기에 적절한 transform을 설정하세요
            'test': None    # 여기에 적절한 transform을 설정하세요
        }
        self.train_batch_size =  32  # 원하는 배치 사이즈로 수정
        self.val_batch_size = 16     # 원하는 배치 사이즈로 수정
        self.test_batch_size = 16    # 원하는 배치 사이즈로 수정
        self.num_w = 4          # 사용할 워커 수


exp_config = ExpConfig()

# Configuration dictionary
class ExpConfigMSMRI:
    def __init__(self):
        self.data_folder = 'C:/Users/Seo/Downloads/cFlow-master/data/our_MS-MRI_dataset_split/'
        self.transform = {
            'train': None,  # 여기에 적절한 transform을 설정하세요
            'val': None,    # 여기에 적절한 transform을 설정하세요
            'test': None    # 여기에 적절한 transform을 설정하세요
        }
        self.train_batch_size =  32  # 원하는 배치 사이즈로 수정
        self.val_batch_size = 16     # 원하는 배치 사이즈로 수정
        self.test_batch_size = 16    # 원하는 배치 사이즈로 수정
        self.num_w = 4          # 사용할 워커 수


exp_config_ms_mri = ExpConfigMSMRI()
