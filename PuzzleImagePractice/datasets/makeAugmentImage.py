import pandas as pd
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from conf import *

def aug_img_save(train_df, show_num, save_aug=False):

    if not os.path.exists(save_augment_path):
        os.makedirs(save_augment_path)

    # 증강된 이미지 데이터 프레임 생성
    dict_augment = {'ID':[],
                   'img_path':[]}

    for i in range(1,16+1):
        dict_augment[str(i)] = [i]*len(train_df)

    # 출력할 이미지 개수 새기
    count = 0

    if save_aug == False:
        repeat = [i for i in range(show_num)]
    else:
        repeat = [i for i in range(len(train_df))]

    for index in tqdm(repeat):

        sample_df = train_df.iloc[index]

        # train 이미지 불러오기
        train_path = sample_df['img_path'].split('/')[-1]
        train_img = Image.open(data_path+'/train/'+train_path)

        width, height = train_img.size
        cell_width = width // 4
        cell_height = height // 4

        numbers = list(sample_df)[2:]

        i = 0
        dict_tile = {}

        for row in range(4):
            for col in range(4):
                left = col * cell_width
                upper = row * cell_height
                right = left + cell_width
                lower = upper + cell_height

                # 부분 이미지 추출
                tile = train_img.crop((left, upper, right, lower))
                dict_tile[numbers[i]] = tile

                i += 1

        import random
        random_numbers = random.sample(range(1, 16 + 1), 16)

        # 4x4 이미지 행렬 생성
        augment_img = Image.new("RGB", (width, height))

        # 각 부분 이미지 크기 계산
        tile_width = augment_img.width // 4
        tile_height = augment_img.height // 4

        # 16개 부분 이미지를 4x4 행렬로 배열
        i = 0
        for row in range(4):
            for col in range(4):

                random = random_numbers[i]
                tile = dict_tile[random]

                i += 1

                # 부분 이미지를 4x4 행렬 위치에 합성
                left = col * tile_width
                upper = row * tile_height
                right = left + tile_width
                lower = upper + tile_height
                augment_img.paste(tile, (left, upper, right, lower))

        # 재정려된 이미지 저장
        if save_aug == False:
            pass
        else:
            augment_name = f'AUGMENT{count:05}.jpg'
            augment_path = save_augment_path + '/' + augment_name
            augment_img.save(augment_path)  
            dict_augment['ID'].append(augment_name)
            dict_augment['img_path'].append(augment_path)

        # train 및 재정렬된 이미지 출력
        fig = plt.figure()

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(train_img)
        ax1.set_title('Train Image')
        ax1.axis('off')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(augment_img)
        ax2.set_title('Augmented Image')
        ax2.axis('off')

        if count > show_num:
          pass
        else:
            print(train_path)
            plt.show()
            print()

        count += 1

    # 재정렬한 이미지 데이터 프레임 저장
    if save_aug == False:
        pass

    else:
        augment_df = pd.DataFrame(dict_augment)
        augment_df.to_csv(data_path+'/augment.csv', index=False)
