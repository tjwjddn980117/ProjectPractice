import pandas as pd
from tqdm.auto import tqdm

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import matplotlib.pyplot as plt
from conf import *

def check_img_save_origin(train_df, show_num, save_origin=False):

    if not os.path.exists(save_origin_path):
        os.makedirs(save_origin_path)

    # 재정렬한 이미지 데이터 프레임 생성
    dict_origin = {'ID':[],
                   'img_path':[]}

    for i in range(1,16+1):
        dict_origin[str(i)] = [i]*len(train_df)
    
    #  dict_origin ->
    #  {'ID': [],
    #   'img_path': [],
    #   '1': [1, 1, 1, ..., 1], '1'이 len(train_df)만큼 반복
    #   '2': [2, 2, 2, ..., 2], '2'가 len(train_df)만큼 반복
    #   ...
    #   '16': [16, 16, 16, ..., 16] '16'이 len(train_df)만큼 반복}

    # 출력할 이미지 개수 새기
    count = 0

    # 미리 갯수를 맞춰놓기.
    # save origin 이면 train_df만큼, 
    # save origin 이 아니면, 정해놓은 show_num만큼.
    if save_origin == False:
        repeat = [i for i in range(10,show_num)]
    else:
        repeat = [i for i in range(len(train_df))]

    for index in tqdm(repeat):

        sample_df = train_df.iloc[index]    

        # train 이미지 불러오기
        # train_img is the path
        # ex) ..data/train/TRAIN_00000.jpg
        train_path = sample_df['img_path'].split('/')[-1]
        train_img = Image.open(data_path+'/train/'+train_path)
        raw_img = Image.open(data_path+'/train/'+train_path) 

        # train 이미지에 숫자 표기
        #draw = ImageDraw.Draw(train_img)    
        width, height = train_img.size  
        cell_width = width // 4
        cell_height = height // 4   
        #font_size = 50
        #font = ImageFont.truetype("LiberationSans-Regular.ttf", font_size)  

        # 데이터에 저장되어있는 숫자들. 
        # ex) 8,1,16,12,5,10,...
        numbers = list(sample_df)[2:]   
        #for i, number in enumerate(numbers):
        #    row = i // 4
        #    col = i % 4
        #    x = col * cell_width + (cell_width - font_size) // 2
        #    y = row * cell_height + (cell_height - font_size) // 2
        #    draw.text((x, y), str(number), fill="red", font=font)   

        # 정렬된 이미지 생성 및 저장
        i = 0
        dict_tile = {}  
        for row in range(4):
            for col in range(4):
                left = col * cell_width
                upper = row * cell_height
                right = left + cell_width
                lower = upper + cell_height 
                # 부분 이미지 추출
                tile = raw_img.crop((left, upper, right, lower))
                dict_tile[numbers[i]] = tile    
                i += 1  

        # 4x4 이미지 행렬 생성
        origin_img = Image.new("RGB", (width, height))  
        # 각 부분 이미지 크기 계산
        tile_width = origin_img.width // 4
        tile_height = origin_img.height // 4    

        # 16개 부분 이미지를 4x4 행렬로 배열
        i = 1
        for row in range(4):
            for col in range(4):
                tile = dict_tile[i] 
                i += 1  
                # 부분 이미지를 4x4 행렬 위치에 합성
                left = col * tile_width
                upper = row * tile_height
                right = left + tile_width
                lower = upper + tile_height
                origin_img.paste(tile, (left, upper, right, lower)) 

        # 재정려된 이미지 저장
        if save_origin == False:
            pass

        else:
            origin_name = f'ORIGIN_{count:05}.jpg'
            origin_path = save_origin_path + '/' + origin_name
            origin_img.save(origin_path) 
            dict_origin['ID'].append(origin_name)
            dict_origin['img_path'].append(origin_path)  

        # train 및 재정렬된 이미지 출력
        fig = plt.figure()  
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(train_img)
        ax1.set_title('Train Image')
        ax1.axis('off') 
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(origin_img)
        ax2.set_title('Original Image')
        ax2.axis('off') 

        if count > show_num:
            pass
        else:
            print(train_path)
            plt.show()
            print()  
        count += 1

    # 재정렬한 이미지 데이터 프레임 저장
    if save_origin == False:
        pass

    else:
        origin_df = pd.DataFrame(dict_origin)
        origin_df.to_csv(data_path+'/origin.csv', index=False)