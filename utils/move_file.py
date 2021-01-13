import os
import shutil

# 경로 변경 
# target = "/home/bong20/data/iitp/train_data/track4/train_data"
# result = "/home/bong20/data/iitp/train_data/track4/train"

target = "/home/bong20/data/iitp/track4/train_data"
result = "/home/bong20/data/iitp/track4/train"

# 파일 포멧 형식 추가 

for dirName, subdirList, fileList in os.walk(target):
    if os.path.basename(dirName) != "image":
        continue

    p_dir = os.path.dirname(dirName)
    data_dir = os.path.basename(p_dir)
    result_dir = os.path.join(result, data_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for filename in fileList:
        _, ext = os.path.splitext(filename)
        if ext.lower() not in [".jpg", ".jpeg", ".png", "bmp"]:
            continue

        shutil.copy(os.path.join(dirName, filename), os.path.join(result_dir, filename))
        # 체크부분
        print(result_dir)
