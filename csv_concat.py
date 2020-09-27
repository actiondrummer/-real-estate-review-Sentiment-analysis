import os
import pandas as pd

def csv_add():###리뷰 폴더만 csv 합치기
    #폴더안의 파일들 리스트형태로 변경
    file_list=os.listdir("./review/")
    score_list = []
    desc_list = []
    # 추출해야되는 데이터
    extraction_data_score = ['score', 'trafficScore', 'careScore', 'residentScore', 'aroundScore']
    extraction_data_desc = ['desc', 'trafficDesc', 'careDesc', 'residentDesc', 'aroundDesc']

    for file in file_list:
        test_data=pd.read_csv("./review/"+file)
        for i in range(len(test_data)):
            for j in extraction_data_score:
                score=test_data[j].iloc[i]
                score_list.append(score)
            for j in extraction_data_desc:
                txt = test_data[j].iloc[i]
                desc_list.append(txt)
    data={"desc":desc_list,"score":score_list}
    test=pd.DataFrame(data,columns=["desc","score"])
    test.to_csv("output.csv")


if __name__=="__main__":
    csv_add()


